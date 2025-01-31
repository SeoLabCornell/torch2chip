"""
Low precision attention modules
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from transformers.models.llama.configuration_llama import LlamaConfig
from src.models.lm.configuration_retnet import RetNetConfig
from src.module.base import _QBase, _QBaseLinear
from src.module.ops import FloatMatMul, BatchIntMatMul, BatchHeadIntMatMul
from src.module.fuse import MulShift
from src.models.lm.retnet import MultiScaleRetention

from timm.layers import trunc_normal_
from timm.models.swin_transformer import get_relative_position_index

from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache

class QAttention(nn.Module):
    def __init__(
            self,
            dim:int,
            num_heads, 
            qkv_bias=False,
            qk_norm=False, 
            attn_drop=0.0,
            proj_drop=0.0,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        assert dim % num_heads == 0,"dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # attention scale
        self.scale = self.head_dim ** (-0.5)

        self.qkv = _QBaseLinear(dim, int(dim*3), bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        # dropout
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = _QBaseLinear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_scale = MulShift()
        self.qkv_deq = MulShift()

        self.attn_scale = MulShift()
        self.attn_scale.scale.data.copy_(self.scale)

        # quantizers 
        self.xq = _QBase(nbit=8)
        self.qqkv = _QBase(nbit=8)
        self.qproj = _QBase(nbit=8)

        # training flag
        self.train_flag = True

        # matmul operator
        self.qk = BatchHeadIntMatMul(nbit=8)
        self.attnv = FloatMatMul(nbit=8)
    
    def inference(self):
        self.train_flag = False
        
        self.qqkv.inference()
        self.qkv.wq.inference()
        self.xq.inference()
        self.qkv.inference()
        self.qproj.inference()
        self.proj.inference()

    def trainFunc(self, q, k, v):
        attn = q @ k.transpose(-2, -1)  # out dim = token x token
        attn = self.attn_scale(attn)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v                    # out dim = token x head_dim
        x = self.qproj(x)
        return x
    
    def evalFunc(self, q, k, v):
        # q, k, v = q.to(torch.int8), k.to(torch.int8), v.to(torch.int8)
        attn = self.qk(q, k)
        attn = self.attn_scale(attn)

        attn = F.softmax(attn, dim=-1)
        attn = attn.mul(255.).round()

        attn = self.attn_drop(attn)

        x = self.attnv(attn, v)
        x = self.qproj(x)

        return x.to(torch.float32)
    
    def forward(self, x:torch.Tensor):
        B, N, C = x.shape

        x = self.xq(x)
        qkv = self.qkv(x)
        qkv = self.qqkv(qkv)

        # reshape
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # reshape to (qkv), batch, num_heads, token, head_dim
        
        q, k, v = qkv.unbind(0)         # batch, num_heads, token, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        if self.train_flag:
            x = self.trainFunc(q, k, v)
        else:
            x = self.evalFunc(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.qkv_deq(x)
        x = self.proj_drop(x)
        return x

class QWindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports shifted and non-shifted windows.

    Full precision version is adopted from timm. 
    Quantizers are added for T2C
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim = None,
            window_size = 7,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            window_size: The height and width of the window.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w), persistent=False)

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # quantizers
        self.xq = _QBase(nbit=32)
        self.qqkv = _QBase(nbit=32)
        self.qproj = _QBase(nbit=32)

        # training flag
        self.train_flag = True

        # scaler
        self.qkv_deq = MulShift()
        self.attn_scale = MulShift()
        self.attn_scale.scale.data.copy_(self.scale)

    def inference(self):
        self.train_flag = False
        
        self.qqkv.inference()
        self.qkv.wq.inference()
        self.xq.inference()
        self.qkv.inference()
        self.qproj.inference()
        self.proj.inference()

        # matmul operator
        self.qk = BatchIntMatMul(nbit=32)
        self.attnv = BatchIntMatMul(nbit=32)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)
    
    def trainFunc(self, q, k, v, B:int, N:int, mask: torch.Tensor = None):
        attn = q @ k.transpose(-2, -1)
        attn = self.attn_scale(attn)
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.qproj(x)
        return x

    def evalFunc(self, q, k, v, B:int, N:int, mask: torch.Tensor = None):
        q, k, v = q.double(), k.double(), v.double()
        attn = self.qk(q, k.transpose(-2, -1))
        
        # pos_bias is fused into the scaler
        attn = self.attn_scale(attn)
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = attn.mul(255.).round()

        attn = self.attn_drop(attn)

        x = self.attnv(attn, v)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.qproj(x)
        return x.float()
    
    def forward(self, x, mask: torch.Tensor = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        
        x = self.xq(x)
        qkv = self.qkv(x)
        qkv = self.qqkv(qkv)
        
        # reshape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.train_flag:
            x = self.trainFunc(q, k, v, B_, N, mask)
        else:
            x = self.evalFunc(q, k, v, B_, N, mask)

        x = self.proj(x)
        x = self.qkv_deq(x)
        x = self.proj_drop(x)
        return x


class QBertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.t2c_init(config)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def t2c_init(self, config):
        # quantizers
        self.xq = _QBase(nbit=32)
        self.qquery = _QBase(nbit=32)
        self.qkey = _QBase(nbit=32)
        self.qvalue = _QBase(nbit=32)
        
        self.qkv_deq = MulShift()
        self.attn_scale = MulShift()
        self.attn_scale.scale.data.copy_(1 / math.sqrt(self.attention_head_size))

        # train flag
        self.train_flag = True

    def inference(self):
        self.xq.inference()
        self.qquery.inference()
        self.qkey.inference()
        self.qvalue.inference()

        self.query.inference()
        self.key.inference()
        self.value.inference()

        self.train_flag = False

        # matmul
        self.qk = BatchIntMatMul(nbit=32)
        self.attnv = BatchIntMatMul(nbit=32)
    
    def trainFunc(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
            
        hidden_states = self.xq(hidden_states)
        mixed_query_layer = self.query(hidden_states)
        # print(mixed_query_layer.mean().item())

        # low precision Q
        mixed_query_layer = self.qquery(mixed_query_layer)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key = self.key(encoder_hidden_states)
            key_layer = self.qkey(key)
            key_layer = self.transpose_for_scores(key)

            value = self.value(encoder_hidden_states)
            value_layer = self.qvalue(value)
            value_layer = self.transpose_for_scores(value)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key = self.key(hidden_states)
            key_layer = self.qkey(key)
            key_layer = self.transpose_for_scores(key_layer)

            value = self.value(hidden_states)
            value_layer = self.qvalue(value)
            value_layer = self.transpose_for_scores(value_layer)
            
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key = self.key(hidden_states)
            key_layer = self.qkey(key)
            key_layer = self.transpose_for_scores(key_layer)

            value = self.value(hidden_states)
            value_layer = self.qvalue(value)
            value_layer = self.transpose_for_scores(value_layer)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # print(attention_scores.mean().item())

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = self.attn_scale(attention_scores)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.qkv_deq(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def evalFunc(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        hidden_states = self.xq(hidden_states)

        # Q
        mixed_query_layer = self.query(hidden_states)
        mixed_query_layer = self.qquery(mixed_query_layer)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        # K
        key = self.key(hidden_states)
        key_layer = self.qkey(key)
        key_layer = self.transpose_for_scores(key_layer)

        # V
        value = self.value(hidden_states)
        value_layer = self.qvalue(value)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.qk(query_layer, key_layer.transpose(-1, -2))
        attention_scores = self.attn_scale(attention_scores)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # round the attention score to 8-bit (fixed)
        attention_probs = attention_probs.mul(255.).round()

        context_layer = self.attnv(attention_probs, value_layer)
        context_layer = self.qkv_deq(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        
        if self.train_flag:
            output = self.trainFunc(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions
            )
        else:
            output = self.evalFunc(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions
            )

        return output
    
class QLlamaAttention(LlamaAttention):
    """
    Llama Attention with Low precision operations
    """

    def __init__(self, config: LlamaConfig, layer_idx: int, dtype=torch.float16):
        super().__init__(config, layer_idx)

        # t2c base layer
        self.q_proj = _QBaseLinear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias).to(torch.float16)
        self.k_proj = _QBaseLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias).to(torch.float16)
        self.v_proj = _QBaseLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias).to(torch.float16)
        self.o_proj = _QBaseLinear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias).to(torch.float16)

        # batch matmul
        self.qk = BatchHeadIntMatMul(nbit=8)
    
    def manual_sdpa(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()

        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).cuda()
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # TODO: Quantizers
        attn_output = self.manual_sdpa(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=causal_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    
class QMultiScaleRetention(MultiScaleRetention):
    def __init__(self, config: RetNetConfig, gate_fn="swish", use_bias=False, tensor_parallel=False):
        super().__init__(config, gate_fn, use_bias, tensor_parallel)

        self.q_proj = _QBaseLinear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.k_proj = _QBaseLinear(self.embed_dim, self.embed_dim, bias=use_bias)
        self.v_proj = _QBaseLinear(self.embed_dim, self.value_dim, bias=use_bias)
        self.g_proj = _QBaseLinear(self.embed_dim, self.value_dim, bias=use_bias)

        self.out_proj = _QBaseLinear(self.value_dim, self.embed_dim, bias=use_bias)
