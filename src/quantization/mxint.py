"""
Micro Scaling Integer format by MicroSoft

Paper: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

import torch
import torch.nn as nn
from src.module.base import _QBase
from src.quantization.observer import BaseObserver

def _reshape_to_blocks(A, axes, block_size):
    """
    Adopte the reshape function from microscaling
    https://github.com/microsoft/microxcaling/blob/main/mx/mx_ops.py#L95
    """
    if axes is None:
        raise Exception(
            "axes required in order to determine which "
            "dimension toapply block size to"
        )
    if block_size == 0:
        raise Exception("block_size == 0 in _reshape_to_blocks")

    # Fix axes to be positive and sort them
    axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
    assert all(x >= 0 for x in axes)
    axes = sorted(axes)

    # Add extra dimension for tiles
    for i in range(len(axes)):
        axes[i] += i  # Shift axes due to added dimensions
        A = torch.unsqueeze(A, dim=axes[i] + 1)

    # Pad to block_size
    orig_shape = A.size()
    pad = []
    for i in range(len(orig_shape)):
        pad += [0, 0]

    do_padding = False
    for axis in axes:
        pre_pad_size = orig_shape[axis]
        if isinstance(pre_pad_size, torch.Tensor):
            pre_pad_size = int(pre_pad_size.value)
        # Don't pad if the axis is short enough to fit inside one tile
        if pre_pad_size % block_size == 0:
            pad[2 * axis] = 0
        else:
            pad[2 * axis] = block_size - pre_pad_size % block_size
            do_padding = True

    if do_padding:
        pad = list(reversed(pad))
        A = torch.nn.functional.pad(A, pad, mode="constant")

    def _reshape(shape, reshape_block_size):
        for axis in axes:
            # Reshape to tiles if axis length > reshape_block_size
            if shape[axis] >= reshape_block_size:
                assert shape[axis] % reshape_block_size == 0
                shape[axis + 1] = reshape_block_size
                shape[axis] = shape[axis] // reshape_block_size
            # Otherwise preserve length and insert a 1 into the shape
            else:
                shape[axis + 1] = shape[axis]
                shape[axis] = 1
        return shape

    # Reshape to tiles
    padded_shape = A.size()
    reshape = _reshape(list(padded_shape), block_size)

    A = A.view(reshape)
    return A, axes, orig_shape, padded_shape

def _undo_reshape_to_blocks(A, padded_shape, orig_shape, axes):
    # Undo tile reshaping
    A = A.view(padded_shape)
    # Undo padding
    if not list(padded_shape) == list(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        A = A[slices]
    for axis in reversed(axes):
        # Remove extra dimension
        A = torch.squeeze(A, dim=axis + 1)
    return A

class MXINTObserver(BaseObserver):
    def __init__(self, nbit: int, unsigned: bool = True):
        super().__init__(nbit, unsigned)

    def get_bound(self, x:torch.Tensor):
        max_norm = float(2**(self.nbit-1) - 1) / 2**(self.nbit-2)

        # update bound
        self.lb.copy_(-max_norm)
        self.ub.copy_(max_norm)
    
    def get_shared_exp(self, x:torch.Tensor, axis=0) -> torch.Tensor:
        shared_exp, _ = torch.max(torch.abs(x), dim=axis, keepdim=True)
        shared_exp = torch.floor(torch.log2(shared_exp))
        return shared_exp

    def calculate_qparam(self, x:torch.Tensor, axis:int=2):
        scale = torch.tensor(2**(self.nbit-2), device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        shared_exp = self.get_shared_exp(x, axis=axis)
        return scale, zero_point, shared_exp
    
    def forward(self, x:torch.Tensor, axis:int=2):
        self.get_bound(x)
        scale, zero_point, shared_exp = self.calculate_qparam(x, axis)
        return scale, zero_point, shared_exp


class MXChannelWiseWeightQuantizer(_QBase):
    """
    Micro-scaling format for channel-wise weight quantization

    [WARNING]: The current implementation of MXINT8 cannot be directly employed to the INT8 matmul kernel of t2c_gemm.
    """
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = False, block_size:int=32):
        super().__init__(nbit, train_flag, unsigned)

        assert nbit in [4, 8], "According to Microsoft, current MX INT format only support 4bit and 8bit."

        self.block_size = block_size
        self.observer = MXINTObserver(nbit=self.nbit, unsigned=self.unsigned)

    def register_qparams(self):
        super().register_qparams()
        self.register_buffer("shared_exp", torch.tensor(1.0))

    def reshape(self, x:torch.Tensor):
        if len(x.shape) == 4:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [0], block_size=self.block_size
            )
        elif len(x.shape) == 2:
            x, axes, orig_shape, padded_shape = _reshape_to_blocks(
                x, [0], block_size=self.block_size
            )
        else:
            raise NotImplementedError(f"Non-supported Layer Type with the shape of {list(x.size())}!")
        return x, axes, orig_shape, padded_shape
    
    def q(self, x:torch.Tensor):
        xg, axes, orig_shape, padded_shape = self.reshape(x)
        shared_exp_axes = [x + 1 for x in axes]

        if self.train_flag:
            scale, zero_point, shared_exp = self.observer(xg, shared_exp_axes[0])

            self.scale.data = 1 / scale
            self.zero_point.data = zero_point
            self.shared_exp.data = shared_exp

        xg = xg / (2**self.shared_exp)
        xg = xg / (self.scale)

        # round to nearest
        xg = torch.sign(xg) * torch.floor(torch.abs(xg) + 0.5)
        xg = xg.clamp(self.qlb, self.qub)

        # rescale back
        xg = xg.mul(2**self.shared_exp)

        # reshape the tensor
        xg = _undo_reshape_to_blocks(xg, padded_shape, orig_shape, axes)

        if self.dequantize:
            xg = xg.mul(self.scale)

        return xg

    def trainFunc(self, x:torch.Tensor):
        xq = self.q(x)
        return xq
    
    def evalFunc(self, x: torch.Tensor):
        xq = self.trainFunc(x)
        return xq