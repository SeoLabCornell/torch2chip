"""
Rescale MX
"""

import torch
import triton
import triton.language as tl

@triton.jit
def rescale_mx(
    A_ptr,
    As_ptr,
    C_ptr, 
    M, K,
    stride_am, stride_ak,
    stride_asm, stride_ask,
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)

    row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    for col_start in range(0, K, BLOCK_SIZE_K):
        col_offsets = col_start + tl.arange(0, BLOCK_SIZE_K)

        valid_rows = row_offsets < M
        valid_cols = col_offsets < K
        mask = valid_rows[:, None] & valid_cols[None, :]

        chunk_idx = col_start // BLOCK_SIZE_K

        A_tile_ptr = A_ptr + (row_offsets[:, None] * stride_am) + (col_offsets[None, :] * stride_ak)
        As_tile_ptr = As_ptr + (row_offsets[:, None] * stride_asm) + chunk_idx * stride_ask

        a = tl.load(A_tile_ptr, mask=mask, other=0.0)
        s = tl.load(As_tile_ptr, mask=valid_rows[:, None], other=0.0)

        # broadcast the shape
        s = tl.broadcast_to(s, a.shape)

        # multiply
        z = a * s

        C_tile_ptr = C_ptr + (row_offsets[:, None] * stride_cm) + (col_offsets[None, :] * stride_ck)
        tl.store(C_tile_ptr, z, mask=mask)

def mx_scale(A, As, BLOCK_SIZE_M=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8):
    assert A.shape[0] == As.shape[0], "Dimension mismatch!"
    
    M, K = A.shape
    C = torch.empty((M, K), device=A.device, dtype=torch.float32)

    stride_am, stride_ak = A.stride()
    stride_asm, stride_ask = As.stride()
    stride_cm, stride_ck = C.stride()

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * GROUP_SIZE_M,)
    rescale_mx[grid](
        A, As, C,
        M, K,
        stride_am, stride_ak,
        stride_asm, stride_ask,
        stride_cm, stride_ck,
        BLOCK_SIZE_M, BLOCK_SIZE_K, GROUP_SIZE_M
    )
    return C