"""
Triton-based GEMM
"""

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,

):
    # block indexes along the row and column dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # initialize the accumulator (precision = int32)
    C = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    # iterate over the block by block computation along the K dimension
    for k_start in range(0, K, BLOCK_SIZE_K):
        # a_row_ptr <=> a block of tensor with shape of [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_row_ptr = A_ptr + (row_offs[:, None] * stride_am) + ((k_start + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_ak)
        b_col_ptr = B_ptr + ((k_start + tl.arange(0, BLOCK_SIZE_K))[:, None] * stride_bk) + (col_offs[None, :] * stride_bn)

        # now load the block
        a = tl.load(a_row_ptr, mask=(row_offs[:, None] < M) & (k_start + tl.arange(0, BLOCK_SIZE_K))[None, :] < K, other=0)
        b = tl.load(b_col_ptr, mask=(col_offs[None, :] < N) & (k_start + tl.arange(0, BLOCK_SIZE_K))[:, None] < K, other=0)

        # accumulator
        C += tl.dot(a, b)

    # fetch the memory address of the block [BLOCK_SIZE_M, BLOCK_SIZE_N]
    c_ptr = C_ptr + (row_offs[:, None] * stride_cm) + (col_offs[None, :] * stride_cn)
    tl.store(c_ptr, C, mask=(row_offs[:, None] < M) & (col_offs[None, :] < N))

@triton.jit
def matmul_optimized_cache(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # program_id along the row dimension
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # creating pointers
    A_ptr = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    z = torch.empty(BLOCK_SIZE_M, BLOCK_SIZE_N)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        pass



def matmul_func(
        x:torch.Tensor, 
        y:torch.Tensor,
        BLOCK_SIZE_M:int = 64,
        BLOCK_SIZE_N:int = 64,
        BLOCK_SIZE_K:int = 32
    ):
    assert x.dtype == torch.int8
    assert y.dtype == torch.int8

    M, K = x.shape
    _, N = y.shape

    z = torch.empty((M, N), dtype=torch.int32, device=x.device)

    # default stride
    stride_xm = x.stride(0)
    stride_xk = x.stride(1)

    stride_yk = y.stride(0)
    stride_yn = y.stride(1)

    stride_zm = z.stride(0)
    stride_zn = z.stride(1)

    # configuration of the block

    grid = (
        ( (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M ),  # number of blocks in the M dimension
        ( (K + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N )   # number of blocks in the N dimension
    )

    # Launch the Triton kernel
    matmul_kernel[grid](
        x, y, z,
        M, K, N,
        stride_xm, stride_xk,
        stride_yk, stride_yn,
        stride_zm, stride_zn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return z