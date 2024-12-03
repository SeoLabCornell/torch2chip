"""
Test
"""
import math
import torch
import t2c_gemm

@torch.no_grad()
def test_bmm_int8():
    B = 5
    M = 197
    a = torch.randint(-128, 127, (B, M, 64), dtype=torch.int8).cuda()
    b = torch.randint(-128, 127, (B, M, 64), dtype=torch.int8).cuda()
    scale = 1.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print(a.shape)
    start.record()
    c = t2c_gemm.bmm_int8(a, b, scale)
    end.record()
    torch.cuda.synchronize()
    latency_int8 = start.elapsed_time(end)

    print(c.shape)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    c_gt = torch.bmm(a.float(), b.float().transpose(1, 2)) * scale
    end.record()

    torch.cuda.synchronize()

    latency_fp32 = start.elapsed_time(end)
    err = torch.mean((c_gt - c) ** 2)
    
    print(f"Error: {err}")

    print(f"Latency of fp32 = {latency_fp32:.3f} ms")
    print(f"Latency of int8 = {latency_int8:.3f} ms")


@torch.no_grad()
def test_bmw_int8():
    B, M, K, N = 100, 196, 384, 384
    a = torch.randint(-128, 127, (B, M, K), dtype=torch.int8).cuda()
    b = torch.randint(-128, 127, (512, K), dtype=torch.int8).cuda()
    scale = 1.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    print(a.shape)
    c = t2c_gemm.bmw_int8(a, b, scale)
    end.record()
    torch.cuda.synchronize()
    latency_int8 = start.elapsed_time(end)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    c_gt = torch.matmul(a.float(), b.float().transpose(0, 1)) * scale
    end.record()

    torch.cuda.synchronize()

    latency_fp32 = start.elapsed_time(end)
    err = torch.mean((c_gt - c) ** 2)
    
    print(f"Error: {err}")

    print(f"Latency of fp32 = {latency_fp32:.3f} ms")
    print(f"Latency of int8 = {latency_int8:.3f} ms")


@torch.no_grad()
def test_bcmm_int8():
    B = 5
    H = 6
    M = 197
    a = torch.randint(-128, 127, (B, H, M, M), dtype=torch.int8).cuda()
    b = torch.randint(-128, 127, (B, H, 64, M), dtype=torch.int8).cuda()
    scale = 1.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print(a.shape)
    start.record()
    c = t2c_gemm.bcmm_int8(a, b, scale)
    end.record()
    torch.cuda.synchronize()
    latency_int8 = start.elapsed_time(end)

    print(c.shape)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    c_gt = torch.matmul(a.float(), b.float().transpose(-1, -2)) * scale
    end.record()

    torch.cuda.synchronize()

    latency_fp32 = start.elapsed_time(end)
    err = torch.mean((c_gt - c) ** 2)
    
    print(f"Error: {err}")

    print(f"Latency of fp32 = {latency_fp32:.3f} ms")
    print(f"Latency of int8 = {latency_int8:.3f} ms")


if __name__ == "__main__":
    print("TEST")
    # test_bmw_int8()
    test_bcmm_int8()