import torch
import torch.nn as nn

try: 
    import t2c_gemm
    INTMM = True
except:
    print("Torch-gemm is not installed!")
    INTMM = False

class BatchIntMatMul(nn.Module):
    def __init__(self, nbit:int):
        super().__init__()
        self.nbit = nbit
    
    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int8)
        y = y.to(torch.int8)

        z = t2c_gemm.bmm_int8(x, y, 1.0)
        return z

class BatchHeadIntMatMul(nn.Module):
    def __init__(self, nbit:int):
        super().__init__()
        self.nbit = nbit
    
    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int8)
        y = y.to(torch.int8)

        z = t2c_gemm.bcmm_int8(x, y, 1.0)
        return z

class IntActWeight(nn.Module):
    def __init__(self, nbit:int, dtype=torch.float32):
        super().__init__()
        self.register_buffer("scale", torch.ones(1, 1, 1, dtype=torch.float32))
        self.nbit = nbit
        self.dtype = dtype

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x = x.to(torch.int8)
        y = y.to(torch.int8)
        z = t2c_gemm.bmw_int8(x, y, self.scale)
        return z.to(self.dtype)
    
class FloatActWeight(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        
        z = torch.matmul(x, y.transpose(0,1))
        return z.to(x.dtype)

class FloatMatMul(nn.Module):
    def __init__(self, nbit:int):
        super().__init__()
        self.nbit = nbit

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        z = torch.matmul(x, y)
        return z