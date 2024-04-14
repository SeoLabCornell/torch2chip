"""
Export the MatMul Results to PE
"""

import math
import torch
import torch.nn.functional as F

class MatMulWriter(object):
    def __init__(self, pe:int=4, save_path:str=""):
        self.pe = pe
        self.save_path = save_path

    def matmul_shape(self, x:torch.Tensor):
        shape = list(x.size())
        if len(shape) == 4:
            batch, channel, row, col = shape
        
        elif len(shape) == 3:
            batch, channel, row, col = shape[0], 1, shape[1], shape[2]
            x = x.view(batch, channel, row, col)

        elif len(shape) == 2:
            batch, channel, row, col = 1, 1, shape[0], shape[1]
            x = x.view(batch, channel, row, col)

        return x, batch, channel, row, col
    
    def round2pe(self, dim:int):
        if dim % self.pe != 0:
            ndiff = math.ceil(dim / self.pe)
            npad = abs(ndiff*self.pe - dim)
        else:
            npad = 0

        return npad
    
    def padxy(self, x:torch.Tensor, y:torch.Tensor):
        assert len(x.shape) == 2, "two-dim matrix only!"
        assert len(y.shape) == 2, "two-dim matrix only!"

        rx, cx = x.shape
        ry, cy = y.shape

        assert cx == ry, "inavlid matmul size!" 

        rxpad = self.round2pe(rx)
        cxpad = self.round2pe(cx)
        cypad = self.round2pe(cy)

        px = [0, cxpad, 0, rxpad]
        py = [0, cypad, 0, cxpad]

        xpad = F.pad(x, px, value=0.0)
        ypad = F.pad(y, py, value=0.0)
        return xpad, ypad
    
    def mat2pe(self, mat:torch.Tensor):
        assert len(mat.shape) == 2, "two-dim matrix only!"
        r, c = mat.shape
        
        mat4d = mat.unsqueeze(0).unsqueeze(2)
        mat4d = mat.contiguous().view(int(r/self.pe), self.pe, int(c/self.pe), self.pe)

        return mat4d.permute(0,2,1,3)

    def pe_stats(self, tensor:torch.Tensor):
        return None
    
    def tensor2pe(self, x:torch.Tensor, y:torch.Tensor):
        x, bx, chx, rx, cx = self.matmul_shape(x)
        y, by, chy, ry, cy = self.matmul_shape(y)

        for h in range(chx):
            matx = x[0, h, ...]
            maty = y[0, h, ...]

            pmatx, pmaty = self.padxy(matx, maty)
            
            pmatx = self.mat2pe(pmatx)
            pmaty = self.mat2pe(pmaty)

    def write(self, x:torch.Tensor, y:torch.Tensor):
        self.tensor2pe(x, y)