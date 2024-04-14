"""
Profiler of FLOPs and MACs
"""

import torch
import torch.nn as nn

from typing import List
from src.module.base import IntMatMul
from src.module.fuse import MulShift, MulQuant

class Profiler(object):
    def __init__(self, model:nn.Module):
        self.model = model
        self.profile = {}

    def get_row_col(self, shape:List):
        if len(shape) == 4:
            scale, row, col = shape[1], shape[2], shape[3]
        elif len(shape) == 3:
            scale, row, col = 1.0, shape[1], shape[2]
        elif len(shape) == 2:
            scale, row, col = 1.0, shape
        return scale, row, col

    def flops(self):
        total_flops = {}

        for n, m in self.model.named_modules():
            if isinstance(m, IntMatMul):
                x_shape = m.x_shape.tolist()
                y_shape = m.y_shape.tolist()
                
                sx, rx, cx = self.get_row_col(x_shape)
                sy, ry, cy = self.get_row_col(y_shape)

                assert cx == ry, "ERROR: incorrect MatMul Shape"
                flops = (cx + (cx - 1)) * rx * cy

                total_flops[n] = int(flops) * sx

            elif isinstance(m, (MulShift, MulQuant)):
                pass

                
        return total_flops
    