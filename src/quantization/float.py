"""
Low Precision Floating Point
"""

import torch
import torch.nn as nn
from src.module.base import _QBase

class FP(_QBase):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True):
        super().__init__(nbit, train_flag, unsigned)
        

class BFP(_QBase):
    def __init__(self, nbit: int, train_flag: bool = True, unsigned: bool = True):
        super().__init__(nbit, train_flag, unsigned)