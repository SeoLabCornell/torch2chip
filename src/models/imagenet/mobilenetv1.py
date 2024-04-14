"""
MobileNet-V1
"""

import torch
import torch.nn as nn

class Net(nn.Module):
    """
    MobileNetV1 model
    """
    def __init__(self, alpha=1.0, num_classes=1000):
        super(Net, self).__init__()
        self.alpha = alpha   # width multiplier of the model

        def conv_bn(inp, oup, stride):
            layer = nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, padding=1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                )
            return layer


        def conv_dw(inp, oup, stride):
            layer = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
            return layer

        self.model = nn.Sequential(
            conv_bn(3, int(32*self.alpha), 2), 
            conv_dw(int(32*self.alpha),  int(64*self.alpha), 1),
            conv_dw(int(64*self.alpha), int(128*self.alpha), 2),
            conv_dw(int(128*self.alpha), int(128*self.alpha), 1),
            conv_dw(int(128*self.alpha), int(256*self.alpha), 2),
            conv_dw(int(256*self.alpha), int(256*self.alpha), 1),
            conv_dw(int(256*self.alpha), int(512*self.alpha), 2),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(1024*self.alpha), 2),
            conv_dw(int(1024*self.alpha), int(1024*self.alpha), 1),
        )
        self.pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(int(1024*self.alpha), num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = x.view(-1, int(1024*self.alpha))
        x = self.fc(x)
        return x

def mobilenetv1(num_classes=1000):
    model = Net(num_classes=num_classes)
    return model