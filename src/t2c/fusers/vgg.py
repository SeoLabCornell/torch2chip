import torch.nn as nn
from src.t2c.fusers.fusers import LayerFuser
from src.module.fuse import _QBaseConv2d, _QBaseLinear

class VGGFuser(LayerFuser):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def layers(self):
        """
        Fetch layer information from pretrained model
        """
        conv_bn_relu = []
        for i, m in enumerate(self.model.features):
            if isinstance(m, _QBaseConv2d):
                conv_bn_relu = []
                conv_bn_relu.append(m)

            elif isinstance(m, nn.BatchNorm2d):
                conv_bn_relu.append(m)
            
            elif isinstance(m, nn.ReLU):
                conv_bn_relu.append(m)
                self.groups.append(conv_bn_relu)
            
            elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
                self.groups.append([m])
                
    def fuse(self):
        # update the groups
        self.layers()
        
        features = []
        for cbr in self.groups:
            if len(cbr) == 3:
                new_layer = self.conv_bn_relu(cbr)
                features.append(new_layer)
            else:
                features.append(*cbr)

        self.model.features = nn.Sequential(*features)
        
        classifier = self.model.classifier
        for n, m in classifier.named_modules():
            if isinstance(m, _QBaseLinear):
                new_layer = self.fuse_linear(m)
                classifier[int(n)] = new_layer

        self.model.classifier = classifier
        return self.model