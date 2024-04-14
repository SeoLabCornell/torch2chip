import torch
import torch.nn as nn
from src.t2c.fusers.fusers import LayerFuser
from src.module.base import _QBaseConv2d, _QBaseLinear

class MobileNetV1Fuser(LayerFuser):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def inference(self):
        """
        Switch to inference mode
        """
        for n, m in self.model.named_modules():
            if hasattr(m, "inference"):
                m.inference()

    def layers(self):
        """
        Fetch layer information from pretrained model
        """
        conv_bn_relu = []
        l = 0
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, "wbit"):
                self.fpl += 1
            
            elif isinstance(m, _QBaseConv2d):
                self.flag = True
                conv_bn_relu.append(m)

                # scales and boundaries
                self.xscales.append(m.aq.scale.data)
                self.xzps.append(m.aq.zero_point.data)
                l += 1
            
            elif isinstance(m, nn.BatchNorm2d) and self.flag:
                conv_bn_relu.append(m)
            
            elif isinstance(m, nn.ReLU) and self.flag:
                conv_bn_relu.append(m)
                self.groups.append(conv_bn_relu)
                
                # reset
                self.flag = False
                conv_bn_relu = []
            
            elif isinstance(m, _QBaseLinear):
                self.fpc = False
                
                if not isinstance(m.aq, nn.Identity):
                    # scales and boundaries
                    self.xscales.append(m.aq.scale.data)
                    self.xzps.append(m.aq.zero_point.data)
                    l += 1
    
    def fuse(self):
        """
        Fuse conv, layer, relu for MobileNet architecture
        """
        l = 0   # layer counter
        
        fused_model = self.model

        # update the groups
        self.layers()

        for name, module in self.model.named_children():
            if isinstance(module, (nn.AvgPool2d, nn.Linear)):
                continue
            else:
                # layers in the bottom level sequential
                for n, m in module.named_children():
                    assert len(m) > 0
                    seq = []
                    for layer in m.modules():
                        if isinstance(layer, nn.Conv2d) and not hasattr(layer, "wbit"):
                            seq.append(layer)

                        elif isinstance(layer, _QBaseConv2d):
                            # fetch the module
                            conv_bn_relu = self.groups[l]
                            self.flag = True
                            l += 1

                            if l < len(self.xscales)-1:
                                snxt = self.xscales[l+1]
                                zpnxt = self.xzps[l+1]
                                int_out = True
                            else:
                                snxt = torch.tensor(1.0)
                                zpnxt = torch.tensor(0.0)
                                int_out = False

                            tmp = self.conv_bn_relu(conv_bn_relu, l=l, snxt=snxt, zpnxt=zpnxt, int_out=int_out)
                            seq.append(tmp)

                        elif isinstance(layer, nn.BatchNorm2d):
                            if l != 0:
                                tmp = nn.Identity()
                                seq.append(tmp)
                            else:
                                seq.append(layer)
                        
                        elif isinstance(layer, nn.ReLU):
                            if l != 0:
                                tmp = nn.Identity()
                                seq.append(tmp)
                            else:
                                seq.append(layer)
                            self.flag = False
                    
                    # reconstruct    
                    seq = nn.Sequential(*seq)
                    setattr(module, n, seq)
                setattr(fused_model, name, module)

        # linear = fused_model.fc 
        fused_linear = self.fuse_linear(fused_model.fc)
        setattr(fused_model, "fc", fused_linear)
        
        return fused_model