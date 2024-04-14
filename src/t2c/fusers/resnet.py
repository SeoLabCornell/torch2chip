import torch.nn as nn
from src.t2c.fusers.fusers import LayerFuser

class ResNet18Fuser(LayerFuser):
    def __init__(self, model: nn.Module):
        super().__init__(model)
                    
    def fuse(self):
        for name, module in self.model.named_children():
            if "layer" in name:
                for basic_block_name, basic_block in module.named_children():
                    cbr = [basic_block.conv1, basic_block.bn1, basic_block.relu]
                    cb = [basic_block.conv2, basic_block.bn2, nn.Identity()]
                    
                    # get fused modules
                    fm1 = self.conv_bn_relu(cbr)
                    fm2 = self.conv_bn_relu(cb)

                    # update modules
                    basic_block.conv1 = fm1
                    basic_block.conv2 = fm2 

                    # disable other modules
                    basic_block.bn1 = nn.Identity()
                    basic_block.bn2 = nn.Identity()

                    for sub_block_name, sub_block in basic_block.named_children():
                        if "shortcut" in sub_block_name or "downsample" in sub_block_name:
                            if len(sub_block) > 0:
                                cbr = list(sub_block)
                                cbr.append(nn.Identity())
                                fsc = self.conv_bn_relu(cbr)
                                
                                # update shortcut
                                setattr(basic_block, sub_block_name, fsc)
            
            # special treatment on the first conv-bn-relu block
            elif "conv1" in name:
                cbr = [self.model.conv1, self.model.bn1, self.model.relu]
                
                # get fused modules
                fm1 = self.conv_bn_relu(cbr)

                # update the module
                self.model.conv1 = fm1
                
                # disable other modules
                self.model.bn1 = nn.Identity()
                self.model.relu = nn.Identity()

            elif "fc" in name:
                fm1 = self.fuse_linear(self.model.fc)
                self.model.fc = fm1

        return self.model
    
class ResNet34Fuser(ResNet18Fuser):
    def __init__(self, model: nn.Module):
        super().__init__(model)
    
class ResNet50Fuser(LayerFuser):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def fuse(self):
        for name, module in self.model.named_children():
            if "layer" in name:
                for basic_block_name, basic_block in module.named_children():
                    cb0 = [basic_block.conv1, basic_block.bn1, basic_block.relu]
                    cb1 = [basic_block.conv2, basic_block.bn2, basic_block.relu]
                    cb2 = [basic_block.conv3, basic_block.bn3, nn.Identity()]

                    # get fused modules
                    fm0 = self.conv_bn_relu(cb0)
                    fm1 = self.conv_bn_relu(cb1)
                    fm2 = self.conv_bn_relu(cb2)

                    # update modules
                    basic_block.conv1 = fm0
                    basic_block.conv2 = fm1
                    basic_block.conv3 = fm2

                    # disable other modules
                    basic_block.bn1 = nn.Identity()
                    basic_block.bn2 = nn.Identity()
                    basic_block.bn3 = nn.Identity()

                    for sub_block_name, sub_block in basic_block.named_children():
                        if "shortcut" in sub_block_name or "downsample" in sub_block_name:
                            if len(sub_block) > 0:
                                cbr = list(sub_block)
                                cbr.append(nn.Identity())
                                fsc = self.conv_bn_relu(cbr)
                                
                                # update shortcut
                                setattr(basic_block, sub_block_name, fsc)
                    
            # special treatment on the first conv-bn-relu block
            elif "conv1" in name:
                cbr = [self.model.conv1, self.model.bn1, self.model.relu]
                
                # get fused modules
                fm1 = self.conv_bn_relu(cbr)

                # update the module
                self.model.conv1 = fm1
                
                # disable other modules
                self.model.bn1 = nn.Identity()
                self.model.relu = nn.Identity()

            elif "fc" in name:
                fm1 = self.fuse_linear(self.model.fc)
                self.model.fc = fm1

        return self.model