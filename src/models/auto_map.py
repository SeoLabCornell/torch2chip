"""
Load model architecture from different sources
"""

import torch
import timm
import torchvision
import src.models as t2c_models

from transformers import AutoModelForCausalLM
from src.models.lm.retnet import RetNetForCausalLM

# TODO: expand this list to support more model architectures
MODEL_LIBRARY_MAP = {
    'vit_tiny_patch16_224': ('timm', 'vision_transformer'),
    'vit_small_patch16_224': ('timm', 'vision_transformer'),
    'vit_base_patch16_224': ('timm', 'vision_transformer'),
    'swin_tiny_patch4_window7_224': ('timm', 'swin_transformer'),
    'swin_base_patch4_window7_224': ('timm', 'swin_transformer'),
    'resnet18': ('torchvision', 'models'),
    'resnet34': ('torchvision', 'models'),
    'resnet50': ('torchvision', 'models'),
    'vgg16_bn': ('torchvision', 'models'),
    'Spiral-AI/Spiral-RetNet-3b-base': ('retnet', 'RetNetForCausalLM'),
    'meta-llama/Llama-2-7b-hf': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-1B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.2-3B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-8B-Instruct': ('transformers', 'AutoModelForCausalLM'),
    'meta-llama/Llama-3.1-8B': ('transformers', 'AutoModelForCausalLM'),
}

TORCH_WEIGHTS_MAP = {
    'resnet18': 'ResNet18_Weights',
    'resnet34': 'ResNet34_Weights',
    'resnet50': 'ResNet50_Weights',
    'vgg16_bn': 'VGG16_BN_Weights',
}

class ModelMap:
    def __init__(self, model_name:str):
        self.model_name = model_name
        
    def fetch(self):
        if self.model_name not in MODEL_LIBRARY_MAP:
            raise ValueError(f"Model: {self.model_name} is unknown! Available models: {MODEL_LIBRARY_MAP.keys()}")

        lib_name, sub_name = MODEL_LIBRARY_MAP[self.model_name]

        if lib_name == "transformers":
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        elif lib_name == "timm":
            model_lib = getattr(timm, "models")
            sub_lib = getattr(model_lib, sub_name)
            model_func = getattr(sub_lib, self.model_name)

            model = model_func(pretrained=True)

        elif lib_name == "torchvision":
            model_func = getattr(torchvision.models, self.model_name)
            model_weights = getattr(torchvision.models, TORCH_WEIGHTS_MAP[self.model_name])

            if hasattr(model_weights, "IMAGENET1K_V2"):
                model = model_func(weights=model_weights.IMAGENET1K_V2)
            else:
                model = model_func(weights=model_weights.IMAGENET1K_V1)
        
        elif lib_name == "t2c_models":
            if "RetNet" in self.model_name:
                model = RetNetForCausalLM.from_pretrained(
                    self.model_name
                )
        else:
            raise ValueError(f"Unknown model library {lib_name}")

        return model