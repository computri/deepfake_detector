import torch
import torch.nn as nn
from torchvision.models import get_model_weights, get_model

_MODEL_ARG_NAMES = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vit16": "vit_b_16",
    "convnext": "convnext_tiny",
}

def image_size(backbone):
    if backbone.startswith("resnet"):
        return 32
    else:
        return 224
        

class Discriminator(nn.Module):
    def __init__(self, backbone, last_layer_in_features, flatten=False):
        super().__init__()

        self.backbone = backbone
        
        if flatten:
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.flatten = nn.Identity()

        self.head = nn.Linear(last_layer_in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.head(x)
        return x
        

def build_discriminator(
    backbone="resnet18", 
    pretrained=False, 
    fine_tune=False
):
    if fine_tune:
        assert pretrained, "Fine-tuning requires a pretrained model."

    assert backbone in _MODEL_ARG_NAMES, f"Unsupported backbone '{backbone}'. Supported backbones: {list(_MODEL_ARG_NAMES.keys())}"

    if pretrained:
        weights = get_model_weights(_MODEL_ARG_NAMES[backbone])    
    else:
        weights = None
    
    model = get_model(_MODEL_ARG_NAMES[backbone], weights=weights)
    
    flatten = False

    if backbone.startswith("resnet"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        last_layer_in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif backbone == "convnext":
        last_layer_in_features = model.classifier[2].in_features
        model.classifier = nn.Identity()
        flatten = True
    elif backbone == "vit16":
        last_layer_in_features = model.heads.head.in_features
        model.heads.head = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    discriminator = Discriminator(model, last_layer_in_features, flatten=flatten)

    if fine_tune:
        for param in discriminator.backbone.parameters():
            param.requires_grad = False
        discriminator.head.requires_grad = True

    return discriminator
    

def build_generator():
    pass
