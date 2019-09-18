## resnet with attention

from .attention_conv import AugmentedConv

# need adapt to Torchvision style calls to integrate
__all__ = ['attn_resnet18', 'attn_resnet34', 'attn_resnet50', 'attn_resnet101', 'attn_resnet152']

# expects pth files
model_urls = {
    'attn_resnet18': '',
    'attn_resnet34': '',
    'attn_resnet50': '',
    'attn_resnet101': '',
    'attn_resnet152': ''
}
