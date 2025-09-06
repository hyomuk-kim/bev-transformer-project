import torch
import torch.nn as nn
import torchvision


class EncoderResnet50:
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        resnet = torchvision.models.resnet50(pretrained=True)
        # Define layers to extract features from ResNet50
        pass

    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Define decoder layers here.

    def forward(self, x):
        # Define forward pass for decoder
        return x


class Segnet(nn.Module):
    def __init__(self, latent_dim):
        # Call the parent constructor
        super().__init__()
        self.encoder = EncoderResnet50(latent_dim)
        self.bev_compressor = nn.Sequential()
        self.decoder = Decoder(in_channels=latent_dim, num_classes=1)
        # Additional details on layers will be added.

    def forward(self, x):
        x = self.encoder.forward(x)
        # Additional forward operations will be defined.
        return x
    
