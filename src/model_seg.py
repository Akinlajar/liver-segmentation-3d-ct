
import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Two convolutional layers followed by ReLU activations."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU()
    )


class UNet(nn.Module):
    """A simplified UNet architecture using maxpool for downsampling and bilinear upsampling."""
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (downsampling) layers
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)

        # Decoder (upsampling) layers
        self.dec2 = conv_block(256 + 128, 128)
        self.dec1 = conv_block(128 + 64, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        # Maxpool for downsampling
        self.maxpool = nn.MaxPool2d(2)

        # Upsample for upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x_in):
        # Encoder path
        x1_e = self.enc1(x_in)
        x1_m = self.maxpool(x1_e)

        x2_e = self.enc2(x1_m)
        x2_m = self.maxpool(x2_e)

        x3_e = self.enc3(x2_m)

        # Decoder path
        y2_u = self.upsample(x3_e)
        y2_c = torch.cat((y2_u, x2_e), dim=1)
        y2_d = self.dec2(y2_c)

        y1_u = self.upsample(y2_d)
        y1_c = torch.cat((y1_u, x1_e), dim=1)
        y1_d = self.dec1(y1_c)

        # Final output layer
        out = self.final_conv(y1_d)

        return torch.sigmoid(out)