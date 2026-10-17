import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):

    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

    return conv


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)\

        self.down_conv1 = double_conv(in_channels, features[0])
        self.down_conv2 = double_conv(features[0], features[1])
        self.down_conv3 = double_conv(features[1], features[2])
        self.down_conv4 = double_conv(features[2], features[3])
        self.down_conv5 = double_conv(features[3], features[4])

        self.up_transpose1 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.up_conv1 = double_conv(features[4], features[3])

        self.up_transpose2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.up_conv2 = double_conv(features[3], features[2])

        self.up_transpose3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.up_conv3 = double_conv(features[2], features[1])

        self.up_transpose4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.up_conv4 = double_conv(features[1], features[0])

        self.out = nn.Conv2d(features[0], num_classes, kernel_size=1)


    def forward(self, x):

        # Downsampling path
        down1 = self.down_conv1(x)
        down2 = self.pool(down1)
        down3 = self.down_conv2(down2)
        down4 = self.pool(down3)
        down5 = self.down_conv3(down4)
        down6 = self.pool(down5)
        down7 = self.down_conv4(down6)
        down8 = self.pool(down7)
        down9 = self.down_conv5(down8)

        # Upsampling path
        up1 = self.up_transpose1(down9)
        x = torch.cat((up1, down7), dim=1)  # Concatenate along channel dimension
        x = self.up_conv1(x)

        up2 = self.up_transpose2(x)
        x = torch.cat((up2, down5), dim=1)
        x = self.up_conv2(x)

        up3 = self.up_transpose3(x)
        x = torch.cat((up3, down3), dim=1)
        x = self.up_conv3(x)

        up4 = self.up_transpose4(x)
        x = torch.cat((up4, down1), dim=1)
        x = self.up_conv4(x)


        output = self.out(x)
        return output

