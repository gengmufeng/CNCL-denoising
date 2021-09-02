import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################

# generator: CNCL-U-Net

##############################

class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()

        # for content learning
        # encoder
        self.conv1_1_01 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1_1_01 = nn.BatchNorm2d(64)
        self.conv1_2_01 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2_01 = nn.BatchNorm2d(64)

        self.conv2_1_01 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1_01 = nn.BatchNorm2d(128)
        self.conv2_2_01 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2_01 = nn.BatchNorm2d(128)

        self.conv3_1_01 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1_01 = nn.BatchNorm2d(256)
        self.conv3_2_01 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2_01 = nn.BatchNorm2d(256)

        self.conv4_1_01 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1_01 = nn.BatchNorm2d(512)
        self.conv4_2_01 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2_01 = nn.BatchNorm2d(512)

        self.conv5_1_01 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn5_1_01 = nn.BatchNorm2d(1024)
        self.conv5_2_01 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn5_2_01 = nn.BatchNorm2d(512)

        # decoder
        self.upconv4_1_01 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upbn4_1_01 = nn.BatchNorm2d(512)
        self.upconv4_2_01 = nn.Conv2d(512, 256, 3, padding=1)
        self.upbn4_2_01 = nn.BatchNorm2d(256)

        self.upconv3_1_01 = nn.Conv2d(512, 256, 3, padding=1)
        self.upbn3_1_01 = nn.BatchNorm2d(256)
        self.upconv3_2_01 = nn.Conv2d(256, 128, 3, padding=1)
        self.upbn3_2_01 = nn.BatchNorm2d(128)

        self.upconv2_1_01 = nn.Conv2d(256, 128, 3, padding=1)
        self.upbn2_1_01 = nn.BatchNorm2d(128)
        self.upconv2_2_01 = nn.Conv2d(128, 64, 3, padding=1)
        self.upbn2_2_01 = nn.BatchNorm2d(64)

        self.upconv1_1_01 = nn.Conv2d(128, 32, 3, padding=1)
        self.upbn1_1_01 = nn.BatchNorm2d(32)
        self.upconv1_2_01 = nn.Conv2d(32, 1, 3, padding=1)
        self.upbn1_2_01 = nn.BatchNorm2d(64)

        # ************************************************************
        # for noise learning
        # encoder
        self.conv1_1_02 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1_1_02 = nn.BatchNorm2d(64)
        self.conv1_2_02 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2_02 = nn.BatchNorm2d(64)

        self.conv2_1_02 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1_02 = nn.BatchNorm2d(128)
        self.conv2_2_02 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2_02 = nn.BatchNorm2d(128)

        self.conv3_1_02 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1_02 = nn.BatchNorm2d(256)
        self.conv3_2_02 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2_02 = nn.BatchNorm2d(256)

        self.conv4_1_02 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1_02 = nn.BatchNorm2d(512)
        self.conv4_2_02 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2_02 = nn.BatchNorm2d(512)

        self.conv5_1_02 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn5_1_02 = nn.BatchNorm2d(1024)
        self.conv5_2_02 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn5_2_02 = nn.BatchNorm2d(512)

        # decoder
        self.upconv4_1_02 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upbn4_1_02 = nn.BatchNorm2d(512)
        self.upconv4_2_02 = nn.Conv2d(512, 256, 3, padding=1)
        self.upbn4_2_02 = nn.BatchNorm2d(256)

        self.upconv3_1_02 = nn.Conv2d(512, 256, 3, padding=1)
        self.upbn3_1_02 = nn.BatchNorm2d(256)
        self.upconv3_2_02 = nn.Conv2d(256, 128, 3, padding=1)
        self.upbn3_2_02 = nn.BatchNorm2d(128)

        self.upconv2_1_02 = nn.Conv2d(256, 128, 3, padding=1)
        self.upbn2_1_02 = nn.BatchNorm2d(128)
        self.upconv2_2_02 = nn.Conv2d(128, 64, 3, padding=1)
        self.upbn2_2_02 = nn.BatchNorm2d(64)

        self.upconv1_1_02 = nn.Conv2d(128, 32, 3, padding=1)
        self.upbn1_1_02 = nn.BatchNorm2d(32)
        self.upconv1_2_02 = nn.Conv2d(32, 1, 3, padding=1)
        self.upbn1_2_02 = nn.BatchNorm2d(64)

        # ************************************************************
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        # ************************************************************
        # fusion mechanism
        self.final_conv = nn.Conv2d(2, 1, 1)
        # ************************************************************
        self._initialize_weights()

    def forward(self, x0):

        # encoder for content learning
        x1_1_01 = F.relu(self.bn1_1_01(self.conv1_1_01(x0)))
        x1_2_01 = F.relu(self.bn1_2_01(self.conv1_2_01(x1_1_01)))

        x2_0_01 = self.maxpool(x1_2_01)
        x2_1_01 = F.relu(self.bn2_1_01(self.conv2_1_01(x2_0_01)))
        x2_2_01 = F.relu(self.bn2_2_01(self.conv2_2_01(x2_1_01)))

        x3_0_01 = self.maxpool(x2_2_01)
        x3_1_01 = F.relu(self.bn3_1_01(self.conv3_1_01(x3_0_01)))
        x3_2_01 = F.relu(self.bn3_2_01(self.conv3_2_01(x3_1_01)))

        x4_0_01 = self.maxpool(x3_2_01)
        x4_1_01 = F.relu(self.bn4_1_01(self.conv4_1_01(x4_0_01)))
        x4_2_01 = F.relu(self.bn4_2_01(self.conv4_2_01(x4_1_01)))

        x5_0_01 = self.maxpool(x4_2_01)
        x5_1_01 = F.relu(self.bn5_1_01(self.conv5_1_01(x5_0_01)))
        x5_2_01 = F.relu(self.bn5_2_01(self.conv5_2_01(x5_1_01)))

        # decoder for content learning
        upx4_1_01 = self.upsample(x5_2_01)
        upx4_2_01 = F.relu(self.upbn4_1_01(self.upconv4_1_01(torch.cat((upx4_1_01, x4_2_01), 1))))
        upx4_3_01 = F.relu(self.upbn4_2_01(self.upconv4_2_01(upx4_2_01)))

        upx3_1_01 = self.upsample(upx4_3_01)
        upx3_2_01 = F.relu(self.upbn3_1_01(self.upconv3_1_01(torch.cat((upx3_1_01, x3_2_01), 1))))
        upx3_3_01 = F.relu(self.upbn3_2_01(self.upconv3_2_01(upx3_2_01)))

        upx2_1_01 = self.upsample(upx3_3_01)
        upx2_2_01 = F.relu(self.upbn2_1_01(self.upconv2_1_01(torch.cat((upx2_1_01, x2_2_01), 1))))
        upx2_3_01 = F.relu(self.upbn2_2_01(self.upconv2_2_01(upx2_2_01)))

        upx1_1_01 = self.upsample(upx2_3_01)
        upx1_2_01 = self.upconv1_1_01(torch.cat((upx1_1_01, x1_2_01), 1))
        content_1 = self.upconv1_2_01(upx1_2_01)

        # ************************************************************
        # encoder for noise learning
        x1_1_02 = F.relu(self.bn1_1_02(self.conv1_1_02(x0)))
        x1_2_02 = F.relu(self.bn1_2_02(self.conv1_2_02(x1_1_02)))

        x2_0_02 = self.maxpool(x1_2_02)
        x2_1_02 = F.relu(self.bn2_1_02(self.conv2_1_02(x2_0_02)))
        x2_2_02 = F.relu(self.bn2_2_02(self.conv2_2_02(x2_1_02)))

        x3_0_02 = self.maxpool(x2_2_02)
        x3_1_02 = F.relu(self.bn3_1_02(self.conv3_1_02(x3_0_02)))
        x3_2_02 = F.relu(self.bn3_2_02(self.conv3_2_02(x3_1_02)))

        x4_0_02 = self.maxpool(x3_2_02)
        x4_1_02 = F.relu(self.bn4_1_02(self.conv4_1_02(x4_0_02)))
        x4_2_02 = F.relu(self.bn4_2_02(self.conv4_2_02(x4_1_02)))

        x5_0_02 = self.maxpool(x4_2_02)
        x5_1_02 = F.relu(self.bn5_1_02(self.conv5_1_02(x5_0_02)))
        x5_2_02 = F.relu(self.bn5_2_02(self.conv5_2_02(x5_1_02)))

        # decoder for noise learning
        upx4_1_02 = self.upsample(x5_2_02)
        upx4_2_02 = F.relu(self.upbn4_1_02(self.upconv4_1_02(torch.cat((upx4_1_02, x4_2_02), 1))))
        upx4_3_02 = F.relu(self.upbn4_2_02(self.upconv4_2_02(upx4_2_02)))

        upx3_1_02 = self.upsample(upx4_3_02)
        upx3_2_02 = F.relu(self.upbn3_1_02(self.upconv3_1_02(torch.cat((upx3_1_02, x3_2_02), 1))))
        upx3_3_02 = F.relu(self.upbn3_2_02(self.upconv3_2_02(upx3_2_02)))

        upx2_1_02 = self.upsample(upx3_3_02)
        upx2_2_02 = F.relu(self.upbn2_1_02(self.upconv2_1_02(torch.cat((upx2_1_02, x2_2_02), 1))))
        upx2_3_02 = F.relu(self.upbn2_2_02(self.upconv2_2_02(upx2_2_02)))

        upx1_1_02 = self.upsample(upx2_3_02)
        upx1_2_02 = self.upconv1_1_02(torch.cat((upx1_1_02, x1_2_02), 1))
        noise = self.upconv1_2_02(upx1_2_02)

        content_2 = x0 - noise

        # fusion mechanism
        content = self.final_conv(torch.cat((content_1, content_2), 1))

        return torch.cat((content, noise), 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

##############################

# Discriminator: PatchGAN

##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)