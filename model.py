import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),padding=1)
        self.bn = nn.BatchNorm3d(num_features=out_ch)
        self.conv2 = nn.Conv3d(in_ch,out_ch,kernel_size=(3,3,3),padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_ch)
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x,inplace=True)
        return x

class Unet3D(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet3D,self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool3d(2,stride=2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool3d(2,stride=2)
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool3d(2,stride=2)
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool3d(2,stride=2)
        self.conv5 = ConvBlock(512, 1024)


        self.up6 = nn.ConvTranspose3d(1024, 512, 2, stride=2)
        self.conv6 = ConvBlock(1024, 512)
        self.up7 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.conv7 = ConvBlock(512, 256)
        self.up8 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv8 = ConvBlock(256, 128)
        self.up9 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv9 = ConvBlock(128, 64)
        self.conv10 = nn.Conv3d(64, out_ch, (1,1,1))

    def forward(self,x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out



