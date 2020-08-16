import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(double_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,out_channels, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        def forward(self,x):
            x=self.conv(x)
            return x

class up_conv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self,img_channel=3):
        super(U_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv1 = double_conv(in_channels=img_channel,out_channels=64)
        self.conv2 = double_conv(in_channels=64,out_channels=128)
        self.conv3 = double_conv(in_channels=128,out_channels=256)
        self.conv4 = double_conv(in_channels=256,out_channels=512)
        self.conv5 = double_conv(in_channels=512,out_channels=1024)

        self.up5 = up_conv(in_channels=1024,out_channels=512)
        self.up_conv5 = double_conv(in_channels=1024,out_channels=512)

        self.up4 = up_conv(in_channels=512,out_channels=256)
        self.up_conv4 = double_conv(in_channels=512,out_channels=256)

        self.up3 = up_conv(in_channels=256,out_channels=128)
        self.up_conv3 = double_conv(in_channels=256,out_channels=128)

        self.up2 = up_conv(in_channels=128,out_channels=64)
        self.up_conv2 =double_conv(in_channels=128, out_channels=64)

        self.conv_out = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=0)

   def forward(self,x):
       #Encoding
        x1 = self.conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.conv5(x5)

        #Decoding and Concatenation
        d5 = self.up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv2(d2)

        out = self.conv_out(d2)

        return out
