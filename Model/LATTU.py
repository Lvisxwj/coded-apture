from .unet_parts import *
from .TRM_Block import *
from .TRM_Block_with_CNN import *

class LATTU(nn.Module):
    def __init__(self, bilinear=False):
        super(LATTU, self).__init__()
        
        # 保持高度不变的卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), padding=(1, 1))
        # 创建一个序列容器，用于保存多个1x3的卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 3)) for _ in range(9)
        ])

        self.bilinear = bilinear

        self.pre_cat = DoubleConv(192, 64)
        self.inc = DoubleConv(84, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 32)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        """
        self.MS_MSA1 = MS_MSA(dim=128)
        self.MS_MSA2 = MS_MSA(dim=256)
        self.MS_MSA3 = MS_MSA(dim=512)
        self.MS_MSA4 = MS_MSA(dim=1024)
        self.MS_MSA5 = MS_MSA(dim=512)
        self.MS_MSA6 = MS_MSA(dim=256)
        self.MS_MSA7 = MS_MSA(dim=128)
        self.MS_MSA8 = MS_MSA(dim=64)

        """
        
        self.MSAB0 = MSAB(dim=64)
        self.MSAB1 = MSAB(dim=128)
        self.MSAB2 = MSAB(dim=256)
        self.MSAB3 = MSAB(dim=512)
        self.MSAB4 = MSAB(dim=1024)
        self.MSAB5 = MSAB(dim=512)
        self.MSAB6 = MSAB(dim=256)
        self.MSAB7 = MSAB(dim=128)
        self.MSAB8 = MSAB(dim=64)

    def initial_x(self, y):
        """
        :param y: [b,1,256,310]
        :param Phi: [b,28,256,310]
        :return: z: [b,28,256,310]
        """
        nC, step = 84, 2
        bs, row, col = y.shape
        x = torch.zeros(bs, nC, row, row).cuda(3).float()
        for i in range(nC):
            x[:, i, :, :] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        return x
    
    def initial_x_1(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)  # 保持高度64不变
        x = self.relu(x)
        for conv in self.conv_layers:
            x = conv(x)
            x = self.relu(x)
        x = self.conv2(x)  # 保持高度64不变
        x = self.relu(x)
        return x

    def forward(self, x):
        x = self.initial_x(x)

        x1_1_1 = self.inc(x)
        
        x1_1_1 = self.MSAB0(x1_1_1)
        x1_1_2 = self.down1(x1_1_1)
        
        x1_1_2 = self.MSAB1(x1_1_2)
        x1_1_3 = self.down2(x1_1_2)
        
        x1_1_3 = self.MSAB2(x1_1_3)
        x1_1_4 = self.down3(x1_1_3)
        
        x1_1_4 = self.MSAB3(x1_1_4)
        x1_1_5 = self.down4(x1_1_4)

        x1_1 = self.up1(x1_1_5, x1_1_4)
        x1_1 = self.MSAB5(x1_1)

        x1_1 = self.up2(x1_1, x1_1_3)
        x1_1 = self.MSAB6(x1_1)

        x1_1 = self.up3(x1_1, x1_1_2)
        x1_1 = self.MSAB7(x1_1)

        x1_1 = self.up4(x1_1, x1_1_1)
        x1_1 = self.MSAB8(x1_1)

        x1_1 = self.outc(x1_1)
        return x1_1
