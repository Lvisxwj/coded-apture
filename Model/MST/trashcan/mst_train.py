# 测试代码（可临时创建test_mst.py）
import torch
from MST import MST


# 初始化模型
model = MST(in_channels=1, out_channels=3).cuda()
# 模拟输入：batch_size=2，1通道，256x256尺寸
x = torch.randn(2, 1, 256, 256).cuda()
# 模拟掩码
mask = torch.randn(2, 1, 256, 256).cuda()
# 前向传播
out = model(x, mask)
# 检查输出形状是否为 [2, 3, 256, 256]
print(out.shape)  # 预期输出：torch.Size([2, 3, 256, 256])