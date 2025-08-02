import torch

def max_min_norm(input_data):
    """
    对4维数据的每个通道进行Min-Max归一化，使用预设的每个通道的最小值和最大值。 
    参数：
    - input_data: 输入数据，形状为 [bs, channel, W, H]，类型为 PyTorch Tensor
    
    返回：
    - 归一化后的数据，形状与输入相同，类型为 PyTorch Tensor
    """
    bs, channel, W, H = input_data.shape
    
    # 预设的最小值和最大值列表（32个通道的值）
    max_values = torch.tensor([
        3654, 640320, 0.6851, 14.0, 6.34, 22.36, 0.001014, 0.001703,
        2104.54, 1.14, 15.41, 2.45, 0.7564, 13206, 0.7798, 1630, 2.8,
        0.882, 2.43, 24288.5, 1.152, 0.9144, 0.9318, 90.0883, 17182.64,
        7446.72, 7.21, 0.4455, -0.1623, 0.001558, 1.1258, 1.0804
    ])

    min_values = torch.tensor([
        -5770, -757080, -0.4112, -3.4, -2.2, 0.80, -0.000383, -0.000388,
        -2852.4, -0.6414, -0.4169, -0.3517, -0.4276, -8364, -0.5596, -2980, 0.092,
        -1.048, -0.42, -27577.2, -2.287, -0.399, -0.03435, -41.8550, -15440.64,
        -8535.6, 0.4, -0.7954, -0.9229, -0.00044, 0.0448, 0.0327
    ])


    # 确保通道数为32
    assert channel == 32, "输入数据的通道数必须为32"

    # 初始化输出 (使用 torch.zeros_like)
    output_data = torch.zeros_like(input_data)
    
    # 对每个通道进行Min-Max归一化
    for i in range(channel):
        output_data[:, i, :, :] = (input_data[:, i, :, :] - min_values[i]) / (max_values[i] - min_values[i])
    
    return output_data
