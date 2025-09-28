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


def reverse_max_min_norm(normalized_data):
    """
    对经过Min-Max归一化的数据进行反归一化，恢复到原始数据。
    参数：
    - normalized_data: 已经归一化的数据，形状为 [bs, channel, W, H]，类型为 PyTorch Tensor
    
    返回：
    - 反归一化后的数据，形状与输入相同，类型为 PyTorch Tensor
    """
    bs, channel, W, H = normalized_data.shape
    
    # 预设的最小值和最大值列表（32个通道的值），与归一化时保持一致
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
    original_data = torch.zeros_like(normalized_data)
    
    # 对每个通道进行反归一化
    for i in range(channel):
        original_data[:, i, :, :] = normalized_data[:, i, :, :] * (max_values[i] - min_values[i]) + min_values[i]
    
    return original_data

def XA_max_min_norm(input_data):
    """
    对4维数据的每个通道进行Min-Max归一化，使用预设的每个通道的最小值和最大值。 
    参数：
    - input_data: 输入数据，形状为 [bs, channel, W, H]，类型为 PyTorch Tensor
    
    返回：
    - 归一化后的数据，形状与输入相同，类型为 PyTorch Tensor
    """
    bs, channel, W, H = input_data.shape
    
    max_values = torch.tensor([
        3025, 521360, 0.284532, 837, 391, 3.48898, 0.000136779, 0.00020001,
        1747.45, 0.720419, 1.79104, 0.905292, 0.480308, 5143, 0.553202, 8735, 1.93984,
        0.733002, 1.26033, 13228.1, 0.905988, 0.671108, 0.591036, 64.2795, 10269.4,
        8670.99, 2.84843, 0.282241, 0, 0.000164337, 1.18954, 1.56656
    ])

    min_values = torch.tensor([
        -2702, -469620, -0.284098, -1097, -663, 0, -4.57047e-05, -0.000132116,
        -4198.1, -0.579791, -0.50372, -0.245862, -0.386539, -7479, -0.44216, -8405, 0,
        -0.681453, -0.321003, -15411.6, -1.26246, -0.294866, -0.199074, -32.8796, -9767.28,
        -9418.38, 0, -0.365827, -0.915261, -0.000176148, 0, 0
    ])

    # 确保通道数为32
    assert channel == 32, "输入数据的通道数必须为32"

    # 初始化输出 (使用 torch.zeros_like)
    output_data = torch.zeros_like(input_data)
    
    # 对每个通道进行Min-Max归一化
    for i in range(channel):
        output_data[:, i, :, :] = (input_data[:, i, :, :] - min_values[i]) / (max_values[i] - min_values[i])
    
    return output_data

def reverse_XA_max_min_norm(normalized_data):
    """
    对经过Min-Max归一化的数据进行反归一化，恢复到原始数据。
    参数：
    - normalized_data: 已经归一化的数据，形状为 [bs, channel, W, H]，类型为 PyTorch Tensor
    
    返回：
    - 反归一化后的数据，形状与输入相同，类型为 PyTorch Tensor
    """
    bs, channel, W, H = normalized_data.shape
    
    # 预设的最小值和最大值列表（32个通道的值），与归一化时保持一致
    max_values = torch.tensor([
        3025, 521360, 0.284532, 837, 391, 3.48898, 0.000136779, 0.00020001,
        1747.45, 0.720419, 1.79104, 0.905292, 0.480308, 5143, 0.553202, 8735, 1.93984,
        0.733002, 1.26033, 13228.1, 0.905988, 0.671108, 0.591036, 64.2795, 10269.4,
        8670.99, 2.84843, 0.282241, 0, 0.000164337, 1.18954, 1.56656
    ])

    min_values = torch.tensor([
        -2702, -469620, -0.284098, -1097, -663, 0, -4.57047e-05, -0.000132116,
        -4198.1, -0.579791, -0.50372, -0.245862, -0.386539, -7479, -0.44216, -8405, 0,
        -0.681453, -0.321003, -15411.6, -1.26246, -0.294866, -0.199074, -32.8796, -9767.28,
        -9418.38, 0, -0.365827, -0.915261, -0.000176148, 0, 0
    ])
    
    # 确保通道数为32
    assert channel == 32, "输入数据的通道数必须为32"
    
    # 初始化输出 (使用 torch.zeros_like)
    original_data = torch.zeros_like(normalized_data)
    
    # 对每个通道进行反归一化
    for i in range(channel):
        original_data[:, i, :, :] = normalized_data[:, i, :, :] * (max_values[i] - min_values[i]) + min_values[i]
    
    return original_data

def Chi_max_min_norm(input_data):
    """
    对4维数据的每个通道进行Min-Max归一化，使用预设的每个通道的最小值和最大值。 
    参数：
    - input_data: 输入数据，形状为 [bs, channel, W, H]，类型为 PyTorch Tensor
    
    返回：
    - 归一化后的数据，形状与输入相同，类型为 PyTorch Tensor
    """
    bs, channel, W, H = input_data.shape
    
    max_values = torch.tensor([
        1598, 388400, 0.800098, 1037, 747, 3632, 0.320175, 0.327451,
        1049.74, 1.49886, 18.8421, 2.19722, 1, 8300, 0.99996, 6380, 402,
        1, 61.4654, 9909.96, 5.15174, 1.15995, 1, 74.9172, 8366.68,
        3588, 1780, 1, 1, 0.0337637, 4.11732, 7.81148
    ])

    min_values = torch.tensor([
        -2504, -205340, -0.777778, -943, -269, 0, -0.00252438, -0.00521619,
        -1015.39, -1.46301, -0.983088, -0.950081, -0.978022, -2145, -1.29884, -7920, 0,
        -12.4753, -0.902503, -7865.28, -4.73155, -1.01309, -0.966835, -34.035, -3424.88,
        -83119.8, 0, -0.817035, -0.99778, -0.00433179, 0, 0
    ])

    # 确保通道数为32
    assert channel == 32, "输入数据的通道数必须为32"

    # 初始化输出 (使用 torch.zeros_like)
    output_data = torch.zeros_like(input_data)
    
    # 对每个通道进行Min-Max归一化
    for i in range(channel):
        output_data[:, i, :, :] = (input_data[:, i, :, :] - min_values[i]) / (max_values[i] - min_values[i])
    
    return output_data

def reverse_Chi_max_min_norm(normalized_data):
    """
    对经过Min-Max归一化的数据进行反归一化，恢复到原始数据。
    参数：
    - normalized_data: 已经归一化的数据，形状为 [bs, channel, W, H]，类型为 PyTorch Tensor
    
    返回：
    - 反归一化后的数据，形状与输入相同，类型为 PyTorch Tensor
    """
    bs, channel, W, H = normalized_data.shape
    
    # 预设的最小值和最大值列表（32个通道的值），与归一化时保持一致
    max_values = torch.tensor([
        1598, 388400, 0.800098, 1037, 747, 3632, 0.320175, 0.327451,
        1049.74, 1.49886, 18.8421, 2.19722, 1, 8300, 0.99996, 6380, 402,
        1, 61.4654, 9909.96, 5.15174, 1.15995, 1, 74.9172, 8366.68,
        3588, 1780, 1, 1, 0.0337637, 4.11732, 7.81148
    ])

    min_values = torch.tensor([
        -2504, -205340, -0.777778, -943, -269, 0, -0.00252438, -0.00521619,
        -1015.39, -1.46301, -0.983088, -0.950081, -0.978022, -2145, -1.29884, -7920, 0,
        -12.4753, -0.902503, -7865.28, -4.73155, -1.01309, -0.966835, -34.035, -3424.88,
        -83119.8, 0, -0.817035, -0.99778, -0.00433179, 0, 0
    ])
    
    # 确保通道数为32
    assert channel == 32, "输入数据的通道数必须为32"
    
    # 初始化输出 (使用 torch.zeros_like)
    original_data = torch.zeros_like(normalized_data)
    
    # 对每个通道进行反归一化
    for i in range(channel):
        original_data[:, i, :, :] = normalized_data[:, i, :, :] * (max_values[i] - min_values[i]) + min_values[i]
    
    return original_data