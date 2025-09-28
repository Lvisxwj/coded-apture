import torch
import numpy as np

def HSI_max_min_norm(input_data):
    """
    Min-Max normalization for 84-channel HSI data using precomputed statistics
    Args:
        input_data: Input data with shape [bs, 84, W, H], PyTorch Tensor
    Returns:
        Normalized data with same shape, PyTorch Tensor
    """
    bs, channel, W, H = input_data.shape

    # Precomputed min-max values for 84 HSI channels
    max_values = torch.tensor([
        1339.0, 1501.0, 1621.0, 1683.0, 1701.0, 1937.0, 2165.0, 2263.0,
        2413.0, 2497.0, 2697.0, 2987.0, 3075.0, 3103.0, 3085.0, 3021.0, 3257.0,
        3489.0, 3699.0, 3705.0, 3787.0, 3909.0, 4223.0, 4743.0, 5173.0,
        5601.0, 5813.0, 5893.0, 5933.0, 6545.0, 7153.0, 7703.0, 8019.0,
        8601.0, 8841.0, 8887.0, 9047.0, 9177.0, 9421.0, 9959.0, 10737.0,
        11079.0, 11383.0, 11733.0, 11649.0, 11877.0, 11997.0, 12163.0, 12487.0,
        12775.0, 13319.0, 13535.0, 13719.0, 13397.0, 12837.0, 12003.0, 11629.0,
        11179.0, 10939.0, 10965.0, 11355.0, 11795.0, 12041.0, 12351.0, 12407.0,
        12349.0, 12091.0, 11753.0, 10997.0, 10685.0, 10425.0, 10293.0, 10467.0,
        10439.0, 10831.0, 10967.0, 10907.0, 10931.0, 10675.0, 10419.0, 10197.0,
        9515.0, 9257.0, 8777.0
    ], device=input_data.device)

    min_values = torch.tensor([
        455.0, 459.0, 455.0, 457.0, 457.0, 459.0, 459.0, 453.0,
        463.0, 461.0, 463.0, 467.0, 461.0, 461.0, 465.0, 461.0, 465.0,
        463.0, 469.0, 465.0, 467.0, 467.0, 469.0, 469.0, 471.0,
        473.0, 467.0, 467.0, 473.0, 473.0, 475.0, 473.0, 473.0,
        477.0, 479.0, 475.0, 479.0, 479.0, 477.0, 477.0, 479.0,
        485.0, 477.0, 485.0, 477.0, 485.0, 485.0, 483.0, 487.0,
        489.0, 495.0, 495.0, 497.0, 495.0, 505.0, 511.0, 511.0,
        513.0, 515.0, 517.0, 517.0, 511.0, 519.0, 525.0, 519.0,
        523.0, 527.0, 521.0, 517.0, 519.0, 515.0, 513.0, 517.0,
        513.0, 517.0, 513.0, 517.0, 511.0, 509.0, 507.0, 509.0,
        513.0, 511.0, 507.0
    ], device=input_data.device)

    assert channel == 84, f"Input data must have 84 channels, got {channel}"

    # Initialize output
    output_data = torch.zeros_like(input_data)

    # Apply per-channel min-max normalization
    for i in range(channel):
        output_data[:, i, :, :] = (input_data[:, i, :, :] - min_values[i]) / (max_values[i] - min_values[i])

    return output_data

def HSI_reverse_max_min_norm(normalized_data):
    """
    Reverse min-max normalization for 84-channel HSI data
    Args:
        normalized_data: Normalized data with shape [bs, 84, W, H], PyTorch Tensor
    Returns:
        Denormalized data with same shape, PyTorch Tensor
    """
    bs, channel, W, H = normalized_data.shape

    # Same statistics as normalization
    max_values = torch.tensor([
        1339.0, 1501.0, 1621.0, 1683.0, 1701.0, 1937.0, 2165.0, 2263.0,
        2413.0, 2497.0, 2697.0, 2987.0, 3075.0, 3103.0, 3085.0, 3021.0, 3257.0,
        3489.0, 3699.0, 3705.0, 3787.0, 3909.0, 4223.0, 4743.0, 5173.0,
        5601.0, 5813.0, 5893.0, 5933.0, 6545.0, 7153.0, 7703.0, 8019.0,
        8601.0, 8841.0, 8887.0, 9047.0, 9177.0, 9421.0, 9959.0, 10737.0,
        11079.0, 11383.0, 11733.0, 11649.0, 11877.0, 11997.0, 12163.0, 12487.0,
        12775.0, 13319.0, 13535.0, 13719.0, 13397.0, 12837.0, 12003.0, 11629.0,
        11179.0, 10939.0, 10965.0, 11355.0, 11795.0, 12041.0, 12351.0, 12407.0,
        12349.0, 12091.0, 11753.0, 10997.0, 10685.0, 10425.0, 10293.0, 10467.0,
        10439.0, 10831.0, 10967.0, 10907.0, 10931.0, 10675.0, 10419.0, 10197.0,
        9515.0, 9257.0, 8777.0
    ], device=normalized_data.device)

    min_values = torch.tensor([
        455.0, 459.0, 455.0, 457.0, 457.0, 459.0, 459.0, 453.0,
        463.0, 461.0, 463.0, 467.0, 461.0, 461.0, 465.0, 461.0, 465.0,
        463.0, 469.0, 465.0, 467.0, 467.0, 469.0, 469.0, 471.0,
        473.0, 467.0, 467.0, 473.0, 473.0, 475.0, 473.0, 473.0,
        477.0, 479.0, 475.0, 479.0, 479.0, 477.0, 477.0, 479.0,
        485.0, 477.0, 485.0, 477.0, 485.0, 485.0, 483.0, 487.0,
        489.0, 495.0, 495.0, 497.0, 495.0, 505.0, 511.0, 511.0,
        513.0, 515.0, 517.0, 517.0, 511.0, 519.0, 525.0, 519.0,
        523.0, 527.0, 521.0, 517.0, 519.0, 515.0, 513.0, 517.0,
        513.0, 517.0, 513.0, 517.0, 511.0, 509.0, 507.0, 509.0,
        513.0, 511.0, 507.0
    ], device=normalized_data.device)

    assert channel == 84, f"Input data must have 84 channels, got {channel}"

    # Initialize output
    original_data = torch.zeros_like(normalized_data)

    # Apply per-channel denormalization
    for i in range(channel):
        original_data[:, i, :, :] = normalized_data[:, i, :, :] * (max_values[i] - min_values[i]) + min_values[i]

    return original_data

def adaptive_normalize(data, percentile=99.5):
    """
    Adaptive normalization based on data statistics
    Args:
        data: Input tensor [bs, channels, H, W]
        percentile: Percentile for robust normalization
    Returns:
        Normalized data with same shape
    """
    bs, channels, H, W = data.shape
    normalized_data = torch.zeros_like(data)

    for i in range(channels):
        channel_data = data[:, i, :, :]
        # Use percentile for robust normalization
        min_val = torch.quantile(channel_data, (100 - percentile) / 100)
        max_val = torch.quantile(channel_data, percentile / 100)

        # Avoid division by zero
        if max_val - min_val > 1e-8:
            normalized_data[:, i, :, :] = (channel_data - min_val) / (max_val - min_val)
        else:
            normalized_data[:, i, :, :] = channel_data

    return normalized_data

def znormalize(data, dim=None):
    """
    Z-score normalization (mean=0, std=1)
    Args:
        data: Input tensor
        dim: Dimensions to normalize over
    Returns:
        Z-normalized data
    """
    if dim is None:
        # Normalize over spatial dimensions but keep batch and channel separate
        dim = [-2, -1]  # Normalize over H, W dimensions

    mean = torch.mean(data, dim=dim, keepdim=True)
    std = torch.std(data, dim=dim, keepdim=True)

    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)

    return (data - mean) / std

def shift_back(inputs, step=2):
    """
    Shift-back operation for MST (from original MST code)
    Args:
        inputs: Input tensor [bs, nC, row, col]
        step: Shift step size
    Returns:
        Shifted tensor [bs, nC, row, row]
    """
    bs, nC, row, col = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row

    for i in range(nC):
        inputs[:, i, :, :out_col] = inputs[:, i, :, int(step*i):int(step*i)+out_col]

    return inputs[:, :, :, :out_col]

def prepare_mask_for_mst(mask, target_shape):
    """
    Prepare mask for MST model with proper shape and normalization
    Args:
        mask: Original mask tensor
        target_shape: Target shape [bs, channels, H, W]
    Returns:
        Processed mask tensor
    """
    bs, channels, H, W = target_shape

    if mask.dim() == 3:  # [channels, H, W]
        mask = mask.unsqueeze(0)  # Add batch dimension

    if mask.shape[0] == 1 and bs > 1:
        mask = mask.expand(bs, -1, -1, -1)

    # Ensure mask has correct number of channels
    if mask.shape[1] != channels:
        # Repeat or select channels as needed
        if mask.shape[1] == 1:
            mask = mask.expand(-1, channels, -1, -1)
        else:
            # Take first 'channels' channels or repeat as needed
            mask = mask[:, :channels, :, :]

    return mask