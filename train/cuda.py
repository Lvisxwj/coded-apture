import torch

if torch.cuda.is_available():
    print("CUDA is available.")
    print("Number of available GPUs:", torch.cuda.device_count())
else:
    print("CUDA is not available.")