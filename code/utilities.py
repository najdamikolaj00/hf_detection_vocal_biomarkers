"""
This Python function, `check_cuda`, checks for the availability of a GPU (CUDA) using the torch library.

Functionality:
1. Checks if a CUDA-enabled GPU is available.
2. If available, prints information about the GPU and returns the device name.
3. If not available, prints a message indicating that GPU is not available and returns 0.

Note: This function is commonly used to determine whether GPU acceleration can be utilized for tensor operations in PyTorch. 
It provides information about the GPU device if available or signals the need to perform computations on the CPU if no GPU is present.
"""
import torch

def check_cuda():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name()
        print(f'GPU: {device}')
        return device
    else:
        print('GPU is not available. Training will be performed on CPU.')
        return 0