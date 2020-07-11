import torch
import numpy as np

def square(x):
    return x*x

def safe_log(x, eps=1e-6):
    """ Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))

