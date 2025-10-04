import torch

def get_device(logger=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if logger is not None:
        logger.warning(f"DEVICE: {device}")
    return device