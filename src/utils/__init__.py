"""Utility functions for medical image generation project."""

import os
import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    **kwargs: Any,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model to save.
        optimizer: Optimizer state.
        epoch: Current epoch.
        loss: Current loss value.
        filepath: Path to save checkpoint.
        **kwargs: Additional data to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file.
        model: PyTorch model to load state into.
        optimizer: Optional optimizer to load state into.
        device: Device to load checkpoint on.
        
    Returns:
        Dict containing checkpoint data.
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def normalize_image(image: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """Normalize image tensor to specified range.
    
    Args:
        image: Input image tensor.
        min_val: Minimum value for normalization.
        max_val: Maximum value for normalization.
        
    Returns:
        Normalized image tensor.
    """
    img_min = image.min()
    img_max = image.max()
    
    if img_max == img_min:
        return torch.zeros_like(image)
    
    normalized = (image - img_min) / (img_max - img_min)
    return normalized * (max_val - min_val) + min_val


def denormalize_image(image: torch.Tensor, original_min: float, original_max: float) -> torch.Tensor:
    """Denormalize image tensor back to original range.
    
    Args:
        image: Normalized image tensor.
        original_min: Original minimum value.
        original_max: Original maximum value.
        
    Returns:
        Denormalized image tensor.
    """
    return image * (original_max - original_min) + original_min


def create_synthetic_medical_image(
    height: int = 256,
    width: int = 256,
    num_objects: int = 3,
    noise_level: float = 0.1,
) -> np.ndarray:
    """Create synthetic medical image for testing.
    
    Args:
        height: Image height.
        width: Image width.
        num_objects: Number of synthetic objects to add.
        noise_level: Amount of noise to add.
        
    Returns:
        Synthetic medical image as numpy array.
    """
    # Create base image
    image = np.zeros((height, width), dtype=np.float32)
    
    # Add synthetic anatomical structures
    center_y, center_x = height // 2, width // 2
    
    # Add rib-like structures
    for i in range(5):
        y_offset = (i - 2) * 30
        for x in range(width):
            y = center_y + y_offset + 10 * np.sin(x * 0.02)
            if 0 <= y < height:
                image[int(y), x] += 0.3
    
    # Add lung fields using numpy operations instead of cv2
    lung_left = np.zeros((height, width), dtype=np.float32)
    lung_right = np.zeros((height, width), dtype=np.float32)
    
    # Create elliptical masks for lungs
    y, x = np.ogrid[:height, :width]
    
    # Left lung ellipse
    center_left_x, center_left_y = width//3, center_y
    a_left, b_left = width//4, height//3
    mask_left = ((x - center_left_x) / a_left) ** 2 + ((y - center_left_y) / b_left) ** 2 <= 1
    lung_left[mask_left] = 0.8
    
    # Right lung ellipse
    center_right_x, center_right_y = 2*width//3, center_y
    a_right, b_right = width//4, height//3
    mask_right = ((x - center_right_x) / a_right) ** 2 + ((y - center_right_y) / b_right) ** 2 <= 1
    lung_right[mask_right] = 0.8
    
    image += lung_left + lung_right
    
    # Add heart shadow
    center_heart_x, center_heart_y = center_x, center_y + 20
    a_heart, b_heart = width//6, height//4
    mask_heart = ((x - center_heart_x) / a_heart) ** 2 + ((y - center_heart_y) / b_heart) ** 2 <= 1
    image[mask_heart] += 0.6
    
    # Add noise
    noise = np.random.normal(0, noise_level, image.shape)
    image += noise
    
    # Normalize to [0, 1]
    image = np.clip(image, 0, 1)
    
    return image


def calculate_image_metrics(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
) -> Dict[str, float]:
    """Calculate basic image quality metrics.
    
    Args:
        real_images: Real image tensor.
        fake_images: Generated image tensor.
        
    Returns:
        Dictionary of calculated metrics.
    """
    # Convert to numpy for easier calculation
    real_np = real_images.detach().cpu().numpy()
    fake_np = fake_images.detach().cpu().numpy()
    
    # Mean Squared Error
    mse = np.mean((real_np - fake_np) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Structural Similarity Index (simplified)
    mu_real = np.mean(real_np)
    mu_fake = np.mean(fake_np)
    sigma_real = np.var(real_np)
    sigma_fake = np.var(fake_np)
    sigma_cross = np.mean((real_np - mu_real) * (fake_np - mu_fake))
    
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_real * mu_fake + c1) * (2 * sigma_cross + c2)) / \
           ((mu_real ** 2 + mu_fake ** 2 + c1) * (sigma_real + sigma_fake + c2))
    
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }
