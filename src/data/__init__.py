"""Data pipeline for medical image generation."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchio as tio
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Resize,
    NormalizeIntensity,
    RandFlip,
    RandRotate,
    RandZoom,
    RandGaussianNoise,
    RandGaussianBlur,
)


class SyntheticMedicalDataset(Dataset):
    """Synthetic medical image dataset for training and testing.
    
    Args:
        num_samples: Number of synthetic samples to generate.
        img_size: Size of the images (assumes square).
        img_channels: Number of image channels.
        transform: Optional transform to apply to images.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        img_size: int = 256,
        img_channels: int = 1,
        transform: Optional[Callable] = None,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.img_channels = img_channels
        self.transform = transform
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[np.ndarray]:
        """Generate synthetic medical images."""
        from ..utils import create_synthetic_medical_image
        
        data = []
        for _ in range(self.num_samples):
            # Create synthetic medical image
            img = create_synthetic_medical_image(
                height=self.img_size,
                width=self.img_size,
                num_objects=np.random.randint(1, 5),
                noise_level=np.random.uniform(0.05, 0.2),
            )
            
            # Convert to tensor format
            if self.img_channels == 1:
                img = img[np.newaxis, ...]  # Add channel dimension
            else:
                img = np.repeat(img[np.newaxis, ...], self.img_channels, axis=0)
            
            data.append(img)
        
        return data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item by index."""
        img = self.data[idx]
        img_tensor = torch.from_numpy(img).float()
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor


class MedicalImageDataset(Dataset):
    """Dataset for loading real medical images.
    
    Args:
        data_dir: Directory containing medical images.
        img_size: Target image size.
        img_channels: Number of image channels.
        transform: Optional transform to apply.
        file_extensions: Supported file extensions.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        img_size: int = 256,
        img_channels: int = 1,
        transform: Optional[Callable] = None,
        file_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.nii', '.nii.gz', '.dcm'),
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.img_channels = img_channels
        self.transform = transform
        
        # Find all image files
        self.image_paths = []
        for ext in file_extensions:
            self.image_paths.extend(list(self.data_dir.rglob(f'*{ext}')))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item by index."""
        img_path = self.image_paths[idx]
        
        # Load image based on extension
        if img_path.suffix in ['.nii', '.gz']:
            # Load NIfTI files
            img = self._load_nifti(img_path)
        elif img_path.suffix in ['.dcm']:
            # Load DICOM files
            img = self._load_dicom(img_path)
        else:
            # Load standard image formats
            img = self._load_image(img_path)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float()
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor
    
    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load NIfTI image."""
        import nibabel as nib
        
        nii = nib.load(str(path))
        img = nii.get_fdata()
        
        # Handle 3D images by taking middle slice
        if len(img.shape) == 3:
            img = img[:, :, img.shape[2] // 2]
        
        return img
    
    def _load_dicom(self, path: Path) -> np.ndarray:
        """Load DICOM image."""
        import pydicom
        
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array
        
        # Convert to float and normalize
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        
        return img
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load standard image format."""
        from PIL import Image
        
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        return img


def get_medical_transforms(
    img_size: int = 256,
    is_training: bool = True,
    augmentation_prob: float = 0.5,
) -> Callable:
    """Get medical image transforms.
    
    Args:
        img_size: Target image size.
        is_training: Whether to apply training augmentations.
        augmentation_prob: Probability of applying augmentations.
        
    Returns:
        Transform function.
    """
    transforms_list = [
        EnsureChannelFirst(),
        Resize((img_size, img_size)),
        NormalizeIntensity(),
    ]
    
    if is_training:
        transforms_list.extend([
            RandFlip(prob=augmentation_prob, spatial_axis=1),
            RandRotate(prob=augmentation_prob, range_x=0.1),
            RandZoom(prob=augmentation_prob, min_zoom=0.9, max_zoom=1.1),
            RandGaussianNoise(prob=augmentation_prob, std=0.01),
            RandGaussianBlur(prob=augmentation_prob, sigma_x=0.5),
        ])
    
    return Compose(transforms_list)


def get_torchio_transforms(
    img_size: int = 256,
    is_training: bool = True,
) -> tio.Transform:
    """Get TorchIO transforms for medical images.
    
    Args:
        img_size: Target image size.
        is_training: Whether to apply training augmentations.
        
    Returns:
        TorchIO transform.
    """
    transforms_list = [
        tio.Resize((img_size, img_size, 1)),
        tio.ZNormalization(),
    ]
    
    if is_training:
        transforms_list.extend([
            tio.RandomFlip(axes=('LR',), flip_probability=0.5),
            tio.RandomAffine(scales=0.1, degrees=10, translation=10),
            tio.RandomNoise(std=0.01),
            tio.RandomBlur(std=0.5),
        ])
    
    return tio.Compose(transforms_list)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for medical images.
    
    Args:
        dataset: Dataset to load.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.
        
    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


class MedicalImageDataModule:
    """Data module for medical image generation.
    
    Args:
        data_dir: Directory containing data.
        batch_size: Batch size.
        img_size: Image size.
        img_channels: Number of image channels.
        num_workers: Number of worker processes.
        synthetic: Whether to use synthetic data.
        num_synthetic_samples: Number of synthetic samples.
    """
    
    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
        img_size: int = 256,
        img_channels: int = 1,
        num_workers: int = 4,
        synthetic: bool = True,
        num_synthetic_samples: int = 1000,
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_workers = num_workers
        self.synthetic = synthetic
        self.num_synthetic_samples = num_synthetic_samples
        
        # Transforms
        self.train_transform = get_medical_transforms(
            img_size=img_size,
            is_training=True,
        )
        self.val_transform = get_medical_transforms(
            img_size=img_size,
            is_training=False,
        )
    
    def setup(self) -> None:
        """Setup datasets."""
        if self.synthetic:
            self.train_dataset = SyntheticMedicalDataset(
                num_samples=self.num_synthetic_samples,
                img_size=self.img_size,
                img_channels=self.img_channels,
                transform=self.train_transform,
            )
            self.val_dataset = SyntheticMedicalDataset(
                num_samples=self.num_synthetic_samples // 4,
                img_size=self.img_size,
                img_channels=self.img_channels,
                transform=self.val_transform,
            )
        else:
            if self.data_dir is None:
                raise ValueError("data_dir must be provided when synthetic=False")
            
            self.train_dataset = MedicalImageDataset(
                data_dir=self.data_dir,
                img_size=self.img_size,
                img_channels=self.img_channels,
                transform=self.train_transform,
            )
            self.val_dataset = MedicalImageDataset(
                data_dir=self.data_dir,
                img_size=self.img_size,
                img_channels=self.img_channels,
                transform=self.val_transform,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return create_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return create_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
