"""Tests for data pipeline."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data import (
    SyntheticMedicalDataset,
    MedicalImageDataset,
    get_medical_transforms,
    create_dataloader,
    MedicalImageDataModule,
)


class TestSyntheticMedicalDataset:
    """Test synthetic medical dataset."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = SyntheticMedicalDataset(num_samples=10, img_size=64)
        
        assert len(dataset) == 10
        assert dataset.img_size == 64
        assert dataset.img_channels == 1
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = SyntheticMedicalDataset(num_samples=5, img_size=32)
        
        item = dataset[0]
        
        assert isinstance(item, torch.Tensor)
        assert item.shape == (1, 32, 32)
        assert item.dtype == torch.float32
    
    def test_dataset_with_transform(self):
        """Test dataset with transform."""
        transform = lambda x: x * 2  # Simple transform
        
        dataset = SyntheticMedicalDataset(
            num_samples=5,
            img_size=32,
            transform=transform
        )
        
        item = dataset[0]
        assert isinstance(item, torch.Tensor)


class TestMedicalTransforms:
    """Test medical image transforms."""
    
    def test_get_medical_transforms(self):
        """Test getting medical transforms."""
        transform = get_medical_transforms(img_size=128, is_training=True)
        
        assert callable(transform)
        
        # Test transform on sample data
        sample_img = torch.randn(1, 128, 128)
        transformed = transform(sample_img)
        
        assert isinstance(transformed, torch.Tensor)
    
    def test_get_medical_transforms_validation(self):
        """Test validation transforms."""
        transform = get_medical_transforms(img_size=128, is_training=False)
        
        assert callable(transform)
        
        sample_img = torch.randn(1, 128, 128)
        transformed = transform(sample_img)
        
        assert isinstance(transformed, torch.Tensor)


class TestDataLoader:
    """Test data loader creation."""
    
    def test_create_dataloader(self):
        """Test creating data loader."""
        dataset = SyntheticMedicalDataset(num_samples=20, img_size=32)
        dataloader = create_dataloader(dataset, batch_size=4)
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 4
        
        # Test iteration
        batch = next(iter(dataloader))
        assert batch.shape[0] == 4
        assert batch.shape[1:] == (1, 32, 32)


class TestMedicalImageDataModule:
    """Test medical image data module."""
    
    def test_data_module_creation(self):
        """Test data module creation."""
        data_module = MedicalImageDataModule(
            synthetic=True,
            num_synthetic_samples=50,
            img_size=64,
            batch_size=8
        )
        
        assert data_module.synthetic is True
        assert data_module.num_synthetic_samples == 50
        assert data_module.img_size == 64
        assert data_module.batch_size == 8
    
    def test_data_module_setup(self):
        """Test data module setup."""
        data_module = MedicalImageDataModule(
            synthetic=True,
            num_synthetic_samples=20,
            img_size=32,
            batch_size=4
        )
        
        data_module.setup()
        
        assert hasattr(data_module, 'train_dataset')
        assert hasattr(data_module, 'val_dataset')
        assert len(data_module.train_dataset) == 20
        assert len(data_module.val_dataset) == 5  # 20 // 4
    
    def test_data_module_dataloaders(self):
        """Test data module dataloaders."""
        data_module = MedicalImageDataModule(
            synthetic=True,
            num_synthetic_samples=20,
            img_size=32,
            batch_size=4
        )
        
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        
        # Test train loader
        train_batch = next(iter(train_loader))
        assert train_batch.shape[0] == 4
        assert train_batch.shape[1:] == (1, 32, 32)
        
        # Test val loader
        val_batch = next(iter(val_loader))
        assert val_batch.shape[0] == 4
        assert val_batch.shape[1:] == (1, 32, 32)


class TestDataIntegration:
    """Integration tests for data pipeline."""
    
    def test_full_data_pipeline(self):
        """Test complete data pipeline."""
        data_module = MedicalImageDataModule(
            synthetic=True,
            num_synthetic_samples=40,
            img_size=64,
            batch_size=8,
            num_workers=0  # Use 0 workers for testing
        )
        
        data_module.setup()
        
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # Test multiple batches
        train_batches = []
        for i, batch in enumerate(train_loader):
            train_batches.append(batch)
            if i >= 2:  # Test first 3 batches
                break
        
        assert len(train_batches) == 3
        for batch in train_batches:
            assert batch.shape[0] == 8
            assert batch.shape[1:] == (1, 64, 64)
            assert batch.dtype == torch.float32
    
    def test_data_consistency(self):
        """Test data consistency across runs."""
        # Create two identical datasets
        dataset1 = SyntheticMedicalDataset(num_samples=10, img_size=32)
        dataset2 = SyntheticMedicalDataset(num_samples=10, img_size=32)
        
        # Items should be different (random generation)
        item1 = dataset1[0]
        item2 = dataset2[0]
        
        # They should be different due to randomness
        assert not torch.allclose(item1, item2)
        
        # But shapes should be consistent
        assert item1.shape == item2.shape
        assert item1.dtype == item2.dtype


if __name__ == '__main__':
    pytest.main([__file__])
