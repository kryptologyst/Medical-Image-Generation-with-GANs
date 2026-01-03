"""Evaluation module for medical image generation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from ..models import DCGANGenerator, DCGANDiscriminator
from ..utils import get_device, calculate_image_metrics


class MedicalImageEvaluator:
    """Evaluator for medical image generation models.
    
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        device: Device to run evaluation on.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        self.generator.eval()
        self.discriminator.eval()
    
    def calculate_fid(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        feature_extractor: Optional[nn.Module] = None,
    ) -> float:
        """Calculate Fréchet Inception Distance (FID).
        
        Args:
            real_images: Real images tensor.
            fake_images: Generated images tensor.
            feature_extractor: Optional feature extractor model.
            
        Returns:
            FID score.
        """
        if feature_extractor is None:
            # Use a simple CNN feature extractor
            feature_extractor = self._create_simple_feature_extractor()
        
        feature_extractor = feature_extractor.to(self.device)
        feature_extractor.eval()
        
        with torch.no_grad():
            # Extract features
            real_features = feature_extractor(real_images).cpu().numpy()
            fake_features = feature_extractor(fake_images).cpu().numpy()
        
        # Calculate statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        covmean = self._sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
        
        return float(fid)
    
    def calculate_is(
        self,
        fake_images: torch.Tensor,
        classifier: Optional[nn.Module] = None,
        num_splits: int = 10,
    ) -> Tuple[float, float]:
        """Calculate Inception Score (IS).
        
        Args:
            fake_images: Generated images tensor.
            classifier: Optional classifier model.
            num_splits: Number of splits for calculation.
            
        Returns:
            Tuple of (mean IS, std IS).
        """
        if classifier is None:
            # Use discriminator as a simple classifier
            classifier = self.discriminator
        
        classifier = classifier.to(self.device)
        classifier.eval()
        
        with torch.no_grad():
            # Get predictions
            preds = classifier(fake_images).cpu().numpy()
        
        # Calculate IS
        scores = []
        for i in range(num_splits):
            part = preds[i * (len(preds) // num_splits): (i + 1) * (len(preds) // num_splits)]
            py = np.mean(part, axis=0)
            scores.append(np.exp(np.mean([np.sum(p * np.log(p / py)) for p in part])))
        
        return float(np.mean(scores)), float(np.std(scores))
    
    def calculate_lpips(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> float:
        """Calculate Learned Perceptual Image Patch Similarity (LPIPS).
        
        Args:
            real_images: Real images tensor.
            fake_images: Generated images tensor.
            
        Returns:
            LPIPS score.
        """
        # Simple LPIPS approximation using MSE in feature space
        feature_extractor = self._create_simple_feature_extractor()
        feature_extractor = feature_extractor.to(self.device)
        feature_extractor.eval()
        
        with torch.no_grad():
            real_features = feature_extractor(real_images)
            fake_features = feature_extractor(fake_images)
            
            lpips = torch.mean((real_features - fake_features) ** 2).item()
        
        return lpips
    
    def calculate_ssim_batch(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> float:
        """Calculate Structural Similarity Index (SSIM) for batch.
        
        Args:
            real_images: Real images tensor.
            fake_images: Generated images tensor.
            
        Returns:
            Average SSIM score.
        """
        ssim_scores = []
        
        for i in range(real_images.size(0)):
            real_img = real_images[i].squeeze().cpu().numpy()
            fake_img = fake_images[i].squeeze().cpu().numpy()
            
            ssim = self._calculate_ssim(real_img, fake_img)
            ssim_scores.append(ssim)
        
        return float(np.mean(ssim_scores))
    
    def evaluate_generation_quality(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate overall generation quality.
        
        Args:
            real_images: Real images tensor.
            fake_images: Generated images tensor.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Basic image metrics
        basic_metrics = calculate_image_metrics(real_images, fake_images)
        metrics.update(basic_metrics)
        
        # FID
        try:
            fid = self.calculate_fid(real_images, fake_images)
            metrics['fid'] = fid
        except Exception as e:
            print(f"Error calculating FID: {e}")
            metrics['fid'] = float('inf')
        
        # IS
        try:
            is_mean, is_std = self.calculate_is(fake_images)
            metrics['is_mean'] = is_mean
            metrics['is_std'] = is_std
        except Exception as e:
            print(f"Error calculating IS: {e}")
            metrics['is_mean'] = 0.0
            metrics['is_std'] = 0.0
        
        # LPIPS
        try:
            lpips = self.calculate_lpips(real_images, fake_images)
            metrics['lpips'] = lpips
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
            metrics['lpips'] = float('inf')
        
        # SSIM
        try:
            ssim = self.calculate_ssim_batch(real_images, fake_images)
            metrics['ssim'] = ssim
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            metrics['ssim'] = 0.0
        
        return metrics
    
    def evaluate_discriminator_performance(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate discriminator performance.
        
        Args:
            real_images: Real images tensor.
            fake_images: Generated images tensor.
            
        Returns:
            Dictionary of discriminator metrics.
        """
        with torch.no_grad():
            # Get discriminator predictions
            real_preds = self.discriminator(real_images).cpu().numpy()
            fake_preds = self.discriminator(fake_images).cpu().numpy()
        
        # Create labels
        real_labels = np.ones(len(real_preds))
        fake_labels = np.zeros(len(fake_preds))
        
        # Combine predictions and labels
        all_preds = np.concatenate([real_preds, fake_preds])
        all_labels = np.concatenate([real_labels, fake_labels])
        
        # Calculate metrics
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        
        # Accuracy
        accuracy = np.mean((all_preds > 0.5) == all_labels)
        
        # Real and fake accuracies
        real_accuracy = np.mean(real_preds > 0.5)
        fake_accuracy = np.mean(fake_preds < 0.5)
        
        return {
            'auc': float(auc),
            'ap': float(ap),
            'accuracy': float(accuracy),
            'real_accuracy': float(real_accuracy),
            'fake_accuracy': float(fake_accuracy),
        }
    
    def generate_evaluation_report(
        self,
        dataloader: DataLoader,
        num_samples: int = 1000,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report.
        
        Args:
            dataloader: Data loader for real images.
            num_samples: Number of samples to generate for evaluation.
            save_dir: Directory to save evaluation results.
            
        Returns:
            Comprehensive evaluation report.
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect real images
        real_images = []
        for batch in dataloader:
            real_images.append(batch.to(self.device))
            if len(real_images) * dataloader.batch_size >= num_samples:
                break
        
        real_images = torch.cat(real_images, dim=0)[:num_samples]
        
        # Generate fake images
        with torch.no_grad():
            z = torch.randn(num_samples, self.generator.latent_dim, device=self.device)
            fake_images = self.generator(z)
        
        # Evaluate generation quality
        quality_metrics = self.evaluate_generation_quality(real_images, fake_images)
        
        # Evaluate discriminator performance
        discriminator_metrics = self.evaluate_discriminator_performance(real_images, fake_images)
        
        # Create report
        report = {
            'generation_quality': quality_metrics,
            'discriminator_performance': discriminator_metrics,
            'num_samples': num_samples,
        }
        
        # Save results
        if save_dir is not None:
            # Save metrics
            metrics_path = save_dir / 'evaluation_metrics.json'
            import json
            with open(metrics_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save sample images
            samples_path = save_dir / 'evaluation_samples.npy'
            np.save(samples_path, fake_images.cpu().numpy())
            
            # Create visualizations
            self._create_evaluation_plots(real_images, fake_images, save_dir)
        
        return report
    
    def _create_simple_feature_extractor(self) -> nn.Module:
        """Create a simple CNN feature extractor."""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
        )
    
    def _sqrtm(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate matrix square root."""
        try:
            from scipy.linalg import sqrtm
            return sqrtm(matrix)
        except ImportError:
            # Fallback to eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            eigenvals = np.maximum(eigenvals, 0)  # Ensure non-negative
            return eigenvecs @ np.diag(np.sqrt(eigenvals)) @ eigenvecs.T
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images."""
        # Simple SSIM implementation
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return float(ssim)
    
    def _create_evaluation_plots(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        save_dir: Path,
    ) -> None:
        """Create evaluation plots."""
        # Sample comparison plot
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        
        for i in range(8):
            # Real images
            axes[0, i].imshow(real_images[i].squeeze().cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f'Real {i+1}')
            axes[0, i].axis('off')
            
            # Fake images
            axes[1, i].imshow(fake_images[i].squeeze().cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f'Generated {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'sample_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Distribution plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Pixel value distributions
        real_pixels = real_images.flatten().cpu().numpy()
        fake_pixels = fake_images.flatten().cpu().numpy()
        
        axes[0].hist(real_pixels, bins=50, alpha=0.7, label='Real', density=True)
        axes[0].hist(fake_pixels, bins=50, alpha=0.7, label='Generated', density=True)
        axes[0].set_title('Pixel Value Distribution')
        axes[0].set_xlabel('Pixel Value')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        
        # Discriminator predictions
        with torch.no_grad():
            real_preds = self.discriminator(real_images).cpu().numpy()
            fake_preds = self.discriminator(fake_images).cpu().numpy()
        
        axes[1].hist(real_preds, bins=50, alpha=0.7, label='Real', density=True)
        axes[1].hist(fake_preds, bins=50, alpha=0.7, label='Generated', density=True)
        axes[1].set_title('Discriminator Predictions')
        axes[1].set_xlabel('Prediction Score')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'distribution_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()


def evaluate_model(
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_samples: int = 1000,
    save_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Evaluate a trained GAN model.
    
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        dataloader: Data loader for real images.
        device: Device to run evaluation on.
        num_samples: Number of samples to generate for evaluation.
        save_dir: Directory to save evaluation results.
        
    Returns:
        Evaluation report.
    """
    if device is None:
        device = get_device()
    
    evaluator = MedicalImageEvaluator(generator, discriminator, device)
    report = evaluator.generate_evaluation_report(dataloader, num_samples, save_dir)
    
    return report
