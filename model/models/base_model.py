"""
Base PyTorch Lightning module for audio model pretraining.
Provides common functionality that specific architectures can inherit.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from abc import abstractmethod


class BaseAudioPretrainModel(pl.LightningModule):
    """
    Base class for audio pretraining models.
    Subclass this and implement the abstract methods for specific architectures.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        max_steps: int = 400000,
        optimizer: str = "adamw",
        scheduler: str = "linear_warmup_cosine",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        
    @abstractmethod
    def forward(self, audio: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            audio: Input audio tensor [batch, time] or [batch, channels, time]
            **kwargs: Additional architecture-specific inputs
            
        Returns:
            Dictionary containing model outputs (e.g., 'logits', 'features', etc.)
        """
        pass
    
    @abstractmethod
    def compute_loss(self, batch: Dict[str, Any], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the pretraining loss.
        
        Args:
            batch: Batch dictionary from dataloader
            outputs: Model outputs from forward()
            
        Returns:
            Loss tensor
        """
        pass
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with automatic loss computation and logging."""
        outputs = self(**batch)
        loss = self.compute_loss(batch, outputs)
        
        # Log main loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log additional metrics if provided
        if "metrics" in outputs:
            for key, value in outputs["metrics"].items():
                self.log(f"train/{key}", value, on_step=True, on_epoch=True, sync_dist=True)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            self.log("train/nan_loss", 1.0, on_step=True)
            return None
            
        return loss

    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get all parameters that require gradients
        params = [p for p in self.parameters() if p.requires_grad]
        
        # Create optimizer
        if self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        # Create scheduler
        if self.scheduler_name.lower() == "linear_warmup_cosine":
            def lr_lambda(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
        elif self.scheduler_name.lower() == "constant":
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")
    
    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms for debugging."""
        if self.global_step % 100 == 0:
            grad_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            self.log("train/grad_norm", grad_norm, on_step=True, sync_dist=True)
