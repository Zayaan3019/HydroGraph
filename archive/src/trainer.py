"""
Phase 5: Training and Evaluation Pipeline

This module implements the training loop, focal loss for class imbalance,
and comprehensive evaluation metrics.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import torchmetrics
from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where p_t is the model's estimated probability for the ground truth class.
    
    Parameters
    ----------
    alpha : float
        Weighting factor for the positive class (default: 0.25)
    gamma : float
        Focusing parameter to reduce loss for well-classified examples (default: 2.0)
    reduction : str
        Reduction method: 'none', 'mean', or 'sum' (default: 'mean')
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        logger.info(f"FocalLoss initialized: alpha={alpha}, gamma={gamma}")
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Parameters
        ----------
        inputs : torch.Tensor
            Predicted probabilities (N, 1)
        targets : torch.Tensor
            Ground truth labels (N, 1)
            
        Returns
        -------
        torch.Tensor
            Focal loss value
        """
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        
        # Compute focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Compute alpha term
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal loss
        loss = alpha_t * focal_term * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping (default: 20)
    min_delta : float
        Minimum change to qualify as improvement (default: 1e-4)
    mode : str
        'min' or 'max' for metric (default: 'min' for loss)
    """
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        logger.info(f"EarlyStopping: patience={patience}, mode={mode}")
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Parameters
        ----------
        score : float
            Current validation metric
            
        Returns
        -------
        bool
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning(f"Early stopping triggered after {self.counter} epochs")
        
        return self.early_stop


class MetricsTracker:
    """
    Track and compute evaluation metrics.
    
    Tracks: F1-Score, Precision, Recall, ROC-AUC, Average Precision
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Initialize torchmetrics
        self.f1 = torchmetrics.F1Score(task='binary').to(device)
        self.precision = torchmetrics.Precision(task='binary').to(device)
        self.recall = torchmetrics.Recall(task='binary').to(device)
        self.auroc = torchmetrics.AUROC(task='binary').to(device)
        self.avg_precision = torchmetrics.AveragePrecision(task='binary').to(device)
        
        logger.info("MetricsTracker initialized")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with new predictions.
        
        Parameters
        ----------
        preds : torch.Tensor
            Predicted probabilities
        targets : torch.Tensor
            Ground truth labels
        """
        preds = preds.to(self.device)
        targets = targets.to(self.device)
        
        self.f1.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.auroc.update(preds, targets)
        self.avg_precision.update(preds, targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of metric values
        """
        metrics = {
            'f1_score': self.f1.compute().item(),
            'precision': self.precision.compute().item(),
            'recall': self.recall.compute().item(),
            'roc_auc': self.auroc.compute().item(),
            'avg_precision': self.avg_precision.compute().item(),
        }
        return metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        self.auroc.reset()
        self.avg_precision.reset()


class Trainer:
    """
    Main training class for HydroGraph ST-GNN.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Parameters
    ----------
    model : nn.Module
        HydroGraph ST-GNN model
    train_loader : NeighborLoader
        Training data loader
    val_loader : NeighborLoader
        Validation data loader
    config : dict
        Training configuration
    device : str
        Device for training ('cpu' or 'cuda')
    checkpoint_dir : Path
        Directory for saving checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: NeighborLoader,
        val_loader: NeighborLoader,
        config: dict,
        device: str = 'cuda',
        checkpoint_dir: Path = Path('checkpoints'),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        loss_config = config['training']['loss']
        if loss_config['type'] == 'FocalLoss':
            self.criterion = FocalLoss(
                alpha=loss_config['alpha'],
                gamma=loss_config['gamma'],
            )
        else:
            self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )
        
        # Learning rate scheduler
        scheduler_config = config['training']['scheduler']
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                verbose=True,
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
        )
        
        # Metrics
        self.metrics_tracker = MetricsTracker(device=device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_roc_auc': [],
            'val_avg_precision': [],
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info("=" * 80)
        logger.info("PHASE 5: TRAINING & EVALUATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Device: {device}")
        logger.info(f"Loss function: {loss_config['type']}")
        logger.info(f"Optimizer: AdamW (lr={config['training']['learning_rate']})")
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index)
            
            # Compute loss
            loss = self.criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate on validation set.
        
        Returns
        -------
        Tuple[float, Dict[str, float]]
            (average validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        self.metrics_tracker.reset()
        
        pbar = tqdm(self.val_loader, desc='Validation', leave=False)
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            out = self.model(batch.x, batch.edge_index)
            
            # Compute loss
            loss = self.criterion(out, batch.y)
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            self.metrics_tracker.update(out, batch.y)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        metrics = self.metrics_tracker.compute()
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        is_best : bool
            Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Load model from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
            
        Returns
        -------
        int
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def train(self, num_epochs: int) -> None:
        """
        Main training loop.
        
        Parameters
        ----------
        num_epochs : int
            Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        start_time = datetime.now()
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 40)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val F1: {metrics['f1_score']:.4f}")
            logger.info(f"Val Precision: {metrics['precision']:.4f}")
            logger.info(f"Val Recall: {metrics['recall']:.4f}")
            logger.info(f"Val ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"Val Avg Precision: {metrics['avg_precision']:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(metrics['f1_score'])
            self.history['val_precision'].append(metrics['precision'])
            self.history['val_recall'].append(metrics['recall'])
            self.history['val_roc_auc'].append(metrics['roc_auc'])
            self.history['val_avg_precision'].append(metrics['avg_precision'])
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.warning(f"Early stopping at epoch {epoch}")
                break
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        logger.success("=" * 80)
        logger.success("PHASE 5 COMPLETE: Training finished")
        logger.success(f"Best epoch: {self.best_epoch}")
        logger.success(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.success(f"Training time: {training_time:.2f} seconds")
        logger.success("=" * 80)
    
    def save_history(self, output_path: Path) -> None:
        """
        Save training history to CSV.
        
        Parameters
        ----------
        output_path : Path
            Path to save history CSV
        """
        df = pd.DataFrame(self.history)
        df.to_csv(output_path, index_label='epoch')
        logger.info(f"Saved training history to {output_path}")
