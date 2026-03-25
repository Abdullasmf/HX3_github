"""
Contains the deep learning model used for training.
This can be used on multiple scripts for training and testing.
"""
"""Model definitions for MLP architectures (standard & tapered)."""
import torch.nn as nn
from torch import Tensor
import torch
import numpy as np
#import F




class MultiLayerNN(nn.Module):
    """
    Deep learning model with a customizable number of layers.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MultiLayerNN, self).__init__()
        
        if num_layers < 1:
            raise ValueError("The number of layers must be at least 1.")
        
        # Define layers as a ModuleList to allow dynamic creation
        self.layers = nn.ModuleList()
        
        # First layer: input to first hidden
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        
        # Hidden layers: hidden to hidden
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        
        # Output layer: last hidden to output
        self.layers.append(nn.Linear(hidden_size, output_size))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        """
        for layer in self.layers:
            x = layer(x)
        return x

class TaperedMultiLayerNN(nn.Module):
    """Tapered MLP with optional dropout regularization.

    Hidden layer sizes decay by a constant factor (0.7) each layer while
    respecting a minimum width (4 * output_size). Dropout (if > 0) is applied
    after each ReLU (except the final output layer).
    """
    def __init__(self, input_size, output_size, num_layers, initial_hidden_size=None, dropout: float = 0.0):
        super(TaperedMultiLayerNN, self).__init__()

        if num_layers < 1:
            raise ValueError("The number of layers must be at least 1.")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")

        if initial_hidden_size is None:
            initial_hidden_size = max(64, input_size * 16)

        hidden_sizes = []
        for i in range(num_layers):
            size = int(initial_hidden_size * (0.7 ** i))
            size = max(size, output_size * 4)
            hidden_sizes.append(size)

        self.layers = nn.ModuleList()
        self._use_dropout = dropout > 0.0
        self._dropout_p = dropout

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        if self._use_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        # Hidden tapered layers
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.ReLU())
            if self._use_dropout:
                self.layers.append(nn.Dropout(p=dropout))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    

def get_device():
    """
    This function checks if a GPU is available for use. If not, it defaults to CPU.
    """
    # Step 2: Check if a GPU is available
    if torch.cuda.is_available():
        # Step 3: Set the device to GPU
        device = torch.device("cuda")
        print("GPU is available. Using GPU.")
    else:
        # Step 3: Default to CPU
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")
            # Step 4: Return the device
    return device

class PowerWeightedMSELoss(nn.Module):
    def __init__(self, output_scaler):
        super().__init__()
        self.output_scaler = output_scaler
        self.mse = nn.MSELoss(reduction='none')
        
        # Handle both single and multi-output cases
        if hasattr(output_scaler.mean_, '__len__') and len(output_scaler.mean_) > 1:
            # Multi-output case
            self.scale_mean = torch.tensor(output_scaler.mean_[0], dtype=torch.float32)
            self.scale_std = torch.tensor(output_scaler.scale_[0], dtype=torch.float32)
        else:
            # Single output case
            self.scale_mean = torch.tensor(float(output_scaler.mean_), dtype=torch.float32)
            self.scale_std = torch.tensor(float(output_scaler.scale_), dtype=torch.float32)
    
    def forward(self, pred_scaled, target_scaled):
        # Standard MSE in scaled space
        mse_losses = self.mse(pred_scaled, target_scaled)
        
        # Convert targets to original power space WITHOUT breaking gradients
        target_power = target_scaled.squeeze() * self.scale_std.to(target_scaled.device) + \
                      self.scale_mean.to(target_scaled.device)
        
        # Ensure positive power values for weighting
        target_power = torch.clamp(target_power, min=1.0)  # Minimum 1W
        
        # Inverse power weighting (more weight to low power predictions)
        weights = 1.0 / torch.sqrt(target_power + 50.0)
        weights = weights / weights.mean()
        
        return (mse_losses.squeeze() * weights).mean()

#import F
import torch.nn.functional as F

class PowerRelativeErrorLoss(nn.Module):
    """Loss function that emphasizes relative accuracy across all power ranges"""
    def __init__(self, output_scaler):
        super().__init__()
        self.output_scaler = output_scaler
        self.scale_mean = torch.tensor(float(output_scaler.mean_), dtype=torch.float32)
        self.scale_std = torch.tensor(float(output_scaler.scale_), dtype=torch.float32)
    
    def forward(self, pred_scaled, target_scaled):
        # Convert to original power space
        pred_power = pred_scaled.squeeze() * self.scale_std.to(pred_scaled.device) + \
                    self.scale_mean.to(pred_scaled.device)
        target_power = target_scaled.squeeze() * self.scale_std.to(target_scaled.device) + \
                      self.scale_mean.to(target_scaled.device)
        
        # Ensure positive values
        pred_power = torch.clamp(pred_power, min=1.0)
        target_power = torch.clamp(target_power, min=1.0)
        
        # Relative error loss (percentage-based)
        relative_error = (pred_power - target_power) / target_power
        
        # Combine relative and absolute error for stability
        relative_loss = torch.mean(relative_error ** 2)
        absolute_loss = F.mse_loss(pred_power, target_power) / 10000.0  # Scaled down
        
        return relative_loss + 0.1 * absolute_loss