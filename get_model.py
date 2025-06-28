import torch.nn as nn
from functools import partial
from mamba_linoss import create_mamba_linoss_fast
import torch
import torch.nn.functional as F
import math
from mamba import Mamba, MambaConfig
from mamba_momentum import MambaMomentum, MambaMomentumConfig

class EnhancedRMSNorm(nn.Module):
    """Slightly enhanced RMS normalization for better performance"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * x / (norm + self.eps)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: [..., d_model]
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * x / (norm + self.eps)




def get_model(args=None, device="cuda:0", use_improved=True, model_type="diagram", seq_len=256, 
              momentum_beta=0.8, momentum_alpha=1.0, momentum_mode="after_zoh"):
    """
    Get model with different architectures
    
    Args:
        args: argument object (can be None)
        model_type: "diagram", "lightweight", "multiscale", "original", or "momentum"
        seq_len: sequence length for investigation
        momentum_beta: momentum decay factor (for momentum model)
        momentum_alpha: momentum input scaling (for momentum model)
        momentum_mode: "before_zoh" or "after_zoh" (for momentum model)
    """
    num_classes = 32
    
    # Extract parameters from args if provided
    if args is not None:
        d_model = getattr(args, 'd_model', 128)
        num_classes = getattr(args, 'num_classes', 32)
        input_channels = getattr(args, 'input_channels', 6)
        seq_len = getattr(args, 'seq_len', seq_len)
        
        # Override momentum parameters if provided in args
        momentum_beta = getattr(args, 'momentum_beta', momentum_beta)
        momentum_alpha = getattr(args, 'momentum_alpha', momentum_alpha)
        momentum_mode = getattr(args, 'momentum_mode', momentum_mode)
        
        # Override model_type if provided in args
        model_type = getattr(args, 'model_type', model_type)

    
    print(f"Creating {model_type} model with d_model={d_model}, momentum_beta={momentum_beta}, momentum_alpha={momentum_alpha}")
    
    # Mamba Momentum model - FIXED
    if model_type == "momentum":
        model = Mamba_Momentum_Direct(  # Use the optimized direct version
            d_model=d_model,
            num_classes=num_classes,
            input_channels=input_channels,
            n_layers=2,  # Use 2 layers for better performance
            momentum_beta=momentum_beta,
            momentum_alpha=momentum_alpha,
            # momentum_mode=momentum_mode
        ).to(device)
        
    else:
        # Default to original MAMCA model for compatibility
        config = {"d_model": d_model}
        model = MAMCA(config=config, length=seq_len, num_claasses=num_classes).to(device)
    
    return model



class MAMCA(nn.Module):  # Keep original for compatibility, but fix the bug
    def __init__(
        self,
        config=None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        length=128,
        num_claasses=32
    ) -> None:
        self.config = config
        d_model = config['d_model']

        super().__init__()
        
        print("d_model: ", d_model)

        self.backbone = create_mamba_linoss_fast(d_model=d_model, n_layer=2)

        # Fixed conv layers with proper normalization
        self.conv1 = nn.Sequential(
            nn.Conv1d(6, 6, kernel_size=3, padding=1, groups=6, bias=False),
            nn.Conv1d(6, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),  # Use BatchNorm1d for conv outputs
            nn.GELU()
        )
        
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(int(d_model), num_claasses)
        
        self._init_weights()

    def _init_weights(self):
        """Lightweight weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Conservative initialization for classifier
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, hidden_states, inference_params=None):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.drop(hidden_states)
        
        # Transpose from (batch, d_model, seq_len) to (batch, seq_len, d_model) for MambaLinOSS
        hidden_states = hidden_states.transpose(1, 2)
        
        hidden_states = self.backbone(hidden_states)
        hidden_states = self.drop(hidden_states)        
        hidden_states = hidden_states.mean(dim=1)
        hidden_states = self.fc(hidden_states)
        
        return hidden_states
    

class Mamba_Momentum_Direct(nn.Module):
    """
    OPTIMIZED: Direct Mamba Momentum implementation that avoids double momentum blocks.
    Architecture: Conv1d -> Dropout -> Direct MambaMomentum -> RMS Norm -> Dropout -> Global Avg Pool -> Linear
    
    This is much faster than using two MambaMomentumBlocks because it avoids redundant momentum computations.
    """
    def __init__(self, d_model=128, num_classes=32, input_channels=6, n_layers=2,
                 momentum_beta=0.8, momentum_alpha=1.0):
        super().__init__()
        
        print(f"Creating Mamba_Momentum_Direct with d_model={d_model}, n_layers={n_layers}, momentum_beta={momentum_beta}, momentum_alpha={momentum_alpha}")
        
        # Store momentum parameters
        self.momentum_beta = momentum_beta
        self.momentum_alpha = momentum_alpha
        
        # Initial Conv1d layer: (B,6,L) -> (B,d_model,L) with BatchNorm1d and ReLU
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # Dropout after conv1d (p=0.1 as optimized)
        self.dropout1 = nn.Dropout(0.1)
        
        # Direct MambaMomentum model (much more efficient)
        config = MambaMomentumConfig(
            d_model=d_model,
            n_layers=n_layers,  # Use n_layers instead of separate blocks
            momentum_beta=momentum_beta,
            momentum_alpha=momentum_alpha,
            d_state=64,
            expand_factor=2,
            d_conv=4
        )
        # config = MambaConfig(
        #     d_model=d_model,
        #     n_layers=n_layers,
        #     # dt_rank=16,
        #     d_state=64,
        #     expand_factor=2,
        #     d_conv=4)
        
        self.mamba_momentum = MambaMomentum(config)

        # RMS Norm (as shown in diagram)
        self.rms_norm = RMSNorm(d_model)
        
        # Dropout before classifier (p=0.1 as optimized)
        self.dropout2 = nn.Dropout(0.1)
        
        # Final linear layer: D -> n_class
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Enhanced weight initialization optimized for Mamba Momentum SOTA performance"""
        if isinstance(module, nn.Conv1d):
            # Enhanced He initialization with momentum consideration
            fan_out = module.kernel_size[0] * module.out_channels
            if hasattr(module, 'groups'):
                fan_out //= module.groups
            # Boost factor based on momentum parameters
            momentum_boost = 1.0 + 0.15 * self.momentum_beta + 0.05 * self.momentum_alpha
            std = math.sqrt(2.0 / fan_out) * momentum_boost
            nn.init.normal_(module.weight, 0, std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
        elif isinstance(module, nn.Linear):
            fan_in = module.in_features
            fan_out = module.out_features
            
            # Enhanced initialization for final classifier with momentum consideration
            if fan_out == 32:  # num_classes = 32
                # Momentum-aware classifier initialization
                std = 0.008 + 0.005 * self.momentum_beta  # More conservative with momentum
                nn.init.normal_(module.weight, 0, std)
            else:
                # Enhanced Xavier initialization with momentum scaling
                momentum_scale = 1.0 + 0.1 * self.momentum_alpha
                bound = math.sqrt(6.0 / (fan_in + fan_out)) * 0.9 * momentum_scale
                nn.init.uniform_(module.weight, -bound, bound)
                
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                    
        elif isinstance(module, nn.BatchNorm1d):
            # Momentum-aware BatchNorm initialization
            weight_init = 1.0 + 0.02 * self.momentum_beta  # Slightly higher with momentum
            nn.init.constant_(module.weight, weight_init)
            nn.init.constant_(module.bias, 0.0)
            
        elif isinstance(module, (RMSNorm, EnhancedRMSNorm)):
            # Enhanced momentum-aware RMSNorm initialization
            norm_init = 0.93 + 0.05 * self.momentum_beta + 0.01 * self.momentum_alpha
            nn.init.constant_(module.weight, norm_init)
            
        # Enhanced Mamba Momentum initialization
        if hasattr(module, 'layers'):  # MambaMomentum has layers attribute
            self._init_mamba_momentum_weights(module)

    def _init_mamba_momentum_weights(self, mamba_momentum_module):
        """
        ENHANCED: Optimized Mamba Momentum initialization with advanced strategies for higher accuracy
        """
        for name, param in mamba_momentum_module.named_parameters():
            if 'weight' in name:
                if 'in_proj' in name or 'out_proj' in name:
                    if param.dim() >= 2:
                        # Enhanced orthogonal initialization with momentum-specific scaling
                        gain = 0.95 + 0.1 * self.momentum_beta  # Adaptive gain
                        nn.init.orthogonal_(param, gain=gain)
                        # Add momentum-aware perturbation for better convergence
                        with torch.no_grad():
                            momentum_scale = 1.0 + 0.08 * self.momentum_beta
                            param.mul_(momentum_scale)
                            
                elif 'x_proj' in name or 'dt_proj' in name:
                    fan_in = param.size(-1) if param.dim() >= 2 else param.size(0)
                    # Enhanced momentum-aware initialization scaling
                    momentum_factor = 1.0 + 0.25 * self.momentum_alpha + 0.1 * self.momentum_beta
                    std = (0.95 * momentum_factor) / math.sqrt(fan_in)
                    nn.init.normal_(param, 0, std)
                    
                elif 'conv1d' in name:
                    if param.dim() >= 2:
                        fan_out = param.size(0) * param.size(2) if param.dim() == 3 else param.size(0)
                        # Enhanced He initialization for momentum with boost factor
                        momentum_conv_boost = 1.0 + 0.12 * self.momentum_beta
                        std = math.sqrt(2.3 / fan_out) * momentum_conv_boost
                        nn.init.normal_(param, 0, std)
                        
                elif 'momentum' in name.lower():
                    # Special initialization for momentum-specific weights
                    if param.dim() >= 2:
                        gain = 1.15 + 0.05 * self.momentum_alpha  # Adaptive gain
                        nn.init.xavier_uniform_(param, gain=gain)
                    else:
                        std = 0.008 + 0.003 * self.momentum_beta
                        nn.init.normal_(param, 0, std)
                        
                else:
                    # Enhanced general weight initialization with momentum consideration
                    if param.dim() >= 2:
                        fan_in, fan_out = param.size(-1), param.size(0)
                        # Adaptive bound based on momentum parameters
                        momentum_adapt = 1.0 + 0.08 * self.momentum_beta + 0.04 * self.momentum_alpha
                        bound = math.sqrt(6.0 / (fan_in + fan_out)) * 1.05 * momentum_adapt
                        nn.init.uniform_(param, -bound, bound)
                    else:
                        std = 0.018 * (1.0 + 0.12 * self.momentum_beta)
                        nn.init.normal_(param, 0, std)
                        
            elif 'bias' in name:
                if 'dt_proj' in name:
                    # Enhanced bias initialization for dt projection in momentum context
                    with torch.no_grad():
                        dt_init_std = 0.08 * (1.0 + 0.06 * self.momentum_alpha + 0.03 * self.momentum_beta)
                        param.uniform_(-dt_init_std, dt_init_std)
                else:
                    nn.init.constant_(param, 0)
                    
            elif 'A_log' in name:
                with torch.no_grad():
                    # Enhanced A_log initialization with momentum-aware range
                    a_min = -4.0 + 0.15 * self.momentum_beta  
                    a_max = -1.0 - 0.1 * self.momentum_alpha
                    param.uniform_(a_min, a_max)
                    
            elif 'D' in name:
                # Momentum-aware D parameter initialization
                d_init = 1.0 + 0.05 * self.momentum_beta
                nn.init.constant_(param, d_init)
                
            elif 'momentum_beta' in name:
                # Initialize momentum parameters if they exist as learnable parameters
                nn.init.constant_(param, self.momentum_beta)
    
    def forward(self, x):
        # Input: (B, 6, L)
        
        # Conv1d + BatchNorm1d + ReLU
        x = self.conv1d(x)  # (B, 6, L) -> (B, d_model, L)
        
        # Dropout
        x = self.dropout1(x)
        
        # Transpose for Mamba Momentum: (B, d_model, L) -> (B, L, d_model)
        x = x.transpose(1, 2)
        
        # Direct Mamba Momentum (much faster than separate blocks)
        x = self.mamba_momentum(x)
        
        # RMS Norm
        x = self.rms_norm(x)
        
        # Dropout
        x = self.dropout2(x)
        
        # Global Average Pooling: (B, L, d_model) -> (B, d_model)
        x = x.mean(dim=1)
        
        # Linear classifier: d_model -> num_classes
        x = self.classifier(x)
        
        return x
    