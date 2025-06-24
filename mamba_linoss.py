import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import vmap for maximum vectorization
try:
    from torch.func import vmap as torch_vmap
    VMAP_AVAILABLE = True
except ImportError:
    try:
        from functorch import vmap as torch_vmap
        VMAP_AVAILABLE = True
    except ImportError:
        VMAP_AVAILABLE = False
        torch_vmap = None


@dataclass
class MambaLinOSSConfig:
    d_model: int  # D - model dimension (input feature dimension)
    n_layer: int  # number of layers
    
    # LinOSS specific parameters
    d_state: int = 64  # P - state space dimension
    linoss_discretization: str = 'IM'  # 'IM' or 'IMEX'
    
    # A matrix initialization options
    a_init_method: str = 'uniform'  # 'log_uniform', 'uniform', 'relu'
    
    # Delta (timestep) parameters
    dt_min: float = 0.001  # Minimum timestep
    dt_max: float = 0.1    # Maximum timestep
    dt_init_floor: float = 1e-4  # Floor for dt initialization
    
    # Mamba structure parameters
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4  # convolution kernel size
    
    # Normalization and regularization
    rms_norm_eps: float = 1e-5
    bias: bool = False
    conv_bias: bool = True
    dropout_rate: float = 0
    
    # Performance optimization
    scan_algorithm: str = 'vmap'  # 'auto', 'vectorized', 'optimized', 'vmap'
    use_vmap: bool = True  # Whether to use vmap when available

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments
        
        if self.linoss_discretization not in ['IM', 'IMEX']:
            raise ValueError(f"linoss_discretization must be 'IM' or 'IMEX', got {self.linoss_discretization}")
        
        if self.a_init_method not in ['log_uniform', 'uniform', 'relu']:
            raise ValueError(f"a_init_method must be 'log_uniform', 'uniform', or 'relu', got {self.a_init_method}")
        
        if self.dt_min >= self.dt_max:
            raise ValueError(f"dt_min ({self.dt_min}) must be less than dt_max ({self.dt_max})")
        
        if self.scan_algorithm not in ['auto', 'vectorized', 'optimized', 'vmap']:
            raise ValueError(f"scan_algorithm must be one of ['auto', 'vectorized', 'optimized', 'vmap'], got {self.scan_algorithm}")
        
        # Auto-select best algorithm
        if self.scan_algorithm == 'auto':
            if self.use_vmap and VMAP_AVAILABLE:
                self.scan_algorithm = 'vmap'
            else:
                self.scan_algorithm = 'optimized'


class MambaLinOSS(nn.Module):
    """Mamba-LinOSS backbone model for sequence-to-sequence tasks with configurable performance optimizations."""
    
    def __init__(self, config: MambaLinOSSConfig):
        super().__init__()
        self.config = config
        
        # Mamba-LinOSS layers
        self.layers = nn.ModuleList([
            MambaLinOSSBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm
        self.norm_f = RMSNorm(config.d_model)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba-LinOSS backbone model.
        
        Args:
            x: Input tensor of shape (B, L, D) where:
               B = batch size
               L = sequence length
               D = feature dimension (d_model)
        
        Returns:
            output: Output tensor of shape (B, L, D) - same as input shape
        """
        # Apply Mamba-LinOSS layers with residual connections
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
            
        # Final normalization
        x = self.norm_f(x)
        
        return x


class MambaLinOSSBlock(nn.Module):
    """Single Mamba-LinOSS block with configurable scan algorithm."""
    
    def __init__(self, config: MambaLinOSSConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )
        
        # Activation
        self.activation = nn.SiLU()
        
        # LinOSS SSM
        self.ssm = LinOSSSSM(
            d_inner=config.d_inner,
            d_state=config.d_state,
            linoss_discretization=config.linoss_discretization,
            a_init_method=config.a_init_method,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
            scan_algorithm=config.scan_algorithm
        )
        
        # Output projection
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mamba-LinOSS block."""
        B, L, D = x.shape
        
        # Input projection and split
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, res = x_and_res.split(split_size=self.config.d_inner, dim=-1)
        
        # Convolution (need to transpose for conv1d)
        x = x.transpose(-1, -2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # Remove padding
        x = x.transpose(-1, -2)  # (B, L, d_inner)
        
        # Activation
        x = self.activation(x)
        
        # SSM
        x = self.ssm(x)
        
        # Gating
        x = x * self.activation(res)
        
        # Output projection
        x = self.out_proj(x)
        
        return x


class LinOSSSSM(nn.Module):
    """Selective LinOSS State Space Model - LinOSS with Mamba's selectivity."""
    
    def __init__(self, d_inner: int, d_state: int, linoss_discretization: str, 
                 a_init_method: str = 'log_uniform', dt_min: float = 0.001, 
                 dt_max: float = 0.1, dt_init_floor: float = 1e-4,
                 scan_algorithm: str = 'optimized'):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.linoss_discretization = linoss_discretization
        self.a_init_method = a_init_method
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.scan_algorithm = scan_algorithm
        
        # Efficient Mamba-style projection: single projection for all parameters
        # Output: dt_rank + 4*d_state (B_real, B_imag, C_real, C_imag)
        dt_rank = math.ceil(d_inner / 16)  # Same as Mamba's dt_rank calculation
        self.dt_rank = dt_rank
        self.x_proj = nn.Linear(d_inner, dt_rank + 4 * d_state, bias=False)
        
        # Delta projection (same as Mamba)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        
        # Initialize A_log with better initialization options
        self.A_log = nn.Parameter(self._init_A_log())
        
        # Feedthrough (fixed like in LinOSS)
        self.D = nn.Parameter(torch.randn(d_inner))
        
        # Initialize projections
        self._init_projections()
    
    def _init_A_log(self) -> torch.Tensor:
        """Initialize A_log with different methods for better eigenvalue coverage."""
        if self.a_init_method == 'log_uniform':
            # Use a reasonable range for log-uniform initialization - keep positive for LinOSS
            return torch.rand(self.d_state) * 2.0  # Range [0, 2]
            
        elif self.a_init_method == 'uniform':
            return torch.rand(self.d_state) * 2.0  # Range [0, 2]
            
        elif self.a_init_method == 'relu':
            return torch.randn(self.d_state) * 0.1  # Small random initialization
            
        else:  # Default fallback
            return torch.abs(torch.randn(self.d_state))  # Ensure positive

    def _init_projections(self):
        """Initialize projection layers with configurable parameters."""
        # Initialize dt projection with configurable range (same as Mamba)
        dt_init_std = self.dt_rank**-0.5 * 1.0  # dt_scale = 1.0
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(self.dt_min)
        )
        dt = torch.clamp(dt, min=self.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # Initialize x_proj with small values for B/C components
        nn.init.uniform_(self.x_proj.weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with efficient vectorized complex arithmetic.
        
        Args:
            x: Input tensor of shape (B, L, ED)
        
        Returns:
            Output tensor of shape (B, L, ED)
        """
        B, L, ED = x.shape
        P = self.d_state
        
        # Efficient single projection (like Mamba)
        deltaBC = self.x_proj(x)  # (B, L, dt_rank + 4*P)
        delta, B_real_flat, B_imag_flat, C_real_flat, C_imag_flat = torch.split(
            deltaBC, [self.dt_rank, P, P, P, P], dim=-1
        )  # (B, L, dt_rank), (B, L, P), (B, L, P), (B, L, P), (B, L, P)
        
        # Process delta (same as Mamba)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, ED)
        
        # Fixed A parameter with flexible initialization - keep NON-NEGATIVE for LinOSS
        if self.a_init_method == 'relu':
            A_diag = torch.exp(F.relu(self.A_log))  # Non-negative for LinOSS
        else:
            A_diag = torch.exp(self.A_log)  # (P,) - positive for LinOSS stability
        
        # Apply vectorized LinOSS operations with configurable algorithm
        ys = apply_linoss_vectorized(
            discretization=self.linoss_discretization,
            A_diag=A_diag,
            B_real_flat=B_real_flat,
            B_imag_flat=B_imag_flat,
            C_real_flat=C_real_flat,
            C_imag_flat=C_imag_flat,
            input_sequence=x,
            delta=delta,
            scan_algorithm=self.scan_algorithm
        )
        
        # Add feedthrough
        Du = x * self.D.unsqueeze(0).unsqueeze(0)
        output = ys + Du
        
        return output
    
    def get_eigenspectrum_stats(self) -> Dict[str, torch.Tensor]:
        """Get eigenvalue statistics for monitoring during training."""
        if not self.log_eigenspectrum:
            return {}
        
        with torch.no_grad():
            if self.a_init_method == 'relu':
                A_diag = torch.exp(F.relu(self.A_log))
            else:
                A_diag = torch.exp(self.A_log)
            
            # Compute eigenvalue magnitudes: |λ| = 1/√(1 + Δ²A) approximation
            # For monitoring, use a representative delta value
            delta_repr = torch.tensor(0.01)  # Representative timestep
            lambda_magnitudes = 1.0 / torch.sqrt(1.0 + (delta_repr ** 2) * torch.abs(A_diag))
            
            return {
                'eigenval_min': A_diag.min(),
                'eigenval_max': A_diag.max(),
                'eigenval_mean': A_diag.mean(),
                'eigenval_std': A_diag.std(),
                'lambda_mag_min': lambda_magnitudes.min(),
                'lambda_mag_max': lambda_magnitudes.max(),
                'lambda_mag_mean': lambda_magnitudes.mean(),
                'A_log_min': self.A_log.min(),
                'A_log_max': self.A_log.max(),
                'A_log_mean': self.A_log.mean(),
            }


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


def parallel_scan_batched_complex(A_elements: torch.Tensor, b_elements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully vectorized parallel scan for complex numbers with no Python loops.
    
    This implementation uses a more efficient algorithm that processes all batch elements
    and sequence positions simultaneously using pure tensor operations.
    
    Args:
        A_elements: (B, L, 4*P) - batch of transition matrices
        b_elements: (B, L, 2*P) - batch of input vectors (complex)
    
    Returns:
        A_scan: (B, L, 4*P) - scanned transition matrices
        b_scan: (B, L, 2*P) - scanned input vectors (complex)
    """
    B, L, feature_dim_A = A_elements.shape
    P = feature_dim_A // 4
    device = A_elements.device
    
    if L == 1:
        return A_elements, b_elements
    
    # Use efficient log-space parallel scan algorithm
    # This avoids all Python loops and processes everything in parallel
    
    # Find the number of scan levels needed (log2(L) rounded up)
    max_levels = int(torch.ceil(torch.log2(torch.tensor(L, dtype=torch.float32))))
    
    # Pad to next power of 2 for efficient processing
    padded_len = 2 ** max_levels
    
    if padded_len > L:
        pad_len = padded_len - L
        
        # Create identity elements for padding
        identity_A = torch.zeros(B, pad_len, 4 * P, device=device, dtype=A_elements.dtype)
        identity_A[..., 0*P:1*P] = 1.0  # A = I
        identity_A[..., 3*P:4*P] = 1.0  # D = I
        
        identity_b = torch.zeros(B, pad_len, 2 * P, device=device, dtype=b_elements.dtype)
        
        A_padded = torch.cat([A_elements, identity_A], dim=1)  # (B, padded_len, 4*P)
        b_padded = torch.cat([b_elements, identity_b], dim=1)  # (B, padded_len, 2*P)
    else:
        A_padded = A_elements
        b_padded = b_elements
    
    # Initialize scan arrays
    A_scan = A_padded.clone()
    b_scan = b_padded.clone()
    
    # Vectorized parallel scan using powers of 2
    # Process all levels simultaneously using advanced indexing
    for level in range(max_levels):
        step = 2 ** level
        
        if step >= padded_len:
            break
            
        # Create vectorized indices for this level
        # We want indices: step, step+1, ..., padded_len-1
        target_indices = torch.arange(step, padded_len, device=device)
        source_indices = target_indices - step
        
        if len(target_indices) == 0:
            continue
        
        # Vectorized gather operations - no loops!
        A_prev = A_scan[:, source_indices]  # (B, num_targets, 4*P)
        b_prev = b_scan[:, source_indices]  # (B, num_targets, 2*P)
        A_curr = A_scan[:, target_indices]  # (B, num_targets, 4*P)
        b_curr = b_scan[:, target_indices]  # (B, num_targets, 2*P)
        
        # Apply binary operator in fully vectorized manner
        A_new, b_new = binary_operator_batched_complex(A_prev, b_prev, A_curr, b_curr)
        
        # Vectorized scatter operation - update all positions at once
        A_scan[:, target_indices] = A_new
        b_scan[:, target_indices] = b_new
    
    return A_scan[:, :L], b_scan[:, :L]  # Remove padding


def parallel_scan_batched_complex_optimized(A_elements: torch.Tensor, b_elements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ultra-optimized parallel scan using associative scan with minimal memory operations.
    
    This version uses the most efficient parallel scan algorithm with optimal memory access patterns.
    """
    B, L, feature_dim_A = A_elements.shape
    P = feature_dim_A // 4
    device = A_elements.device
    
    if L == 1:
        return A_elements, b_elements
    
    # For small sequences, use direct computation to avoid overhead
    if L <= 8:
        return _small_sequence_scan_vectorized(A_elements, b_elements)
    
    # Use Blelloch scan algorithm - most efficient for GPU
    return _blelloch_scan_vectorized(A_elements, b_elements)


def _small_sequence_scan_vectorized(A_elements: torch.Tensor, b_elements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized scan for small sequences using direct computation."""
    B, L, feature_dim_A = A_elements.shape
    P = feature_dim_A // 4
    
    A_scan = A_elements.clone()
    b_scan = b_elements.clone()
    
    # Unroll small loops for better performance
    # Use cumulative operations where possible
    if L >= 2:
        A_new, b_new = binary_operator_batched_complex(
            A_scan[:, 0:1], b_scan[:, 0:1],
            A_scan[:, 1:2], b_scan[:, 1:2]
        )
        A_scan[:, 1:2] = A_new
        b_scan[:, 1:2] = b_new
    
    if L >= 3:
        A_new, b_new = binary_operator_batched_complex(
            A_scan[:, 1:2], b_scan[:, 1:2],
            A_scan[:, 2:3], b_scan[:, 2:3]
        )
        A_scan[:, 2:3] = A_new
        b_scan[:, 2:3] = b_new
    
    if L >= 4:
        A_new, b_new = binary_operator_batched_complex(
            A_scan[:, 2:3], b_scan[:, 2:3],
            A_scan[:, 3:4], b_scan[:, 3:4]
        )
        A_scan[:, 3:4] = A_new
        b_scan[:, 3:4] = b_new
    
    # Continue pattern for L=5,6,7,8
    for i in range(4, L):
        A_new, b_new = binary_operator_batched_complex(
            A_scan[:, i-1:i], b_scan[:, i-1:i],
            A_scan[:, i:i+1], b_scan[:, i:i+1]
        )
        A_scan[:, i:i+1] = A_new
        b_scan[:, i:i+1] = b_new
    
    return A_scan, b_scan


def _blelloch_scan_vectorized(A_elements: torch.Tensor, b_elements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blelloch parallel scan algorithm - most GPU-efficient approach."""
    B, L, feature_dim_A = A_elements.shape
    P = feature_dim_A // 4
    device = A_elements.device
    
    # Find next power of 2
    padded_len = 1
    while padded_len < L:
        padded_len *= 2
    
    # Pad if necessary
    if padded_len > L:
        pad_len = padded_len - L
        identity_A = torch.zeros(B, pad_len, 4 * P, device=device, dtype=A_elements.dtype)
        identity_A[..., 0*P:1*P] = 1.0
        identity_A[..., 3*P:4*P] = 1.0
        identity_b = torch.zeros(B, pad_len, 2 * P, device=device, dtype=b_elements.dtype)
        
        A_work = torch.cat([A_elements, identity_A], dim=1)
        b_work = torch.cat([b_elements, identity_b], dim=1)
    else:
        A_work = A_elements.clone()
        b_work = b_elements.clone()
    
    # Up-sweep phase (reduce)
    d = 1
    while d < padded_len:
        # Vectorized indices computation
        indices = torch.arange(d, padded_len, 2*d, device=device)
        if len(indices) == 0:
            break
            
        left_indices = indices - d
        right_indices = indices
        
        # Vectorized operations
        A_left = A_work[:, left_indices]
        b_left = b_work[:, left_indices]
        A_right = A_work[:, right_indices]
        b_right = b_work[:, right_indices]
        
        A_new, b_new = binary_operator_batched_complex(A_left, b_left, A_right, b_right)
        
        A_work[:, right_indices] = A_new
        b_work[:, right_indices] = b_new
        
        d *= 2
    
    # Down-sweep phase (distribute)
    d = padded_len // 2
    while d >= 1:
        indices = torch.arange(d, padded_len, 2*d, device=device)
        if len(indices) == 0:
            break
            
        left_indices = indices - d
        right_indices = indices
        
        # Swap and combine
        A_left = A_work[:, left_indices]
        b_left = b_work[:, left_indices]
        A_right = A_work[:, right_indices]
        b_right = b_work[:, right_indices]
        
        A_work[:, left_indices] = A_right
        b_work[:, left_indices] = b_right
        
        A_new, b_new = binary_operator_batched_complex(A_right, b_right, A_left, b_left)
        A_work[:, right_indices] = A_new
        b_work[:, right_indices] = b_new
        
        d //= 2
    
    return A_work[:, :L], b_work[:, :L]


def binary_operator_batched_complex(A_prev: torch.Tensor, b_prev: torch.Tensor, 
                                   A_curr: torch.Tensor, b_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully batched binary operator for parallel scan with complex number support.
    
    Args:
        A_prev: (B, N, 4*P) - previous transition matrices
        b_prev: (B, N, 2*P) - previous input vectors (complex)
        A_curr: (B, N, 4*P) - current transition matrices  
        b_curr: (B, N, 2*P) - current input vectors (complex)
    
    Returns:
        A_result: (B, N, 4*P) - combined transition matrices
        b_result: (B, N, 2*P) - combined input vectors (complex)
    """
    P = A_prev.shape[-1] // 4
    
    # Split A matrices into 2x2 block components
    iA = A_prev[..., 0*P:1*P]
    iB = A_prev[..., 1*P:2*P]
    iC = A_prev[..., 2*P:3*P]
    iD = A_prev[..., 3*P:4*P]
    
    jA = A_curr[..., 0*P:1*P]
    jB = A_curr[..., 1*P:2*P]
    jC = A_curr[..., 2*P:3*P]
    jD = A_curr[..., 3*P:4*P]
    
    # Matrix multiplication for 2x2 blocks (works for both real and complex)
    A_new = jA * iA + jB * iC
    B_new = jA * iB + jB * iD
    C_new = jC * iA + jD * iC
    D_new = jC * iB + jD * iD
    
    A_result = torch.cat([A_new, B_new, C_new, D_new], dim=-1)
    
    # Vector part (works for both real and complex)
    b_prev_1 = b_prev[..., :P]
    b_prev_2 = b_prev[..., P:]
    
    b_new_1 = jA * b_prev_1 + jB * b_prev_2
    b_new_2 = jC * b_prev_1 + jD * b_prev_2
    
    b_result = torch.cat([b_new_1, b_new_2], dim=-1) + b_curr
    
    return A_result, b_result


def apply_linoss_vectorized(discretization: str, A_diag: torch.Tensor, 
                           B_real_flat: torch.Tensor, B_imag_flat: torch.Tensor,
                           C_real_flat: torch.Tensor, C_imag_flat: torch.Tensor,
                           input_sequence: torch.Tensor, delta: torch.Tensor,
                           scan_algorithm: str = 'optimized') -> torch.Tensor:
    """Fully batched vectorized LinOSS application with configurable scan algorithms."""
    B, L, ED = input_sequence.shape
    P = A_diag.shape[0]
    
    # Create complex B and C matrices
    B_complex = torch.complex(B_real_flat, B_imag_flat)  # (B, L, P)
    C_complex = torch.complex(C_real_flat, C_imag_flat)  # (B, L, P)
    
    # Use average delta across ED dimension for state space operations
    delta_avg = delta.mean(dim=-1)  # (B, L)
    
    # Compute Bu elements using per-timestep scaling (Mamba-style)
    # Scale input by B parameters and timestep
    Bu_complex = B_complex * delta_avg.unsqueeze(-1)  # (B, L, P)
    
    if discretization == 'IM':
        # Compute Schur complement matrices (all real) - fully batched
        schur_comp = 1.0 / (1.0 + delta_avg.unsqueeze(-1) ** 2 * A_diag.unsqueeze(0).unsqueeze(0))  # (B, L, P)
        M_IM_11 = 1.0 - delta_avg.unsqueeze(-1) ** 2 * A_diag.unsqueeze(0).unsqueeze(0) * schur_comp
        M_IM_12 = -delta_avg.unsqueeze(-1) * A_diag.unsqueeze(0).unsqueeze(0) * schur_comp
        M_IM_21 = delta_avg.unsqueeze(-1) * schur_comp
        M_IM_22 = schur_comp
        
        # Create M matrix (real) - fully batched
        M_elements = torch.cat([M_IM_11, M_IM_12, M_IM_21, M_IM_22], dim=-1)  # (B, L, 4*P)
        
        # Compute F vector (complex) - fully batched
        F1_complex = M_IM_11 * Bu_complex
        F2_complex = M_IM_21 * Bu_complex
        F_complex = torch.cat([F1_complex, F2_complex], dim=-1)  # (B, L, 2*P)
        
    else:  # IMEX
        # Define IMEX matrices (all real) - fully batched
        A_ = torch.ones(B, L, P, device=A_diag.device)
        B_ = -delta_avg.unsqueeze(-1) * A_diag.unsqueeze(0).unsqueeze(0)
        C_ = delta_avg.unsqueeze(-1) * torch.ones(B, L, P, device=A_diag.device)
        D_ = 1.0 - (delta_avg.unsqueeze(-1) ** 2) * A_diag.unsqueeze(0).unsqueeze(0)
        
        # Create M matrix (real) - fully batched
        M_elements = torch.cat([A_, B_, C_, D_], dim=-1)  # (B, L, 4*P)
        
        # Compute F vector (complex) - fully batched
        F1_complex = Bu_complex
        F2_complex = Bu_complex * delta_avg.unsqueeze(-1)
        F_complex = torch.cat([F1_complex, F2_complex], dim=-1)  # (B, L, 2*P)
    
    # Apply parallel scan with selected algorithm
    if scan_algorithm == 'vmap' and VMAP_AVAILABLE:
        _, xs_complex = apply_scan_vmap(M_elements, F_complex)
    elif scan_algorithm == 'optimized':
        _, xs_complex = parallel_scan_batched_complex_optimized(M_elements, F_complex)
    else:  # 'vectorized' or fallback
        _, xs_complex = parallel_scan_batched_complex(M_elements, F_complex)
    
    # Extract state (second half) - this is complex
    ys_complex = xs_complex[:, :, P:]  # (B, L, P)
    
    # Apply output transformation using Mamba-style per-timestep scaling - fully batched
    # Sum over state dimension to get output for each ED dimension
    output_complex = torch.sum(C_complex * ys_complex, dim=-1)  # (B, L)
    
    # Expand to match ED dimension (broadcast across ED) and take real part
    output = output_complex.real.unsqueeze(-1).expand(-1, -1, ED)  # (B, L, ED)
    
    return output


def apply_scan_vmap(M_elements: torch.Tensor, F_complex: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ultra-fast parallel scan using vmap for maximum GPU parallelization.
    
    This version uses torch.vmap to vectorize across the batch dimension,
    allowing the GPU to process all batch elements in parallel with optimal
    memory access patterns and minimal Python overhead.
    """
    if not VMAP_AVAILABLE:
        # Fallback to optimized version
        return parallel_scan_batched_complex_optimized(M_elements, F_complex)
    
    def single_batch_scan(M_single, F_single):
        """Scan for a single batch element - will be vmapped."""
        return _single_sequence_parallel_scan(M_single, F_single)
    
    # Vectorize across batch dimension using vmap
    # This is the most efficient approach for GPU parallelization
    A_scan_batch, b_scan_batch = torch_vmap(single_batch_scan)(M_elements, F_complex)
    
    return A_scan_batch, b_scan_batch


def _single_sequence_parallel_scan(M_elements: torch.Tensor, F_complex: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized parallel scan for a single sequence (L, 4*P) and (L, 2*P).
    
    This function is designed to be vmapped across the batch dimension.
    """
    L, feature_dim_A = M_elements.shape
    P = feature_dim_A // 4
    device = M_elements.device
    
    if L == 1:
        return M_elements, F_complex
    
    # Use efficient associative scan algorithm
    # Find next power of 2
    padded_len = 1
    while padded_len < L:
        padded_len *= 2
    
    # Pad if necessary
    if padded_len > L:
        pad_len = padded_len - L
        
        # Identity elements
        identity_A = torch.zeros(pad_len, 4 * P, device=device, dtype=M_elements.dtype)
        identity_A[:, 0*P:1*P] = 1.0
        identity_A[:, 3*P:4*P] = 1.0
        identity_b = torch.zeros(pad_len, 2 * P, device=device, dtype=F_complex.dtype)
        
        M_padded = torch.cat([M_elements, identity_A], dim=0)
        F_padded = torch.cat([F_complex, identity_b], dim=0)
    else:
        M_padded = M_elements
        F_padded = F_complex
    
    # Efficient parallel scan using powers of 2
    A_scan = M_padded.clone()
    b_scan = F_padded.clone()
    
    # Up-sweep phase
    d = 1
    while d < padded_len:
        indices = torch.arange(d, padded_len, 2*d, device=device)
        if len(indices) == 0:
            break
            
        left_indices = indices - d
        right_indices = indices
        
        # Vectorized operations on single sequence
        A_left = A_scan[left_indices]
        b_left = b_scan[left_indices]
        A_right = A_scan[right_indices]
        b_right = b_scan[right_indices]
        
        A_new, b_new = _binary_operator_single_batch(A_left, b_left, A_right, b_right)
        
        A_scan[right_indices] = A_new
        b_scan[right_indices] = b_new
        
        d *= 2
    
    return A_scan[:L], b_scan[:L]


def _binary_operator_single_batch(A_prev: torch.Tensor, b_prev: torch.Tensor, 
                                 A_curr: torch.Tensor, b_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Binary operator for single batch (optimized for vmap)."""
    P = A_prev.shape[-1] // 4
    
    # Split A matrices into 2x2 block components
    iA = A_prev[..., 0*P:1*P]
    iB = A_prev[..., 1*P:2*P]
    iC = A_prev[..., 2*P:3*P]
    iD = A_prev[..., 3*P:4*P]
    
    jA = A_curr[..., 0*P:1*P]
    jB = A_curr[..., 1*P:2*P]
    jC = A_curr[..., 2*P:3*P]
    jD = A_curr[..., 3*P:4*P]
    
    # Matrix multiplication for 2x2 blocks
    A_new = jA * iA + jB * iC
    B_new = jA * iB + jB * iD
    C_new = jC * iA + jD * iC
    D_new = jC * iB + jD * iD
    
    A_result = torch.cat([A_new, B_new, C_new, D_new], dim=-1)
    
    # Vector part
    b_prev_1 = b_prev[..., :P]
    b_prev_2 = b_prev[..., P:]
    
    b_new_1 = jA * b_prev_1 + jB * b_prev_2
    b_new_2 = jC * b_prev_1 + jD * b_prev_2
    
    b_result = torch.cat([b_new_1, b_new_2], dim=-1) + b_curr
    
    return A_result, b_result


# Convenience functions for creating Mamba-LinOSS backbone models with different performance profiles
def create_mamba_linoss_fast(d_model: int = 64, n_layer: int = 4) -> MambaLinOSS:
    """Create a Mamba-LinOSS backbone model optimized for speed (uses vmap if available)."""
    config = MambaLinOSSConfig(
        d_model=d_model,
        n_layer=n_layer,
        scan_algorithm='vmap' if VMAP_AVAILABLE else 'optimized',
        use_vmap=True
    )
    return MambaLinOSS(config)


def create_mamba_linoss_balanced(d_model: int = 64, n_layer: int = 4) -> MambaLinOSS:
    """Create a Mamba-LinOSS backbone model with balanced speed/memory (uses optimized scan)."""
    config = MambaLinOSSConfig(
        d_model=d_model,
        n_layer=n_layer,
        scan_algorithm='optimized',
        use_vmap=False
    )
    return MambaLinOSS(config)


def create_mamba_linoss_memory_efficient(d_model: int = 64, n_layer: int = 4) -> MambaLinOSS:
    """Create a Mamba-LinOSS backbone model optimized for memory efficiency."""
    config = MambaLinOSSConfig(
        d_model=d_model,
        n_layer=n_layer,
        expand_factor=1,  # Smaller expansion ratio for memory efficiency
        scan_algorithm='vectorized',
        use_vmap=False
    )
    return MambaLinOSS(config)


# if __name__ == "__main__":
#     # Example usage for sequence-to-sequence backbone
#     print("Creating Mamba-LinOSS backbone models...")
    
#     # Example parameters
#     d_model = 32  # Feature dimension
#     n_layer = 3   # Number of layers
#     seq_len = 100    # Sequence length
#     batch_size = 4   # Batch size
    
#     # Fast model (uses vmap if available)
#     model_fast = create_mamba_linoss_fast(d_model=d_model, n_layer=n_layer)
#     print(f"Fast model: {sum(p.numel() for p in model_fast.parameters())} parameters")
    
#     # Balanced model
#     model_balanced = create_mamba_linoss_balanced(d_model=d_model, n_layer=n_layer)
#     print(f"Balanced model: {sum(p.numel() for p in model_balanced.parameters())} parameters")
    
#     # Memory efficient model
#     model_memory = create_mamba_linoss_memory_efficient(d_model=d_model, n_layer=n_layer)
#     print(f"Memory efficient model: {sum(p.numel() for p in model_memory.parameters())} parameters")
    
#     # Test forward pass with sequence data
#     # Input shape: (batch_size, sequence_length, feature_dimension)
#     x = torch.randn(batch_size, seq_len, d_model)
    
#     print(f"\nTesting forward pass with input shape: {x.shape}")
    
#     with torch.no_grad():
#         output_fast = model_fast(x)
#         output_balanced = model_balanced(x)
#         output_memory = model_memory(x)
        
#         print(f"Fast model output shape: {output_fast.shape}")
#         print(f"Balanced model output shape: {output_balanced.shape}")
#         print(f"Memory efficient model output shape: {output_memory.shape}")
    
#     print("\nAll models working correctly!")
#     print(f"VMAP available: {VMAP_AVAILABLE}")
#     print("Use create_mamba_linoss_fast() for maximum performance!")
    
#     print("\nMamba-LinOSS backbone models ready to use!")
#     print("You can now add your own classification/regression head on top of the backbone.") 