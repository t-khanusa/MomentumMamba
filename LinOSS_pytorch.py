import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

# Check if torch.vmap is available (PyTorch 2.0+)
try:
    from torch.func import vmap as torch_vmap
    VMAP_AVAILABLE = True
except ImportError:
    try:
        from functorch import vmap as torch_vmap
        VMAP_AVAILABLE = True
    except ImportError:
        VMAP_AVAILABLE = False


def simple_uniform_init(shape: Tuple[int, ...], std: float = 1.0) -> torch.Tensor:
    """Initialize weights with uniform distribution."""
    weights = torch.rand(shape) * 2.0 * std - std
    return weights


class GLU(nn.Module):
    """Gated Linear Unit with two linear transformations."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w1 = nn.Linear(input_dim, output_dim, bias=True)
        self.w2 = nn.Linear(input_dim, output_dim, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x) * torch.sigmoid(self.w2(x))


def create_identity_padding(A_elements: torch.Tensor, b_elements: torch.Tensor, 
                           pad_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create identity padding for parallel scan instead of zeros."""
    device = A_elements.device
    P = A_elements.shape[-1] // 4
    
    # Create identity matrix: M = I (no change), F = 0 (no input)
    identity_A = torch.zeros(pad_length, 4 * P, device=device)
    identity_A[..., 0*P:1*P] = 1.0  # A = I
    identity_A[..., 3*P:4*P] = 1.0  # D = I
    
    identity_b = torch.zeros(pad_length, 2 * P, device=device, dtype=b_elements.dtype)
    
    return identity_A, identity_b


def parallel_scan_batched(A_elements: torch.Tensor, b_elements: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """True parallel scan using batched operations with identity padding."""
    seq_len = A_elements.shape[0]
    device = A_elements.device
    
    if seq_len == 1:
        return A_elements, b_elements
    
    # Find next power of 2 for efficient parallel scan
    padded_len = 1
    while padded_len < seq_len:
        padded_len *= 2
    
    # Pad with identity elements instead of zeros
    if padded_len > seq_len:
        pad_len = padded_len - seq_len
        identity_A, identity_b = create_identity_padding(A_elements, b_elements, pad_len)
        A_padded = torch.cat([A_elements, identity_A], dim=0)
        b_padded = torch.cat([b_elements, identity_b], dim=0)
    else:
        A_padded = A_elements
        b_padded = b_elements
    
    # Parallel scan using log(n) passes with batched operations
    A_scan = A_padded.clone()
    b_scan = b_padded.clone()
    
    step = 1
    while step < padded_len:
        indices = torch.arange(padded_len, device=device)
        valid_mask = indices >= step
        shifted_indices = torch.clamp(indices - step, min=0)
        
        A_prev = A_scan[shifted_indices]
        b_prev = b_scan[shifted_indices]
        
        A_new, b_new = binary_operator_batched(A_prev, b_prev, A_scan, b_scan)
        
        A_scan = torch.where(valid_mask.unsqueeze(-1), A_new, A_scan)
        b_scan = torch.where(valid_mask.unsqueeze(-1), b_new, b_scan)
        
        step *= 2
    
    return A_scan[:seq_len], b_scan[:seq_len]


def binary_operator_batched(A_prev: torch.Tensor, b_prev: torch.Tensor, 
                           A_curr: torch.Tensor, b_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully batched binary operator for parallel scan."""
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


def complex_matmul_real_input(B_real: torch.Tensor, B_imag: torch.Tensor, 
                             input_real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (B_real + i*B_imag) @ input_real using real arithmetic."""
    # (a + bi) * c = ac + bci
    result_real = torch.einsum('ph,lh->lp', B_real, input_real)
    result_imag = torch.einsum('ph,lh->lp', B_imag, input_real)
    return result_real, result_imag


def complex_matmul_complex_state(C_real: torch.Tensor, C_imag: torch.Tensor,
                                state_real: torch.Tensor, state_imag: torch.Tensor) -> torch.Tensor:
    """Compute real part of (C_real + i*C_imag) @ (state_real + i*state_imag)."""
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    # We only need the real part: ac - bd
    real_part = torch.einsum('hp,lp->lh', C_real, state_real) - torch.einsum('hp,lp->lh', C_imag, state_imag)
    return real_part


def apply_linoss_im_complex(A_diag: torch.Tensor, B_real: torch.Tensor, B_imag: torch.Tensor,
                           C_real: torch.Tensor, C_imag: torch.Tensor,
                           input_sequence: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Apply LinOSS-IM using proper complex arithmetic like JAX version."""
    P = A_diag.shape[0]
    seq_len, H = input_sequence.shape
    
    # Compute Bu elements: B_complex @ input_sequence (complex result)
    Bu_real, Bu_imag = complex_matmul_real_input(B_real, B_imag, input_sequence)
    
    # Compute Schur complement matrices (all real)
    schur_comp = 1.0 / (1.0 + step ** 2 * A_diag)
    M_IM_11 = 1.0 - step ** 2 * A_diag * schur_comp
    M_IM_12 = -step * A_diag * schur_comp
    M_IM_21 = step * schur_comp
    M_IM_22 = schur_comp
    
    # Create M matrix (real)
    M_IM = torch.cat([M_IM_11, M_IM_12, M_IM_21, M_IM_22])
    M_elements = M_IM.unsqueeze(0).expand(seq_len, -1)
    
    # Compute F vector (complex): F = [F1_real + i*F1_imag, F2_real + i*F2_imag]
    F1_real = M_IM_11.unsqueeze(0) * Bu_real * step.unsqueeze(0)
    F1_imag = M_IM_11.unsqueeze(0) * Bu_imag * step.unsqueeze(0)
    F2_real = M_IM_21.unsqueeze(0) * Bu_real * step.unsqueeze(0)
    F2_imag = M_IM_21.unsqueeze(0) * Bu_imag * step.unsqueeze(0)
    
    # Combine real and imaginary parts for parallel scan
    F_real = torch.cat([F1_real, F2_real], dim=1)
    F_imag = torch.cat([F1_imag, F2_imag], dim=1)
    
    # Apply parallel scan to both real and imaginary parts
    _, xs_real = parallel_scan_batched(M_elements, F_real)
    _, xs_imag = parallel_scan_batched(M_elements, F_imag)
    
    # Extract state (second half) - this is complex
    ys_real = xs_real[:, P:]
    ys_imag = xs_imag[:, P:]
    
    # Apply output transformation: C_complex @ ys_complex, take real part
    output = complex_matmul_complex_state(C_real, C_imag, ys_real, ys_imag)
    
    return output


def apply_linoss_imex_complex(A_diag: torch.Tensor, B_real: torch.Tensor, B_imag: torch.Tensor,
                             C_real: torch.Tensor, C_imag: torch.Tensor,
                             input_sequence: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Apply LinOSS-IMEX using proper complex arithmetic like JAX version."""
    P = A_diag.shape[0]
    seq_len, H = input_sequence.shape
    
    # Compute Bu elements: B_complex @ input_sequence (complex result)
    Bu_real, Bu_imag = complex_matmul_real_input(B_real, B_imag, input_sequence)
    
    # Define IMEX matrices (all real)
    A_ = torch.ones_like(A_diag)
    B_ = -step * A_diag
    C_ = step * torch.ones_like(A_diag)
    D_ = 1.0 - (step ** 2) * A_diag
    
    # Create M matrix (real)
    M_IMEX = torch.cat([A_, B_, C_, D_])
    M_elements = M_IMEX.unsqueeze(0).expand(seq_len, -1)
    
    # Compute F vector (complex)
    F1_real = Bu_real * step.unsqueeze(0)
    F1_imag = Bu_imag * step.unsqueeze(0)
    F2_real = Bu_real * (step ** 2).unsqueeze(0)
    F2_imag = Bu_imag * (step ** 2).unsqueeze(0)
    
    # Combine real and imaginary parts for parallel scan
    F_real = torch.cat([F1_real, F2_real], dim=1)
    F_imag = torch.cat([F1_imag, F2_imag], dim=1)
    
    # Apply parallel scan to both real and imaginary parts
    _, xs_real = parallel_scan_batched(M_elements, F_real)
    _, xs_imag = parallel_scan_batched(M_elements, F_imag)
    
    # Extract state (second half) - this is complex
    ys_real = xs_real[:, P:]
    ys_imag = xs_imag[:, P:]
    
    # Apply output transformation: C_complex @ ys_complex, take real part
    output = complex_matmul_complex_state(C_real, C_imag, ys_real, ys_imag)
    
    return output


class LinOSSLayer(nn.Module):
    """Core LinOSS state space layer with proper complex arithmetic matching JAX."""
    
    def __init__(self, ssm_size: int, H: int, discretization: str):
        super().__init__()
        self.discretization = discretization
        
        # Initialize parameters exactly like JAX: separate real and imaginary parts
        self.A_diag = nn.Parameter(torch.rand(ssm_size))
        self.B_real = nn.Parameter(simple_uniform_init((ssm_size, H), std=1.0/math.sqrt(H)))
        self.B_imag = nn.Parameter(simple_uniform_init((ssm_size, H), std=1.0/math.sqrt(H)))
        self.C_real = nn.Parameter(simple_uniform_init((H, ssm_size), std=1.0/math.sqrt(ssm_size)))
        self.C_imag = nn.Parameter(simple_uniform_init((H, ssm_size), std=1.0/math.sqrt(ssm_size)))
        self.D = nn.Parameter(torch.randn(H))
        self.steps = nn.Parameter(torch.rand(ssm_size))
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper complex arithmetic matching JAX implementation."""
        A_diag = F.relu(self.A_diag)
        steps = torch.sigmoid(self.steps)
        
        # Handle both single sequence and batch
        if input_sequence.dim() == 2:
            # Single sequence: (L, H)
            if self.discretization == 'IMEX':
                ys = apply_linoss_imex_complex(A_diag, self.B_real, self.B_imag,
                                              self.C_real, self.C_imag, input_sequence, steps)
            elif self.discretization == 'IM':
                ys = apply_linoss_im_complex(A_diag, self.B_real, self.B_imag,
                                            self.C_real, self.C_imag, input_sequence, steps)
            else:
                raise ValueError(f'Discretization type {self.discretization} not implemented')
            
            # Add feedthrough
            Du = input_sequence * self.D.unsqueeze(0)
            return ys + Du
            
        elif input_sequence.dim() == 3:
            # Batch: (B, L, H)
            if VMAP_AVAILABLE:
                # Use torch.vmap for vectorization
                def single_sequence_fn(seq):
                    if self.discretization == 'IMEX':
                        ys = apply_linoss_imex_complex(A_diag, self.B_real, self.B_imag,
                                                      self.C_real, self.C_imag, seq, steps)
                    else:  # 'IM'
                        ys = apply_linoss_im_complex(A_diag, self.B_real, self.B_imag,
                                                    self.C_real, self.C_imag, seq, steps)
                    return ys + seq * self.D.unsqueeze(0)
                
                return torch_vmap(single_sequence_fn)(input_sequence)
            else:
                # Fallback to loop-based processing
                outputs = []
                for i in range(input_sequence.shape[0]):
                    single_seq = input_sequence[i]
                    
                    if self.discretization == 'IMEX':
                        ys = apply_linoss_imex_complex(A_diag, self.B_real, self.B_imag,
                                                      self.C_real, self.C_imag, single_seq, steps)
                    else:  # 'IM'
                        ys = apply_linoss_im_complex(A_diag, self.B_real, self.B_imag,
                                                    self.C_real, self.C_imag, single_seq, steps)
                    
                    Du = single_seq * self.D.unsqueeze(0)
                    outputs.append(ys + Du)
                
                return torch.stack(outputs, dim=0)
        
        else:
            raise ValueError(f"Input must be 2D (L, H) or 3D (B, L, H), got {input_sequence.dim()}D")


class LinOSSBlock(nn.Module):
    """LinOSS block with configurable normalization (BatchNorm or LayerNorm)."""
    
    def __init__(self, ssm_size: int, H: int, discretization: str, 
                 drop_rate: float = 0.05, norm_type: str = 'batch'):
        super().__init__()
        
        self.norm_type = norm_type
        self.H = H
        
        # Use BatchNorm by default to match original JAX implementation
        if norm_type == 'batch':
            # JAX BatchNorm with axis_name="batch" normalizes across the batch dimension
            # After transpose x.T: (L, H) -> (H, L), it normalizes each of the H features
            # across the L sequence positions within each batch
            # We'll implement this using GroupNorm with num_groups=H (each feature is its own group)
            # This way each feature is normalized independently across the sequence
            self.norm = nn.GroupNorm(num_groups=H, num_channels=H, affine=False)
        elif norm_type == 'layer':
            # LayerNorm works directly on the last dimension
            self.norm = nn.LayerNorm(H, elementwise_affine=False)
        else:
            raise ValueError(f"norm_type must be 'batch' or 'layer', got {norm_type}")
            
        self.ssm = LinOSSLayer(ssm_size, H, discretization)
        self.glu = GLU(H, H)
        self.dropout = nn.Dropout(p=drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with configurable normalization."""
        skip = x
        
        if self.norm_type == 'batch':
            # Implement JAX-style BatchNorm with transpose
            if x.dim() == 2:  # Single sequence (L, H)
                # JAX: x.T -> (H, L), normalize, then x.T -> (L, H)
                # For GroupNorm, we need (N, C, *) format
                # So (L, H) -> (1, H, L) -> normalize -> (1, H, L) -> (L, H)
                x_transposed = x.T.unsqueeze(0)  # (L, H) -> (H, L) -> (1, H, L)
                x_normalized = self.norm(x_transposed)  # GroupNorm on (1, H, L)
                x = x_normalized.squeeze(0).T  # (1, H, L) -> (H, L) -> (L, H)
            else:  # Batch (B, L, H)
                # JAX: x.T -> (B, H, L), normalize, then x.T -> (B, L, H)
                B, L, H = x.shape
                x_transposed = x.transpose(1, 2)  # (B, L, H) -> (B, H, L)
                x_normalized = self.norm(x_transposed)  # GroupNorm on (B, H, L)
                x = x_normalized.transpose(1, 2)  # (B, H, L) -> (B, L, H)
        else:  # LayerNorm
            # LayerNorm works directly on the last dimension
            x = self.norm(x)
        
        x = self.ssm(x)
        x = self.dropout(F.gelu(x))
        x = self.glu(x)
        x = self.dropout(x)
        
        return skip + x


class LinOSS(nn.Module):
    """Complete LinOSS model with configurable normalization."""
    
    def __init__(self, num_blocks: int, N: int, ssm_size: int, H: int, 
                 output_dim: int, classification: bool, output_step: int, 
                 discretization: str, norm_type: str = 'batch'):
        super().__init__()
        self.classification = classification
        self.output_step = output_step
        
        self.linear_encoder = nn.Linear(N, H)
        self.blocks = nn.ModuleList([
            LinOSSBlock(ssm_size, H, discretization, norm_type=norm_type) 
            for _ in range(num_blocks)
        ])
        self.linear_layer = nn.Linear(H, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper output format."""
        x = self.linear_encoder(x)
        
        for block in self.blocks:
            x = block(x)
        
        if self.classification:
            # Average over sequence dimension
            if x.dim() == 2:
                x = torch.mean(x, dim=0)
            else:
                x = torch.mean(x, dim=1)
            
            # Return raw logits for classification
            return self.linear_layer(x)
        else:
            # Sequence prediction with subsampling
            x = x[..., self.output_step - 1::self.output_step, :]
            return torch.tanh(self.linear_layer(x)) 