import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pscan import pscan
from .pscan_optimized import pscan_matrix_optimized

"""
Mamba with Momentum Implementation

This file closely follows the mamba.py structure while implementing momentum using the equation:
u_n = M_n * u_{n-1} + F_n

Where:
- u_n = [h_n, v_n]^T (combined state vector)
- M_n = [[A_n, β], [0, β]] (2x2 transition matrix)  
- F_n = [α*B_n*x_n, α*B_n*x_n]^T (2D input vector)

The momentum system:
h_n = A_n · h_{n-1} + v_n
v_n = β·v_{n-1} + α·B_n · x_n
y_n = C^T · h_n

This is transformed to matrix form as:
h_n = A_n · h_{n-1} + β · v_{n-1} + α·B_n·x_n
v_n = 0 · h_{n-1} + β · v_{n-1} + α·B_n·x_n
"""

@dataclass
class MambaMomentumConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments, matching original default
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4 # matching original default

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    # Momentum parameters
    momentum_beta: float = 0.6  # β in the momentum equation (decay factor)
    momentum_alpha: float = 1.0  # α in the momentum equation (input scaling)

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    pscan: bool = True # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class MambaMomentum(nn.Module):
    def __init__(self, config: MambaMomentumConfig):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, v, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, v, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaMomentumConfig):
        super().__init__()

        self.mixer = MambaMomentumBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        # x : (B, L, D)
        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, v, inputs)
                # h : (B, ED, N)
                # v : (B, ED, N) - momentum state
                # inputs : (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, v, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache


class MambaMomentumBlock(nn.Module):
    def __init__(self, config: MambaMomentumConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization - matching original exactly
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias - matching original exactly
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization - matching original exactly
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # Momentum parameters
        self.momentum_beta = config.momentum_beta
        self.momentum_alpha = config.momentum_alpha

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba - matching original exactly
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)

        x = F.silu(x)
        y = self.ssm_momentum(x)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm_momentum(self, x):
        # x : (B, L, ED)
        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        
        delta = delta.transpose(1, 2) # (B, L, ED)
        delta = F.softplus(delta + self.dt_proj.bias)

        if self.config.pscan:
            y = self.selective_scan_momentum(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_momentum_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan_momentum(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        B_batch, L, ED = x.shape
        N = A.shape[1]
        
        # Standard discretization
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)
        
        # Momentum parameters
        beta = self.momentum_beta
        alpha = self.momentum_alpha
        
        # Construct 2x2 transition matrices M_n = [[A_n, β], [0, β]]
        # Store as (B, L, ED, N, 2, 2)
        zeros = torch.zeros_like(deltaA)
        beta_tensor = torch.full_like(deltaA, beta)
        
        # M_n = [[deltaA, beta_tensor], [zeros, beta_tensor]]
        M_matrices = torch.stack([
            torch.stack([deltaA, beta_tensor], dim=-1),   # First row: [A_n, β]
            torch.stack([zeros, beta_tensor], dim=-1)     # Second row: [0, β]
        ], dim=-2)  # (B, L, ED, N, 2, 2)
        
        # Construct 2D input vectors F_n = [α*B_n*x_n, α*B_n*x_n]^T
        # Store as (B, L, ED, N, 2)
        momentum_input = alpha * deltaB * x.unsqueeze(-1)  # (B, L, ED, N)
        F_vectors = torch.stack([
            momentum_input,  # F_n[0] = α*B_n*x_n
            momentum_input   # F_n[1] = α*B_n*x_n
        ], dim=-1)  # (B, L, ED, N, 2)
        
        # Flatten for pscan_matrix: (B*ED*N, L, 2, 2) and (B*ED*N, L, 2)
        M_flat = M_matrices.permute(0, 2, 3, 1, 4, 5).contiguous().view(B_batch*ED*N, L, 2, 2)
        F_flat = F_vectors.permute(0, 2, 3, 1, 4).contiguous().view(B_batch*ED*N, L, 2)
        
        # Apply optimized pscan_matrix directly
        u_flat = pscan_matrix_optimized(M_flat, F_flat)  # (B*ED*N, L, 2)
        
        # Reshape back to (B, L, ED, N, 2)
        u_states = u_flat.view(B_batch, ED, N, L, 2).permute(0, 3, 1, 2, 4).contiguous()
        
        # Extract h_states (first component of u_n)
        h_states = u_states[..., 0]  # (B, L, ED, N)
        
        # Apply output projection
        y = (h_states @ C.unsqueeze(-1)).squeeze(3)  # (B, L, ED)
        y = y + D * x

        return y
    
    def selective_scan_momentum_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        # Standard discretization
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)
        
        # Momentum parameters
        beta = self.momentum_beta
        alpha = self.momentum_alpha
        
        # Initialize hidden and momentum states
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        v = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        
        hs = []

        for t in range(0, L):
            # Momentum input for this timestep
            momentum_input = alpha * deltaB[:, t] * x[:, t].unsqueeze(-1)  # (B, ED, N)
            
            # Update momentum: v_n = β·v_{n-1} + α·B_n·x_n
            v = beta * v + momentum_input
            
            # Update hidden state: h_n = A_n·h_{n-1} + v_n
            h = deltaA[:, t] * h + v
            
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, v, inputs)
                # h : (B, ED, N)
                # v : (B, ED, N) - momentum state
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, v, inputs)
        
        h, v, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h, v = self.ssm_momentum_step(x, h, v)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, v, inputs)
        
        return output, cache

    def ssm_momentum_step(self, x, h, v):
        # x : (B, ED)
        # h : (B, ED, N)
        # v : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)
        # v : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        # Momentum parameters
        beta = self.momentum_beta
        alpha = self.momentum_alpha

        # Momentum input for this timestep
        momentum_input = alpha * deltaB * x.unsqueeze(-1) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        if v is None:
            v = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        # Update momentum: v_n = β·v_{n-1} + α·B_n·x_n
        v = beta * v + momentum_input
        
        # Update hidden state: h_n = A_n·h_{n-1} + v_n
        h = deltaA * h + v

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)
        y = y + D * x

        return y, h, v


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output 