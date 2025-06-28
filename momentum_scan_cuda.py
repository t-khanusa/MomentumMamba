#!/usr/bin/env python3
"""
CUDA-optimized Momentum Scan Implementation

This module provides high-performance CUDA kernels for momentum scanning in Mamba architectures.
It implements the momentum equations:
- v_n = β·v_{n-1} + input_n  (momentum accumulation)
- h_n = A_n·h_{n-1} + v_n    (hidden state with momentum)

The CUDA implementation provides significant speedup over CPU implementations
while maintaining numerical accuracy and proper gradient computation.
"""

import torch
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void momentum_scan_kernel(
    const float* __restrict__ deltaA,
    const float beta,
    const float* __restrict__ input,
    float* __restrict__ output_h,
    const int batch_size,
    const int seq_len
) {
    const int idx = blockIdx.x;
    
    if (idx >= batch_size) return;
    
    const int offset = idx * seq_len;
    
    if (threadIdx.x == 0) {
        float h = 0.0f;
        float v = 0.0f;
        
        #pragma unroll 4
        for (int i = 0; i < seq_len; i++) {
            v = beta * v + input[offset + i];
            h = deltaA[offset + i] * h + v;
            output_h[offset + i] = h;
        }
    }
}

__global__ void momentum_scan_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ deltaA,
    const float beta,
    const float* __restrict__ output_h,
    float* __restrict__ grad_deltaA,
    float* __restrict__ grad_input,
    const int batch_size,
    const int seq_len
) {
    const int idx = blockIdx.x;
    
    if (idx >= batch_size) return;
    
    const int offset = idx * seq_len;
    
    if (threadIdx.x == 0) {
        float grad_h = 0.0f;
        float grad_v = 0.0f;
        
        for (int t = seq_len - 1; t >= 0; t--) {
            grad_h += grad_output[offset + t];
            grad_v += grad_h;
            
            if (grad_input) {
                grad_input[offset + t] = grad_v;
            }
            
            if (grad_deltaA && t > 0) {
                grad_deltaA[offset + t] = grad_h * output_h[offset + t - 1];
            }
            
            if (t > 0) {
                grad_h *= deltaA[offset + t];
                grad_v *= beta;
            }
        }
    }
}

void launch_forward_cuda(
    torch::Tensor deltaA,
    float beta,
    torch::Tensor input,
    torch::Tensor output_h
) {
    const int batch_size = deltaA.size(0);
    const int seq_len = deltaA.size(1);
    
    momentum_scan_kernel<<<batch_size, 32>>>(
        deltaA.data_ptr<float>(),
        beta,
        input.data_ptr<float>(),
        output_h.data_ptr<float>(),
        batch_size,
        seq_len
    );
    
    cudaDeviceSynchronize();
}

void launch_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor deltaA,
    float beta,
    torch::Tensor output_h,
    torch::Tensor grad_deltaA,
    torch::Tensor grad_input
) {
    const int batch_size = deltaA.size(0);
    const int seq_len = deltaA.size(1);
    
    momentum_scan_backward_kernel<<<batch_size, 32>>>(
        grad_output.data_ptr<float>(),
        deltaA.data_ptr<float>(),
        beta,
        output_h.data_ptr<float>(),
        grad_deltaA.numel() > 0 ? grad_deltaA.data_ptr<float>() : nullptr,
        grad_input.numel() > 0 ? grad_input.data_ptr<float>() : nullptr,
        batch_size,
        seq_len
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_forward_cuda(
    torch::Tensor deltaA,
    float beta,
    torch::Tensor input,
    torch::Tensor output_h
);

void launch_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor deltaA,
    float beta,
    torch::Tensor output_h,
    torch::Tensor grad_deltaA,
    torch::Tensor grad_input
);

void launch_forward(
    torch::Tensor deltaA,
    float beta,
    torch::Tensor input,
    torch::Tensor output_h
) {
    launch_forward_cuda(deltaA, beta, input, output_h);
}

void launch_backward(
    torch::Tensor grad_output,
    torch::Tensor deltaA,
    float beta,
    torch::Tensor output_h,
    torch::Tensor grad_deltaA,
    torch::Tensor grad_input
) {
    launch_backward_cuda(grad_output, deltaA, beta, output_h, grad_deltaA, grad_input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_forward", &launch_forward, "Momentum scan forward pass");
    m.def("launch_backward", &launch_backward, "Momentum scan backward pass");
}
"""

def load_momentum_cuda():
    """Load optimized momentum CUDA kernel"""
    try:
        return load_inline(
            name="momentum_scan_cuda_kernel", 
            cpp_sources=[cpp_source],
            cuda_sources=[cuda_source],
            verbose=False,
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
    except Exception as e:
        print(f"Failed to load momentum CUDA kernel: {e}")
        return None

_momentum_kernel = None

def get_momentum_kernel():
    global _momentum_kernel
    if _momentum_kernel is None:
        _momentum_kernel = load_momentum_cuda()
    return _momentum_kernel

class MomentumScanCUDA(torch.autograd.Function):
    """
    CUDA-accelerated momentum scan with automatic differentiation
    """
    @staticmethod
    def forward(ctx, deltaA, beta, momentum_input):
        kernel = get_momentum_kernel()
        if kernel is None:
            raise RuntimeError("CUDA kernel not available")
        
        # Prepare tensors
        deltaA = deltaA.float().contiguous()
        momentum_input = momentum_input.float().contiguous()
        
        # Output tensor
        output_h = torch.zeros_like(momentum_input)
        
        # Launch CUDA kernel
        kernel.launch_forward(deltaA, float(beta), momentum_input, output_h)
        
        # Save for backward pass
        ctx.save_for_backward(deltaA, momentum_input, output_h)
        ctx.beta = beta
        
        return output_h
    
    @staticmethod
    def backward(ctx, grad_output):
        deltaA, momentum_input, output_h = ctx.saved_tensors
        beta = ctx.beta
        kernel = get_momentum_kernel()
        
        # Initialize gradients
        grad_deltaA = torch.zeros_like(deltaA) if deltaA.requires_grad else torch.empty(0)
        grad_momentum_input = torch.zeros_like(momentum_input) if momentum_input.requires_grad else torch.empty(0)
        
        if grad_deltaA.numel() > 0 or grad_momentum_input.numel() > 0:
            # Launch CUDA backward kernel
            kernel.launch_backward(
                grad_output.contiguous(),
                deltaA,
                float(beta),
                output_h,
                grad_deltaA,
                grad_momentum_input
            )
        
        return grad_deltaA if grad_deltaA.numel() > 0 else None, None, grad_momentum_input if grad_momentum_input.numel() > 0 else None

momentum_scan_cuda = MomentumScanCUDA.apply  # Both use same optimized implementation 