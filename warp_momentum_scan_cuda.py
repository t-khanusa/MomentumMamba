#!/usr/bin/env python3
"""
Experimental Warp-Level Momentum Scan Implementation

This module provides experimental warp-level CUDA kernels for momentum scanning 
using advanced 2x2 matrix segmented scan with warp shuffle instructions.

This is EXPERIMENTAL code - use momentum_scan_cuda.py for production.

The warp-level approach implements:
- State: u_n = [[h_n], [v_n]]
- Matrix: M_n = [[A_n, β], [0, β]]  
- Input: F_n = [[αB_n*x_n], [αB_n*x_n]]
- Update: u_n = M_n * u_{n-1} + F_n

Uses 3-step warp-level algorithm:
1. Chunk processing: Each lane processes L/32 elements
2. Warp shuffle scan: Prefix scan across warp using shuffle instructions  
3. Final replay: Apply accumulated state to compute results
"""

import torch
from torch.utils.cpp_extension import load_inline

# Experimental warp-level CUDA kernel implementation
warp_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ====================== WARP-LEVEL SEGMENTED SCAN KERNEL ======================
// Advanced warp-level 2x2 matrix scan for maximum performance
// Each warp processes one sequence using segmented scan with shuffle instructions

__global__ void warp_mat2_scan_kernel(
    const float* __restrict__ A,       // (batch, L) - deltaA values
    const float* __restrict__ F,       // (batch, L) - momentum input values  
    const float  beta,                 // momentum decay factor
    float*       __restrict__ H_out,   // (batch, L) - output h_n values
    float*       __restrict__ V_out,   // (batch, L) - output v_n values
    int          L                     // sequence length
) {
    int row  = blockIdx.x;             // one sequence per warp
    int lane = threadIdx.x & 31;       // lane in [0..31]
    
    // pointers to this row
    const float* Arow = A     + row*L;
    const float* Frow = F     + row*L;
          float* Hrow = H_out + row*L;
          float* Vrow = V_out + row*L;

    // === ASSOCIATIVE PARALLEL SCAN APPROACH ===
    // Each element is a pair (M, F) where:
    // - M is 2x2 matrix [[A_t, β], [0, β]]
    // - F is 2x1 vector [F_t, F_t]
    // 
    // Associative operation: (M1, F1) ⊕ (M2, F2) = (M2*M1, M2*F1 + F2)
    // This allows proper parallel prefix scan!

    // Initialize carry state for multi-chunk processing
    float carry_h = 0.0f, carry_v = 0.0f;
    
    // Process sequence in chunks of 32
    for (int chunk_start = 0; chunk_start < L; chunk_start += 32) {
        int t = chunk_start + lane;
        
        // --- STEP 1: Build local (M_t, F_t) for this lane ---
        float At = (t < L) ? Arow[t] : 1.0f;  // Identity if out of bounds
        float Ft = (t < L) ? Frow[t] : 0.0f;  // Zero if out of bounds
        
        // Local matrix M_t = [[A_t, β], [0, β]]
        float m00 = At, m01 = beta, m10 = 0.0f, m11 = beta;
        
        // Local vector F_t = [F_t, F_t]  
        float f0 = Ft, f1 = Ft;
        
        // --- STEP 2: Associative parallel prefix scan within chunk ---
        // Use warp shuffle to implement log₂(32) = 5 steps of parallel scan
        
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            // Fetch (M_prev, F_prev) from previous lane
            float pm00 = __shfl_up_sync(0xFFFFFFFF, m00, offset);
            float pm01 = __shfl_up_sync(0xFFFFFFFF, m01, offset);
            float pm10 = __shfl_up_sync(0xFFFFFFFF, m10, offset);
            float pm11 = __shfl_up_sync(0xFFFFFFFF, m11, offset);
            float pf0  = __shfl_up_sync(0xFFFFFFFF, f0,  offset);
            float pf1  = __shfl_up_sync(0xFFFFFFFF, f1,  offset);
            
            if (lane >= offset) {
                // CORRECTED: Associative operation with proper timing convention
                // For left-to-right scan: apply M_prev first, then M_curr
                // Result: (M_curr * M_prev, M_curr * F_prev + F_curr)
                
                // New matrix: M_new = M_curr * M_prev (curr applied after prev)
                float nm00 = m00*pm00 + m01*pm10;  // A_curr*A_prev + β*0
                float nm01 = m00*pm01 + m01*pm11;  // A_curr*β + β*β  
                float nm10 = m10*pm00 + m11*pm10;  // 0*A_prev + β*0 = 0
                float nm11 = m10*pm01 + m11*pm11;  // 0*β + β*β = β²
                
                // New vector: F_new = M_curr * F_prev + F_curr
                float nf0 = m00*pf0 + m01*pf1 + f0;  // A_curr*h_prev + β*v_prev + h_curr
                float nf1 = m10*pf0 + m11*pf1 + f1;  // 0*h_prev + β*v_prev + v_curr
                
                // Update current values
                m00 = nm00; m01 = nm01; m10 = nm10; m11 = nm11;
                f0 = nf0; f1 = nf1;
            }
        }
        
        // --- STEP 3: Apply carry from previous chunks ---
        // Each lane applies the carry state to its local result
        float final_h = m00 * carry_h + m01 * carry_v + f0;
        float final_v = m10 * carry_h + m11 * carry_v + f1;
        
        // --- STEP 4: Store results for this chunk ---
        if (t < L) {
            Hrow[t] = final_h;  // h_t = first component
            Vrow[t] = final_v;  // v_t = second component  
        }
        
        // --- STEP 5: Update carry for next chunk ---
        // Lane 31 holds the accumulated state for the entire chunk
        carry_h = __shfl_sync(0xFFFFFFFF, final_h, 31);
        carry_v = __shfl_sync(0xFFFFFFFF, final_v, 31);
    }
}

// OPTIMIZED: Parallel warp-level backward pass using associative scan
// The backward pass can be formulated as an associative scan in reverse order
__global__ void warp_mat2_scan_backward_kernel(
    const float* __restrict__ grad_h_out,
    const float* __restrict__ grad_v_out,
    const float* __restrict__ A,
    const float* __restrict__ F,
    const float beta,
    const float* __restrict__ h_states,
    const float* __restrict__ v_states,
    float* __restrict__ grad_A,
    float* __restrict__ grad_F,
    int L
) {
    int row = blockIdx.x;
    int lane = threadIdx.x & 31;
    
    // pointers to this row
    const float* grad_h_row = grad_h_out + row * L;
    const float* grad_v_row = grad_v_out + row * L;
    const float* A_row = A + row * L;
    const float* h_row = h_states + row * L;
    float* grad_A_row = grad_A + row * L;
    float* grad_F_row = grad_F + row * L;
    
    // === PARALLEL BACKWARD SCAN USING ASSOCIATIVE OPERATIONS ===
    // Backward gradient equations can be written as associative scan:
    // State: [grad_h, grad_v]
    // Matrix: [[A, 0], [1, β]] (note: chain rule adds grad_h to grad_v)
    // Input: [grad_h_out, grad_v_out]
    
    // Initialize carry state for multi-chunk processing (REVERSE direction)
    float carry_grad_h = 0.0f, carry_grad_v = 0.0f;
    
    // Process sequence in REVERSE chunks of 32
    for (int chunk_end = L; chunk_end > 0; chunk_end -= 32) {
        int chunk_start = max(0, chunk_end - 32);
        
        // Map lane to reverse position: lane 0 gets highest index in chunk
        int t = chunk_end - 1 - lane;
        
        // --- STEP 1: Build local backward (M_t, F_t) for this lane ---
        bool valid = (t >= chunk_start && t >= 0);
        float At = valid ? A_row[t] : 1.0f;  // Identity for out-of-bounds
        float grad_h_out_t = valid ? grad_h_row[t] : 0.0f;
        float grad_v_out_t = valid ? grad_v_row[t] : 0.0f;
        
        // Local backward matrix M_t = [[A_t, 0], [1, β]]
        // This represents: grad_h_{t-1} = A_t * grad_h_t, grad_v_{t-1} = 1 * grad_h_t + β * grad_v_t
        float m00 = At, m01 = 0.0f, m10 = 1.0f, m11 = beta;
        
        // Local input vector F_t = [grad_h_out_t, grad_v_out_t]
        float f0 = grad_h_out_t, f1 = grad_v_out_t;
        
        // --- STEP 2: Associative parallel prefix scan within chunk (REVERSE ORDER) ---
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            // Fetch from "previous" lane in reverse direction (higher index)
            float pm00 = __shfl_down_sync(0xFFFFFFFF, m00, offset);
            float pm01 = __shfl_down_sync(0xFFFFFFFF, m01, offset);
            float pm10 = __shfl_down_sync(0xFFFFFFFF, m10, offset);
            float pm11 = __shfl_down_sync(0xFFFFFFFF, m11, offset);
            float pf0  = __shfl_down_sync(0xFFFFFFFF, f0,  offset);
            float pf1  = __shfl_down_sync(0xFFFFFFFF, f1,  offset);
            
            if (lane + offset < 32) {
                // Associative operation in reverse: M_curr * M_next + F_curr
                float nm00 = m00*pm00 + m01*pm10;  // A_curr*A_next + 0*1
                float nm01 = m00*pm01 + m01*pm11;  // A_curr*0 + 0*β = 0
                float nm10 = m10*pm00 + m11*pm10;  // 1*A_next + β*1 
                float nm11 = m10*pm01 + m11*pm11;  // 1*0 + β*β = β²
                
                float nf0 = m00*pf0 + m01*pf1 + f0;  // A_curr*grad_h_next + 0*grad_v_next + grad_h_curr
                float nf1 = m10*pf0 + m11*pf1 + f1;  // 1*grad_h_next + β*grad_v_next + grad_v_curr
                
                m00 = nm00; m01 = nm01; m10 = nm10; m11 = nm11;
                f0 = nf0; f1 = nf1;
            }
        }
        
        // --- STEP 3: Apply carry from subsequent chunks ---
        float final_grad_h = m00 * carry_grad_h + m01 * carry_grad_v + f0;
        float final_grad_v = m10 * carry_grad_h + m11 * carry_grad_v + f1;
        
        // --- STEP 4: Compute and store gradients ---
        if (valid) {
            // Gradient w.r.t. input F
            grad_F_row[t] = final_grad_v;
            
            // Gradient w.r.t. A (need h from previous timestep)
            if (t > 0) {
                grad_A_row[t] = final_grad_h * h_row[t-1];
            } else {
                grad_A_row[t] = 0.0f;  // No previous state
            }
        }
        
        // --- STEP 5: Update carry for next chunk (in reverse direction) ---
        // Lane 0 (which processes the earliest timestep in this chunk) has the carry
        carry_grad_h = __shfl_sync(0xFFFFFFFF, final_grad_h, 0);
        carry_grad_v = __shfl_sync(0xFFFFFFFF, final_grad_v, 0);
    }
}

// Kernel launchers
void launch_warp_forward_cuda(
    torch::Tensor A,
    torch::Tensor F,
    float beta,
    torch::Tensor H_out,
    torch::Tensor V_out
) {
    const int batch_size = A.size(0);
    const int seq_len = A.size(1);
    
    // Launch with one block per sequence, 32 threads per block (one warp)
    warp_mat2_scan_kernel<<<batch_size, 32>>>(
        A.data_ptr<float>(),
        F.data_ptr<float>(),
        beta,
        H_out.data_ptr<float>(),
        V_out.data_ptr<float>(),
        seq_len
    );
    
    cudaDeviceSynchronize();
}

void launch_warp_backward_cuda(
    torch::Tensor grad_h_out,
    torch::Tensor grad_v_out,
    torch::Tensor A,
    torch::Tensor F,
    float beta,
    torch::Tensor h_states,
    torch::Tensor v_states,
    torch::Tensor grad_A,
    torch::Tensor grad_F
) {
    const int batch_size = A.size(0);
    const int seq_len = A.size(1);
    
    warp_mat2_scan_backward_kernel<<<batch_size, 32>>>(
        grad_h_out.data_ptr<float>(),
        grad_v_out.data_ptr<float>(),
        A.data_ptr<float>(),
        F.data_ptr<float>(),
        beta,
        h_states.data_ptr<float>(),
        v_states.data_ptr<float>(),
        grad_A.data_ptr<float>(),
        grad_F.data_ptr<float>(),
        seq_len
    );
    
    cudaDeviceSynchronize();
}
"""

warp_cpp_source = r"""
#include <torch/extension.h>

// Warp-level kernel declarations
void launch_warp_forward_cuda(
    torch::Tensor A,
    torch::Tensor F,
    float beta,
    torch::Tensor H_out,
    torch::Tensor V_out
);

void launch_warp_backward_cuda(
    torch::Tensor grad_h_out,
    torch::Tensor grad_v_out,
    torch::Tensor A,
    torch::Tensor F,
    float beta,
    torch::Tensor h_states,
    torch::Tensor v_states,
    torch::Tensor grad_A,
    torch::Tensor grad_F
);

// Warp-level wrappers
void launch_warp_forward(
    torch::Tensor A,
    torch::Tensor F,
    float beta,
    torch::Tensor H_out,
    torch::Tensor V_out
) {
    launch_warp_forward_cuda(A, F, beta, H_out, V_out);
}

void launch_warp_backward(
    torch::Tensor grad_h_out,
    torch::Tensor grad_v_out,
    torch::Tensor A,
    torch::Tensor F,
    float beta,
    torch::Tensor h_states,
    torch::Tensor v_states,
    torch::Tensor grad_A,
    torch::Tensor grad_F
) {
    launch_warp_backward_cuda(grad_h_out, grad_v_out, A, F, beta, h_states, v_states, grad_A, grad_F);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_warp_forward", &launch_warp_forward, "Warp-level momentum scan forward pass");
    m.def("launch_warp_backward", &launch_warp_backward, "Warp-level momentum scan backward pass");
}
"""

def load_warp_momentum_cuda():
    """Load experimental warp-level momentum CUDA kernel"""
    try:
        return load_inline(
            name="warp_momentum_scan_cuda_kernel", 
            cpp_sources=[warp_cpp_source],
            cuda_sources=[warp_cuda_source],
            verbose=False,
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math']
        )
    except Exception as e:
        print(f"Failed to load warp momentum CUDA kernel: {e}")
        return None

_warp_momentum_kernel = None

def get_warp_momentum_kernel():
    global _warp_momentum_kernel
    if _warp_momentum_kernel is None:
        _warp_momentum_kernel = load_warp_momentum_cuda()
    return _warp_momentum_kernel

class WarpMomentumScanCUDA(torch.autograd.Function):
    """
    EXPERIMENTAL: Advanced warp-level momentum scan with automatic differentiation
    
    This implements the 2x2 matrix segmented scan approach:
    - State: u_n = [[h_n], [v_n]]  
    - Matrix: M_n = [[A_n, β], [0, β]]
    - Input: F_n = [[αB_n*x_n], [αB_n*x_n]]
    - Update: u_n = M_n * u_{n-1} + F_n
    
    Uses warp-level parallelism for maximum performance.
    
    WARNING: This is experimental code with known correctness issues.
    Use momentum_scan_cuda.py for production workloads.
    """
    @staticmethod
    def forward(ctx, deltaA, beta, momentum_input):
        kernel = get_warp_momentum_kernel()
        if kernel is None:
            raise RuntimeError("Warp CUDA kernel not available")
        
        # Prepare tensors
        deltaA = deltaA.float().contiguous()
        momentum_input = momentum_input.float().contiguous()
        
        # Output tensors for both h and v states
        h_states = torch.zeros_like(momentum_input)
        v_states = torch.zeros_like(momentum_input)
        
        # Launch warp-level CUDA kernel
        kernel.launch_warp_forward(deltaA, momentum_input, float(beta), h_states, v_states)
        
        # Save for backward pass
        ctx.save_for_backward(deltaA, momentum_input, h_states, v_states)
        ctx.beta = beta
        
        return h_states
    
    @staticmethod
    def backward(ctx, grad_output):
        deltaA, momentum_input, h_states, v_states = ctx.saved_tensors
        beta = ctx.beta
        kernel = get_warp_momentum_kernel()
        
        # Initialize gradients
        grad_deltaA = torch.zeros_like(deltaA) if deltaA.requires_grad else torch.empty(0)
        grad_momentum_input = torch.zeros_like(momentum_input) if momentum_input.requires_grad else torch.empty(0)
        
        # Use a small dummy tensor instead of allocating full size for grad_v_output
        # We only care about h_states gradients, not v_states gradients in this implementation
        grad_v_output = torch.zeros_like(grad_output[:1])  # Minimal allocation
        
        if grad_deltaA.numel() > 0 or grad_momentum_input.numel() > 0:
            # Launch CUDA backward kernel
            kernel.launch_warp_backward(
                grad_output.contiguous(),
                grad_v_output.expand_as(grad_output).contiguous(),  # Expand to match expected size
                deltaA,
                momentum_input,
                float(beta),
                h_states,
                v_states,
                grad_deltaA,
                grad_momentum_input
            )
        
        return grad_deltaA if grad_deltaA.numel() > 0 else None, None, grad_momentum_input if grad_momentum_input.numel() > 0 else None

# Export the experimental implementation
warp_momentum_scan_cuda = WarpMomentumScanCUDA.apply 