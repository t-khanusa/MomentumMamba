import torch
from torch import nn
from calflops import calculate_flops

from generate_LinOSS_pytorch import create_pytorch_model
from LinOSS import LinOSS
import jax
import jax.numpy as jnp
from mamba_ssm import Mamba

model = create_pytorch_model(
    model_name='LinOSS',
    data_dim=6,
    label_dim=32,
    hidden_dim=32,
    num_blocks=2,
    ssm_dim=32,
    classification=True,
    output_step=1,
    linoss_discretization='IM',
    norm_type='batch',
    drop_rate=0.05,
    device='cuda',
    seed=42
)


flops, macs, params = calculate_flops(model=model, input_shape=(1, 256, 6), output_as_string=True, output_precision=4)



# JAX model
ssm = LinOSS(
    num_blocks=2,
    N=6,
    ssm_size=32,
    H=32,
    output_dim=32,
    classification=True,
    output_step=1,
    discretization='IM',
    key=jax.random.PRNGKey(42),
)

# Method 1: Using JAX's cost analysis API
def calculate_jax_flops(model, input_shape):
    """Calculate FLOPs for JAX model using JAX's built-in cost analysis."""
    # Create dummy input
    dummy_input = jnp.ones(input_shape, dtype=jnp.float32)
    
    # JIT compile the model
    @jax.jit
    def model_fn(x):
        return model(x)
    
    # Lower and compile to get cost analysis
    compiled = model_fn.lower(dummy_input).compile()
    cost_analysis = compiled.cost_analysis()[0]
    
    return cost_analysis

# Calculate FLOPs for JAX model
try:
    cost_analysis = calculate_jax_flops(ssm, (256, 32))
    flops_jax = cost_analysis.get('flops', 0)
    print(f"\nJAX Model FLOPs: {flops_jax}")
    print(f"JAX Model FLOPs (formatted): {flops_jax/1e6:.4f} MFLOPS")
    print("Full cost analysis:", cost_analysis)
except Exception as e:
    print(f"Error calculating JAX FLOPs: {e}")
    flops_jax = None

# Method 2: Manual FLOP calculation based on model structure
def count_jax_model_params(model):
    """Count parameters in JAX model."""
    def count_params(pytree):
        return sum(x.size for x in jax.tree_util.tree_leaves(pytree) if hasattr(x, 'size'))
    return count_params(model)

def estimate_jax_flops_manual(model, input_shape, num_blocks=2, ssm_size=64, H=64):
    """Manually estimate FLOPs for JAX LinOSS model."""
    batch_size, seq_len, input_dim = input_shape
    
    # Linear encoder: input_dim -> H
    encoder_flops = 2 * batch_size * seq_len * input_dim * H
    
    # LinOSS blocks
    linoss_flops = 0
    for _ in range(num_blocks):
        # SSM operations (approximation)
        ssm_flops = 2 * batch_size * seq_len * ssm_size * H
        # Skip connections and other operations
        other_flops = batch_size * seq_len * H
        linoss_flops += ssm_flops + other_flops
    
    # Output projection: H -> output_dim
    output_flops = 2 * batch_size * seq_len * H * 32  # output_dim=32
    
    total_flops = encoder_flops + linoss_flops + output_flops
    return total_flops

# Manual FLOP estimation
manual_flops = estimate_jax_flops_manual(ssm, (1, 256, 6))

# Parameter count for JAX model
try:
    param_count = count_jax_model_params(ssm)
except Exception as e:
    print(f"Error counting JAX parameters: {e}")
    param_count = None

# Format JAX output similar to PyTorch output
print("\n" + "="*80)
print("JAX LinOSS Model Analysis:")
print("="*80)

if param_count is not None:
    print(f"Total Training Params:                                                  {param_count/1e3:.2f} K")

if flops_jax is not None:
    macs_jax = flops_jax / 2  # Approximate MACs as FLOPs/2
    print(f"fwd MACs:                                                               {macs_jax/1e6:.4f} MMACs")
    print(f"fwd FLOPs:                                                              {flops_jax/1e6:.4f} MFLOPS")
    print(f"fwd+bwd MACs:                                                           {(macs_jax*3)/1e6:.4f} MMACs")
    print(f"fwd+bwd FLOPs:                                                          {(flops_jax*3)/1e6:.4f} MFLOPS")
else:
    # Use manual estimation if JAX cost analysis failed
    macs_manual = manual_flops / 2
    print(f"fwd MACs (manual est.):                                                 {macs_manual/1e6:.4f} MMACs")
    print(f"fwd FLOPs (manual est.):                                               {manual_flops/1e6:.4f} MFLOPS")
    print(f"fwd+bwd MACs (manual est.):                                            {(macs_manual*3)/1e6:.4f} MMACs")
    print(f"fwd+bwd FLOPs (manual est.):                                           {(manual_flops*3)/1e6:.4f} MFLOPS")


mamba = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=32, # Model dimension d_model
    d_state=32,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")


# print("Flops: ", flops)
# print("Macs: ", macs)
# print("Params: ", params)


flops3, macs3, params3 = calculate_flops(model=mamba, input_shape=(1, 256, 32), output_as_string=True, output_precision=4)

print("Flops: ", flops3)
print("Macs: ", macs3)
print("Params: ", params3)