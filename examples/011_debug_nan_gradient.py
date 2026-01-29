# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Example: Debugging NaN in Gradient Computations
================================================

This example demonstrates how to use the `debug_nan=True` parameter in
`brainstate.transform.grad` to identify where NaN values are introduced
during gradient computation.

When `debug_nan=True`, the gradient computation is evaluated equation-by-equation,
and if NaN is detected, a RuntimeError is raised with detailed information about
which primitive operation first introduced the NaN values.
"""

import jax.numpy as jnp

import brainstate


# =============================================================================
# Example 1: NaN from log(0) in forward pass
# =============================================================================

def example_log_of_zero():
    """
    Demonstrates NaN detection when computing log(0).

    The log function produces -inf for 0, and subsequent operations
    can produce NaN.
    """
    print("=" * 60)
    print("Example 1: NaN from log(0)")
    print("=" * 60)

    weight = brainstate.State(jnp.array([0.0, 1.0, 2.0]))  # Contains zero!

    def loss_fn(x):
        # log(0) = -inf, and -inf * x can produce NaN
        return jnp.sum(jnp.log(weight.value) * x)

    # Create gradient function with debug_nan=True
    grad_fn = brainstate.transform.grad(
        loss_fn,
        grad_states=[weight],
        debug_nan=True  # Enable NaN debugging
    )

    @brainstate.transform.jit
    def fn(x):
        return grad_fn(x)

    x = jnp.array([1.0, 1.0, 1.0])
    grads = fn(x)
    print(f"Gradients: {grads}")
    print()


# =============================================================================
# Example 2: NaN from division by zero in backward pass
# =============================================================================

def example_division_by_zero():
    """
    Demonstrates NaN detection when division by zero occurs.

    The gradient of 1/x at x=0 is undefined and produces NaN.
    """
    print("=" * 60)
    print("Example 2: NaN from division by zero")
    print("=" * 60)

    weight = brainstate.State(jnp.array([0.0, 1.0, 2.0]))  # Contains zero!

    def loss_fn(x):
        # 1/weight where weight contains 0 will produce inf
        # The gradient computation will produce NaN
        return jnp.sum(1.0 / weight.value * x)

    grad_fn = brainstate.transform.grad(
        loss_fn,
        grad_states=[weight],
        debug_nan=True
    )

    x = jnp.array([1.0, 1.0, 1.0])
    grads = grad_fn(x)
    print(f"Gradients: {grads}")

    print()


# =============================================================================
# Example 3: NaN from sqrt of negative number
# =============================================================================

def example_sqrt_negative():
    """
    Demonstrates NaN detection when computing sqrt of negative number.
    """
    print("=" * 60)
    print("Example 3: NaN from sqrt of negative number")
    print("=" * 60)

    weight = brainstate.State(jnp.array([-1.0, 1.0, 4.0]))  # Contains negative!

    def loss_fn(x):
        # sqrt of negative number produces NaN
        return jnp.sum(jnp.sqrt(weight.value) * x)

    grad_fn = brainstate.transform.grad(
        loss_fn,
        grad_states=[weight],
        debug_nan=True
    )

    x = jnp.array([1.0, 1.0, 1.0])
    grads = grad_fn(x)
    print(f"Gradients: {grads}")

    print()


# =============================================================================
# Example 4: Comparing with debug_nan=False (default)
# =============================================================================

def example_comparison():
    """
    Demonstrates the difference between debug_nan=True and debug_nan=False.

    With debug_nan=False (default), NaN values are silently propagated.
    With debug_nan=True, a RuntimeError is raised with detailed diagnostics.
    """
    print("=" * 60)
    print("Example 4: Comparison - debug_nan=True vs False")
    print("=" * 60)

    weight = brainstate.State(jnp.array([0.0, 1.0, 2.0]))

    def loss_fn(x):
        return jnp.sum(jnp.log(weight.value) * x)

    # Without debug_nan (default behavior)
    grad_fn_no_debug = brainstate.transform.grad(
        loss_fn,
        grad_states=[weight],
        debug_nan=False  # Default
    )

    print("With debug_nan=False:")
    x = jnp.array([1.0, 1.0, 1.0])
    grads = grad_fn_no_debug(x)
    print(f"  Gradients (may contain NaN): {grads}")
    print(f"  Has NaN: {jnp.any(jnp.isnan(grads))}")

    # Reset weight value
    weight.value = jnp.array([0.0, 1.0, 2.0])

    # With debug_nan
    grad_fn_with_debug = brainstate.transform.grad(
        loss_fn,
        grad_states=[weight],
        debug_nan=True
    )

    print("\nWith debug_nan=True:")
    grads = grad_fn_with_debug(x)
    print(f"  Gradients: {grads}")

    print()


# =============================================================================
# Example 5: Neural network training with NaN detection
# =============================================================================

def example_neural_network():
    """
    Demonstrates using debug_nan in a neural network training scenario.

    This is useful for debugging exploding gradients or numerical instability.
    """
    print("=" * 60)
    print("Example 5: Neural network with potential NaN")
    print("=" * 60)

    # Create a simple neural network with problematic initialization
    W1 = brainstate.State(jnp.array([[1e10, 0.0], [0.0, 1e10]]))  # Very large weights
    W2 = brainstate.State(jnp.array([[1.0], [1.0]]))

    def forward(x):
        h = jnp.tanh(x @ W1.value)  # Can overflow with large weights
        return jnp.sum(h @ W2.value)

    def loss_fn(x, target):
        pred = forward(x)
        return (pred - target) ** 2

    # Create gradient function with debug_nan
    grad_fn = brainstate.transform.grad(
        loss_fn,
        grad_states=[W1, W2],
        debug_nan=True
    )

    x = jnp.array([[1.0, 1.0]])
    target = 0.5
    grads = grad_fn(x, target)
    print(f"W1 gradients shape: {grads[0].shape}")
    print(f"W2 gradients shape: {grads[1].shape}")

    print()


# =============================================================================
# Example 6: Using as a decorator
# =============================================================================

def example_decorator():
    """
    Demonstrates using grad with debug_nan as a decorator.
    """
    print("=" * 60)
    print("Example 6: Using as a decorator")
    print("=" * 60)

    weight = brainstate.State(jnp.array([1.0, 2.0, 3.0]))  # Safe values

    @brainstate.transform.grad(grad_states=[weight], debug_nan=True)
    def loss_fn(x):
        return jnp.sum(weight.value ** 2 * x)

    x = jnp.array([1.0, 1.0, 1.0])
    grads = loss_fn(x)
    print(f"Gradients (no NaN): {grads}")

    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    example_log_of_zero()
    # example_division_by_zero()
    # example_sqrt_negative()
    # example_comparison()
    # example_neural_network()
    # example_decorator()
    #
    # print("=" * 60)
    # print("Examples completed!")
    # print("=" * 60)
