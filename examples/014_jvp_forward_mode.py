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
``brainstate.transform.jvp`` — state-aware forward-mode autodiff.

``jvp`` traces a function (which may read and write :class:`brainstate.State`
objects), applies :func:`jax.jvp` with respect to its positional arguments, and
returns ``(primal_out, tangent_out)``. ``tangent_out`` is the directional
derivative ``J @ v`` of the function along the tangent ``v``: one forward pass
gives one Jacobian *column* (combination), making forward mode the cheap choice
when the number of inputs is small (tall Jacobians).

States are treated as constants for the forward pass (zero tangent), but states
that the function *writes* are still updated afterwards.

This script demonstrates seven use cases:

1. Basic directional derivative of a scalar function.
2. A JVP with states held constant plus a state write-back.
3. ``has_aux=True``.
4. A Jacobian built column-by-column (forward mode); cross-checked vs jacfwd.
5. Tall vs wide Jacobian intuition (forward mode cheap for few inputs).
6. Tangent propagation through a small ``brainstate.nn`` network.
7. A forward-over-reverse Hessian-vector product, cross-checked against the
   reverse-mode HVP.
"""

import jax
import jax.numpy as jnp

import brainstate


# -----------------------------------------------------------------------------
# 1. Basic directional derivative
# -----------------------------------------------------------------------------
# ``primals`` and ``tangents`` are tuples matching the function signature. The
# tangent output is the derivative of ``f`` in the direction of ``tangents``.
def f(x):
    return jnp.sum(x ** 2)


x = jnp.array([1.0, 2.0, 3.0])
v = jnp.array([1.0, 0.0, 0.0])
primal, tangent = brainstate.transform.jvp(f, (x,), (v,))
print("1. Basic directional derivative")
print(f"   f(x) = {primal}")
print(f"   df along v = {tangent}  (expected grad . v = {float(jnp.dot(2.0 * x, v))})")
assert jnp.allclose(tangent, jnp.dot(2.0 * x, v))
print()


# -----------------------------------------------------------------------------
# 2. States held constant, with write-back
# -----------------------------------------------------------------------------
# State values carry zero tangent (they are constants of the forward pass), so
# the tangent output depends only on the positional tangent. A state the function
# writes still receives its new value afterwards.
gain = brainstate.ParamState(jnp.array(4.0))
calls = brainstate.State(jnp.array(0.0))


def scaled(x):
    calls.value = calls.value + 1.0  # written: updated after the trace
    return gain.value * x            # gain: read-only constant


x = jnp.array([1.0, 2.0])
v = jnp.array([1.0, 1.0])
primal, tangent = brainstate.transform.jvp(scaled, (x,), (v,))
print("2. States held constant + write-back")
print(f"   primal = {primal}, tangent = {tangent}  (expected gain * v = {gain.value * v})")
print(f"   calls after trace = {calls.value}")
assert jnp.allclose(tangent, gain.value * v)
assert jnp.allclose(calls.value, 1.0)
print()


# -----------------------------------------------------------------------------
# 3. Auxiliary data
# -----------------------------------------------------------------------------
# ``has_aux=True`` returns a third value that is not differentiated.
def f_aux(x):
    return jnp.sum(x ** 3), {'max': jnp.max(x)}


x = jnp.array([1.0, 2.0, 3.0])
v = jnp.array([1.0, 1.0, 1.0])
primal, tangent, aux = brainstate.transform.jvp(f_aux, (x,), (v,), has_aux=True)
print("3. Auxiliary data")
print(f"   primal = {primal}, tangent = {tangent}, aux['max'] = {aux['max']}")
assert jnp.allclose(tangent, jnp.dot(3.0 * x ** 2, v))
print()


# -----------------------------------------------------------------------------
# 4. Jacobian column-by-column (forward mode)
# -----------------------------------------------------------------------------
# Each JVP with a basis tangent ``e_i`` returns column ``i`` of the Jacobian.
# Mapping over the basis with ``jax.vmap`` assembles the whole matrix; this is
# what ``jax.jacfwd`` does under the hood.
def vector_fn(x):
    return jnp.array([jnp.sum(x), jnp.sum(x ** 2), jnp.prod(x)])


x = jnp.array([1.0, 2.0, 3.0])
basis = jnp.eye(x.shape[0])
columns = jax.vmap(lambda e: brainstate.transform.jvp(vector_fn, (x,), (e,))[1])(basis)
jac = columns.T  # vmap stacks columns along axis 0; transpose to (outputs, inputs)
print("4. Jacobian column-by-column (forward mode)")
print(f"   Jacobian =\n{jac}")
assert jnp.allclose(jac, jax.jacfwd(vector_fn)(x))  # cross-check
print("   matches jax.jacfwd\n")


# -----------------------------------------------------------------------------
# 5. Tall vs wide Jacobian intuition
# -----------------------------------------------------------------------------
# Forward mode costs one pass per *input*; reverse mode one pass per *output*.
# For a tall Jacobian (few inputs, many outputs), forward mode wins: here 2
# inputs map to 50 outputs, so 2 JVPs reconstruct the full Jacobian.
def expand(x):  # R^2 -> R^50
    freqs = jnp.linspace(1.0, 5.0, 50)
    return jnp.sin(freqs * x[0]) + jnp.cos(freqs * x[1])


x = jnp.array([0.3, 0.7])
cols = jax.vmap(lambda e: brainstate.transform.jvp(expand, (x,), (e,))[1])(jnp.eye(2))
jac = cols.T  # (50, 2)
print("5. Tall Jacobian (R^2 -> R^50)")
print(f"   Jacobian shape = {tuple(jac.shape)} from just {x.shape[0]} forward passes")
assert jnp.allclose(jac, jax.jacfwd(expand)(x), atol=1e-5)
print("   matches jax.jacfwd\n")


# -----------------------------------------------------------------------------
# 6. Tangent propagation through a network
# -----------------------------------------------------------------------------
# Push a tangent through a small network's input. The parameters are constant
# (zero tangent), so the output tangent is the network's Jacobian-vector product
# w.r.t. the input -- useful for sensitivity analysis.
net = brainstate.nn.Sequential(
    brainstate.nn.Linear(3, 8),
    brainstate.nn.ReLU(),
    brainstate.nn.Linear(8, 2),
)
x = brainstate.random.randn(3)
v = jnp.array([1.0, 0.0, 0.0])  # perturb only the first input feature


def net_fn(inp):
    return net(inp)


primal, tangent = brainstate.transform.jvp(net_fn, (x,), (v,))
print("6. Tangent propagation through a network")
print(f"   output = {primal}")
print(f"   d(output)/d(input_0) = {tangent}")
assert jnp.allclose(tangent, jax.jacfwd(net_fn)(x) @ v, atol=1e-5)
print("   matches jacfwd(net)(x) @ v\n")


# -----------------------------------------------------------------------------
# 7. Forward-over-reverse Hessian-vector product
# -----------------------------------------------------------------------------
# Composing forward-mode over a reverse-mode gradient gives Hessian-vector
# products: jvp(grad(f))(x; v) == H(x) @ v. This is typically the cheapest HVP.
def rosen(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


x = jnp.array([0.5, 1.5, 2.0])
v = jnp.array([1.0, 0.0, 0.0])
grad_fn = jax.grad(rosen)
_, hvp = brainstate.transform.jvp(grad_fn, (x,), (v,))
print("7. Forward-over-reverse Hessian-vector product")
print(f"   H @ v = {hvp}")
assert jnp.allclose(hvp, jax.hessian(rosen)(x) @ v)
print("   matches jax.hessian(rosen)(x) @ v\n")

print("All jvp examples passed.")
