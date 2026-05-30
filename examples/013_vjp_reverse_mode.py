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
``brainstate.transform.vjp`` — state-aware reverse-mode autodiff.

``vjp`` traces a function (which may read and write :class:`brainstate.State`
objects), applies :func:`jax.vjp`, and returns the primal output together with a
*pullback* ``vjp_fn``. Calling ``vjp_fn(v)`` computes ``v @ J`` where ``J`` is
the Jacobian of the function at the primals. One forward trace amortises any
number of backward passes, which makes ``vjp`` the natural primitive for full
Jacobians, custom cotangents, and higher-order products.

What ``vjp_fn`` returns depends on ``grad_states`` and ``argnums``:

    grad_states   argnums                 vjp_fn(v) returns
    -----------   ---------------------   --------------------------
    None          int / sequence          arg_cotangents
    provided      int / sequence          (state_cts, arg_cts)
    provided      None (or no primals)     state_cotangents

This script demonstrates eight use cases:

1. Plain reverse-mode autodiff (no states); matches ``jax.vjp``.
2. Gradients with respect to states -> ``(state_ct, arg_ct)``.
3. State-only gradients (the canonical neural-network case).
4. ``has_aux=True`` together with a state write-back.
5. A full Jacobian by reusing the pullback over rows of the identity.
6. Multiple arguments via a sequence ``argnums``.
7. A Hessian-vector product (reverse-over-reverse).
8. A small linear-regression training loop driven by ``vjp``.
"""

import jax
import jax.numpy as jnp

import brainstate


# -----------------------------------------------------------------------------
# 1. Plain reverse-mode autodiff (no states)
# -----------------------------------------------------------------------------
# With a scalar ``int`` argnums (default 0) the argument cotangent is returned
# unwrapped. Evaluating ``vjp_fn(1.0)`` on a scalar output reproduces grad(f).
def f(x):
    return jnp.sum(x ** 2)


x = jnp.array([1.0, 2.0, 3.0])
out, vjp_fn = brainstate.transform.vjp(f, x)
grad_x = vjp_fn(1.0)  # d/dx sum(x**2) = 2x
print("1. Plain reverse-mode autodiff")
print(f"   f(x) = {out}")
print(f"   grad = {grad_x}")
assert jnp.allclose(grad_x, 2.0 * x)
assert jnp.allclose(grad_x, jax.grad(f)(x))  # cross-check against jax.grad
print("   matches jax.grad\n")


# -----------------------------------------------------------------------------
# 2. Gradients with respect to states
# -----------------------------------------------------------------------------
# Passing ``grad_states`` also produces cotangents for those states; the pullback
# then returns ``(state_ct, arg_ct)``.
w = brainstate.ParamState(jnp.array([2.0, 3.0]))


def weighted_sum(inp):
    return jnp.sum(w.value * inp)


inp = jnp.array([5.0, 7.0])
out, vjp_fn = brainstate.transform.vjp(weighted_sum, inp, grad_states=w)
state_ct, arg_ct = vjp_fn(1.0)
print("2. Gradients with respect to states")
print(f"   d/dw sum(w*inp) = {state_ct}  (expected inp = {inp})")
print(f"   d/dinp sum(w*inp) = {arg_ct}  (expected w = {w.value})")
assert jnp.allclose(state_ct, inp)
assert jnp.allclose(arg_ct, w.value)
print()


# -----------------------------------------------------------------------------
# 3. State-only gradients (canonical neural-network case)
# -----------------------------------------------------------------------------
# The loss closes over its trainable parameters and takes no differentiable
# positional argument, so the pullback returns just the state cotangents (a list
# matching ``grad_states``). This is exactly what an optimizer consumes.
weight = brainstate.ParamState(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
bias = brainstate.ParamState(jnp.array([0.0, 0.0]))
data = jnp.array([1.0, 1.0])


def predict_loss():
    y = data @ weight.value + bias.value
    return jnp.sum(y ** 2)


out, vjp_fn = brainstate.transform.vjp(predict_loss, grad_states=[weight, bias])
grad_w, grad_b = vjp_fn(1.0)
print("3. State-only gradients")
print(f"   loss = {out}")
print(f"   grad_w shape {tuple(grad_w.shape)}, grad_b shape {tuple(grad_b.shape)}")
# Cross-check against brainstate.transform.grad, which targets the same states.
ref_w, ref_b = brainstate.transform.grad(predict_loss, grad_states=[weight, bias])()
assert jnp.allclose(grad_w, ref_w) and jnp.allclose(grad_b, ref_b)
print("   matches brainstate.transform.grad\n")


# -----------------------------------------------------------------------------
# 4. Auxiliary data and state write-back
# -----------------------------------------------------------------------------
# ``has_aux=True`` returns a side output untouched by differentiation. Any state
# the function writes keeps its new value after the trace (the same side effect
# the function would have produced if called directly).
counter = brainstate.State(jnp.array(0.0))


def loss_with_aux(x):
    counter.value = counter.value + 1.0  # a step counter, written each call
    return jnp.sum(x ** 2), {'mean': jnp.mean(x)}


x = jnp.array([1.0, 2.0])
out, vjp_fn, aux = brainstate.transform.vjp(loss_with_aux, x, has_aux=True)
grad_x = vjp_fn(1.0)
print("4. Auxiliary data and state write-back")
print(f"   aux['mean'] = {aux['mean']}")
print(f"   grad = {grad_x}")
print(f"   counter after trace = {counter.value}")
assert jnp.allclose(grad_x, 2.0 * x)
assert jnp.allclose(counter.value, 1.0)  # write threaded back into the State
print()


# -----------------------------------------------------------------------------
# 5. Full Jacobian by reusing the pullback
# -----------------------------------------------------------------------------
# A single forward trace produces a reusable pullback. Mapping it over the rows
# of the identity yields one gradient per output row -- i.e. the full Jacobian.
def vector_fn(x):
    return jnp.array([jnp.sum(x), jnp.sum(x ** 2)])


x = jnp.array([1.0, 2.0, 3.0])
out, vjp_fn = brainstate.transform.vjp(vector_fn, x)
jac = jax.vmap(vjp_fn)(jnp.eye(2))  # each row = gradient of one output
print("5. Full Jacobian from one trace")
print(f"   Jacobian =\n{jac}")
assert jnp.allclose(jac, jax.jacrev(vector_fn)(x))  # cross-check against jacrev
print("   matches jax.jacrev\n")


# -----------------------------------------------------------------------------
# 6. Multiple arguments via a sequence argnums
# -----------------------------------------------------------------------------
# A sequence ``argnums`` returns a tuple of cotangents, one per requested index.
def bilinear(a, b):
    return jnp.sum(a * b)


a = jnp.array([1.0, 2.0])
b = jnp.array([3.0, 4.0])
out, vjp_fn = brainstate.transform.vjp(bilinear, a, b, argnums=(0, 1))
ga, gb = vjp_fn(1.0)
print("6. Multiple arguments")
print(f"   d/da = {ga} (expected b), d/db = {gb} (expected a)")
assert jnp.allclose(ga, b) and jnp.allclose(gb, a)
print()


# -----------------------------------------------------------------------------
# 7. Hessian-vector product (reverse-over-reverse)
# -----------------------------------------------------------------------------
# A pullback differentiates other gradients too. Here we VJP a gradient function
# to form Hessian-vector products without ever materialising the Hessian.
def rosen(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


x = jnp.array([0.5, 1.5, 2.0])
grad_fn = jax.grad(rosen)
_, hvp_fn = brainstate.transform.vjp(grad_fn, x)
v = jnp.array([1.0, 0.0, 0.0])
hvp = hvp_fn(v)  # (H @ v) because H is symmetric
print("7. Hessian-vector product")
print(f"   H @ v = {hvp}")
assert jnp.allclose(hvp, jax.hessian(rosen)(x) @ v)  # cross-check
print("   matches jax.hessian(rosen)(x) @ v\n")


# -----------------------------------------------------------------------------
# 8. A tiny linear-regression training loop driven by vjp
# -----------------------------------------------------------------------------
# Fit ``y = 2 x + 1`` with plain SGD. The loss closes over the parameters, so the
# state-only pullback (vjp_fn(1.0)) directly yields the parameter gradients.
xs = brainstate.random.uniform(-1.0, 1.0, (128, 1))
ys = 2.0 * xs + 1.0

slope = brainstate.ParamState(jnp.zeros((1, 1)))
intercept = brainstate.ParamState(jnp.zeros((1,)))


def mse():
    pred = xs @ slope.value + intercept.value
    return jnp.mean((pred - ys) ** 2)


lr = 0.5
print("8. Linear-regression training loop")
for step in range(201):
    loss, back = brainstate.transform.vjp(mse, grad_states=[slope, intercept])
    g_slope, g_intercept = back(1.0)
    slope.value = slope.value - lr * g_slope
    intercept.value = intercept.value - lr * g_intercept
    if step % 50 == 0:
        print(f"   step {step:3d}: loss={float(loss):.5f}, "
              f"slope={float(slope.value.reshape(())):.3f}, "
              f"intercept={float(intercept.value.reshape(())):.3f}")
assert jnp.allclose(slope.value, 2.0, atol=1e-2)
assert jnp.allclose(intercept.value, 1.0, atol=1e-2)
print("   converged to slope~2, intercept~1\n")

print("All vjp examples passed.")
