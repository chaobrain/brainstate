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
``brainstate.transform.vmap2`` — state-aware vectorization.

``vmap2`` mirrors :func:`jax.vmap` but routes execution through BrainState's
state-aware mapping engine. On top of ordinary array vectorization it also:

* tracks reads and writes to :class:`brainstate.State` objects across the mapped
  axis (declared with ``state_in_axes`` / ``state_out_axes``),
* splits random states so every lane draws *independent* randomness, then
  restores the global RNG so randomness is consumed exactly once per call, and
* scatters batched writes back onto the correct axis of each state.

Read-only states are broadcast to every lane automatically; only states a lane
*writes* (or that you want sliced on input) need to appear in the axis
selectors. ``...`` (``Ellipsis``) is the "match every state" selector.

This script demonstrates fifteen use cases, each vectorizing a single function:

1.  Basic vectorization over a leading axis.
2.  Broadcasting an argument with ``in_axes=None``.
3.  Relocating the mapped axis with ``out_axes``.
4.  Mapping over a pytree (dict) argument.
5.  Decorator form (``@vmap2``).
6.  A read-only state, broadcast to every lane.
7.  A batched state read and written across lanes.
8.  Per-lane independent random states.
9.  Monte Carlo estimation of pi with ``axis_size`` and no array inputs.
10. An ensemble of models with batched parameters and a shared input.
11. A collective (``psum``) across the mapped axis via ``axis_name``.
12. A state batched on a non-zero axis (axis 1).
13. Nested ``vmap2`` building a 2-D outer product.
14. Composition with ``brainstate.transform.jit``.
15. ``grad`` inside ``vmap2`` for per-lane gradients.
"""

import jax
import jax.numpy as jnp

import brainstate

vmap2 = brainstate.transform.vmap2


# -----------------------------------------------------------------------------
# 1. Basic vectorization
# -----------------------------------------------------------------------------
# Map a pure function over the leading axis of every positional argument, exactly
# like ``jax.vmap``. ``in_axes=(0, 0)`` says both arguments are batched on axis 0.
def add(x, y):
    return x + y


x = jnp.arange(5.0)
y = jnp.arange(5.0) * 2.0
out = vmap2(add, in_axes=(0, 0))(x, y)
print("1. Basic vectorization")
print(f"   add(x, y) = {out}")
assert jnp.allclose(out, x + y)
print()


# -----------------------------------------------------------------------------
# 2. Broadcasting an argument
# -----------------------------------------------------------------------------
# ``in_axes=None`` marks an argument as broadcast: it is shared, unsliced, by
# every lane. Here only ``x`` is mapped; ``factor`` is the same scalar for all.
def scale(x, factor):
    return x * factor


x = jnp.arange(5.0)
factor = jnp.array(10.0)
out = vmap2(scale, in_axes=(0, None))(x, factor)
print("2. Broadcasting an argument (in_axes=None)")
print(f"   scale(x, 10) = {out}")
assert jnp.allclose(out, x * 10.0)
print()


# -----------------------------------------------------------------------------
# 3. Relocating the mapped axis with out_axes
# -----------------------------------------------------------------------------
# ``in_axes=0`` consumes rows of a (5, 3) array; ``out_axes=1`` places the mapped
# axis second in the result, so the output is the transpose of the naive stack.
def double(row):
    return row * 2.0


x = jnp.ones((5, 3))
out = vmap2(double, in_axes=0, out_axes=1)(x)
print("3. Relocating the mapped axis (out_axes=1)")
print(f"   input {x.shape} -> output {out.shape}")
assert out.shape == (3, 5)
assert jnp.allclose(out, (x * 2.0).T)
print()


# -----------------------------------------------------------------------------
# 4. Mapping over a pytree argument
# -----------------------------------------------------------------------------
# ``in_axes`` mirrors the structure of the argument: each leaf gets its own axis
# spec. The single positional argument here is a dict with two batched leaves.
def combine(d):
    return d['a'] + d['b']


d = {'a': jnp.arange(5.0), 'b': jnp.ones(5)}
out = vmap2(combine, in_axes=({'a': 0, 'b': 0},))(d)
print("4. Mapping over a pytree (dict) argument")
print(f"   combine(d) = {out}")
assert jnp.allclose(out, d['a'] + d['b'])
print()


# -----------------------------------------------------------------------------
# 5. Decorator form
# -----------------------------------------------------------------------------
# With no positional function, ``vmap2`` returns a decorator. Bare ``@vmap2``
# uses the defaults (``in_axes=0``, ``out_axes=0``).
@vmap2
def cube(x):
    return x ** 3


out = cube(jnp.arange(5.0))
print("5. Decorator form (@vmap2)")
print(f"   cube(x) = {out}")
assert jnp.allclose(out, jnp.arange(5.0) ** 3)
print()


# -----------------------------------------------------------------------------
# 6. A read-only state, broadcast to every lane
# -----------------------------------------------------------------------------
# A state that is only *read* needs no axis selector: ``vmap2`` broadcasts its
# value to every lane. Each row of ``x`` is dotted with the shared ``weight``.
weight = brainstate.State(jnp.array([1.0, 2.0, 3.0]))


def weighted_sum(row):
    return jnp.sum(row * weight.value)


x = jnp.ones((4, 3))
out = vmap2(weighted_sum, in_axes=0)(x)
print("6. Read-only state broadcast to every lane")
print(f"   weighted_sum(x) = {out}")
assert jnp.allclose(out, jnp.full(4, 6.0))
print()


# -----------------------------------------------------------------------------
# 7. A batched state read and written across lanes
# -----------------------------------------------------------------------------
# ``state_in_axes`` slices the state per lane on input; ``state_out_axes``
# scatters each lane's write back along the mapped axis. ``...`` matches every
# state. After the call the state holds the per-lane results.
counter = brainstate.State(jnp.zeros(5))


def accumulate(x):
    counter.value = counter.value + x
    return counter.value


x = jnp.arange(5.0)
out = vmap2(accumulate, in_axes=0, state_in_axes=..., state_out_axes=...)(x)
print("7. Batched state read + written across lanes")
print(f"   returned = {out}")
print(f"   counter.value = {counter.value}")
assert jnp.allclose(out, x)
assert jnp.allclose(counter.value, x)
print()


# -----------------------------------------------------------------------------
# 8. Per-lane independent random states
# -----------------------------------------------------------------------------
# Random states are split per lane automatically, so a single ``randn()`` call
# inside the mapped function yields a *different* draw in every lane. The global
# RNG is restored afterwards, so randomness is consumed exactly once.
brainstate.random.seed(0)


def add_noise(x):
    return x + brainstate.random.randn()


out = vmap2(add_noise, in_axes=0)(jnp.zeros(5))
print("8. Per-lane independent random states")
print(f"   samples = {out}")
assert out.shape == (5,)
assert not jnp.allclose(out, out[0])  # lanes differ
print()


# -----------------------------------------------------------------------------
# 9. Monte Carlo estimation of pi
# -----------------------------------------------------------------------------
# With no array inputs, ``axis_size`` sets the number of lanes. Each lane draws
# its own point in the square [-1, 1]^2 and reports whether it falls inside the
# unit disk; the hit fraction times 4 estimates pi.
brainstate.random.seed(42)


def inside_unit_disk():
    xy = brainstate.random.uniform(-1.0, 1.0, size=(2,))
    return (jnp.sum(xy ** 2) <= 1.0).astype(jnp.float32)


n_samples = 20000
hits = vmap2(inside_unit_disk, axis_size=n_samples)()
pi_estimate = 4.0 * jnp.mean(hits)
print("9. Monte Carlo estimation of pi")
print(f"   pi ~= {pi_estimate:.4f} from {n_samples} samples")
assert 3.0 < float(pi_estimate) < 3.3
print()


# -----------------------------------------------------------------------------
# 10. An ensemble of models with batched parameters
# -----------------------------------------------------------------------------
# Stack each member's parameters along axis 0 of a state and broadcast the input
# (``in_axes=None``). ``state_in_axes=...`` slices the parameter states per lane,
# so every member runs on the same input and produces its own output.
n_models, in_dim, out_dim = 4, 3, 2
brainstate.random.seed(1)
weights = brainstate.State(brainstate.random.randn(n_models, in_dim, out_dim))
biases = brainstate.State(brainstate.random.randn(n_models, out_dim))


def predict(inp):
    return inp @ weights.value + biases.value


inp = brainstate.random.randn(in_dim)
out = vmap2(predict, in_axes=None, state_in_axes=..., axis_size=n_models)(inp)
print("10. Ensemble of models with batched parameters")
print(f"   ensemble output shape = {out.shape}")
assert out.shape == (n_models, out_dim)
assert jnp.allclose(out[0], inp @ weights.value[0] + biases.value[0], atol=1e-5)
print()


# -----------------------------------------------------------------------------
# 11. A collective across the mapped axis
# -----------------------------------------------------------------------------
# Naming the mapped axis (``axis_name='lane'``) lets a lane talk to its peers via
# collectives. ``psum`` sums each lane's exponential, turning per-lane scalars
# into a softmax over the whole batch.
def softmax_lane(logit):
    e = jnp.exp(logit)
    return e / jax.lax.psum(e, axis_name='lane')


logits = jnp.array([1.0, 2.0, 3.0, 4.0])
out = vmap2(softmax_lane, axis_name='lane')(logits)
print("11. Collective (psum) across the mapped axis")
print(f"   softmax(logits) = {out}")
assert jnp.allclose(out, jax.nn.softmax(logits))
print()


# -----------------------------------------------------------------------------
# 12. A state batched on a non-zero axis
# -----------------------------------------------------------------------------
# State axes need not be 0. Here a (3, 5) buffer is batched on axis 1, so each of
# the 5 lanes sees a length-3 column; writes scatter back along axis 1.
buffer = brainstate.State(jnp.zeros((3, 5)))


def add_column(col):
    buffer.value = buffer.value + col
    return buffer.value


x = jnp.ones((5, 3))
out = vmap2(add_column, in_axes=0, state_in_axes={1: buffer}, state_out_axes={1: buffer})(x)
print("12. State batched on a non-zero axis (axis 1)")
print(f"   buffer.value shape = {buffer.value.shape}")
assert buffer.value.shape == (3, 5)
assert jnp.allclose(buffer.value, jnp.ones((3, 5)))
print()


# -----------------------------------------------------------------------------
# 13. Nested vmap2 (2-D outer product)
# -----------------------------------------------------------------------------
# Composing two ``vmap2`` layers maps over two independent axes. The inner layer
# sweeps ``b`` (``a`` broadcast); the outer layer sweeps ``a`` (``b`` broadcast),
# producing the full outer product ``a[:, None] * b[None, :]``.
def multiply(a, b):
    return a * b


inner = vmap2(multiply, in_axes=(None, 0))
outer = vmap2(inner, in_axes=(0, None))
a = jnp.arange(3.0)
b = jnp.arange(4.0)
out = outer(a, b)
print("13. Nested vmap2 (2-D outer product)")
print(f"   outer product shape = {out.shape}")
assert out.shape == (3, 4)
assert jnp.allclose(out, a[:, None] * b[None, :])
print()


# -----------------------------------------------------------------------------
# 14. Composition with jit
# -----------------------------------------------------------------------------
# ``vmap2`` composes with the other BrainState transforms. Wrapping a vectorized,
# state-mutating call in ``brainstate.transform.jit`` compiles the whole thing;
# the state side effects still apply after the compiled call returns.
gain = brainstate.State(jnp.zeros(5))


def step(x):
    gain.value = gain.value + x
    return gain.value * 2.0


@brainstate.transform.jit
def run(x):
    return vmap2(step, in_axes=0, state_in_axes=..., state_out_axes=...)(x)


x = jnp.arange(5.0)
out = run(x)
print("14. Composition with jit")
print(f"   run(x) = {out}")
print(f"   gain.value = {gain.value}")
assert jnp.allclose(out, x * 2.0)
assert jnp.allclose(gain.value, x)
print()


# -----------------------------------------------------------------------------
# 15. grad inside vmap2 (per-lane gradients)
# -----------------------------------------------------------------------------
# Each lane differentiates its own scalar problem. Mapping a per-example gradient
# over a batch yields the batch of gradients in a single call -- the diagonal of
# the batched Jacobian.
def per_lane_grad(x):
    return brainstate.transform.grad(lambda y: jnp.sum(y ** 2))(x)


x = jnp.arange(5.0)
out = vmap2(per_lane_grad, in_axes=0)(x)
print("15. grad inside vmap2 (per-lane gradients)")
print(f"   d/dx sum(x^2) = {out}")
assert jnp.allclose(out, 2.0 * x)
print()


print("All vmap2 examples passed.")
