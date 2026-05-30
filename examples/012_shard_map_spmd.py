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
``brainstate.transform.shard_map`` — single-program multiple-data (SPMD) sharding.

``shard_map`` runs a *stateful* function over shards of data placed across a
device mesh. It is a state-aware wrapper over :func:`jax.shard_map`: positional
arguments are split per ``in_specs``, :class:`brainstate.State` objects are
replicated (default) or sharded per ``state_in_specs`` / ``state_out_specs``,
and any state writes inside the function are threaded back out afterwards.

This script walks through seven self-contained use cases:

1. Basic data-parallel elementwise map (no state).
2. A replicated read-only parameter (``ParamState`` read but never written).
3. A *sharded* write-back buffer accumulated in place across calls.
4. Cross-shard communication with the ``jax.lax.psum`` collective.
5. Data-parallel evaluation of a ``brainstate.nn`` layer over a batch.
6. Composition with ``jax.jit`` to amortise the per-call re-trace.
7. A 2-D mesh combining data- and model- (tensor-) parallelism.

Most machines expose a single device, so we ask XLA to emulate eight CPU
devices. This MUST happen before JAX is imported, hence it is the very first
thing the script does.
"""

import os

# Emulate 8 devices on a single host. Must be set before importing jax.
os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import brainstate

N = jax.device_count()
print(f"Running shard_map examples across {N} devices: {jax.devices()}\n")

# A 1-D mesh whose single axis we name 'x'. PartitionSpec('x') means "split this
# array's leading dimension across the devices along the 'x' axis"; PartitionSpec()
# (empty) means "replicate".
mesh = jax.make_mesh((N,), ('x',))


# -----------------------------------------------------------------------------
# 1. Basic data-parallel map (no state)
# -----------------------------------------------------------------------------
# Each device receives a contiguous slice of ``data`` and applies ``fun`` to it.
# The global result is the concatenation of every per-shard result, so it is
# identical to running ``fun`` on the whole array.
def double(data):
    return data * 2.0


f_double = brainstate.transform.shard_map(
    double, mesh, in_specs=(P('x'),), out_specs=P('x'),
)
data = jnp.arange(N * 2, dtype=jnp.float32)
out = f_double(data)
print("1. Data-parallel map (no state)")
print(f"   in  = {data}")
print(f"   out = {out}")
assert jnp.allclose(out, data * 2.0)
print("   matches data * 2.0\n")


# -----------------------------------------------------------------------------
# 2. Replicated read-only parameter
# -----------------------------------------------------------------------------
# A scalar ``ParamState`` is read but never written. By default states are
# replicated (PartitionSpec()), so every device sees the same value, and the
# State is left untouched after the call.
weight = brainstate.ParamState(jnp.array(3.0))


def scale(data):
    return data * weight.value


f_scale = brainstate.transform.shard_map(
    scale, mesh, in_specs=(P('x'),), out_specs=P('x'),
)
out = f_scale(data)
print("2. Replicated read-only parameter")
print(f"   out = {out}")
assert jnp.allclose(out, data * 3.0)
assert jnp.allclose(weight.value, 3.0)  # read-only state preserved
print("   parameter replicated and preserved after the call\n")


# -----------------------------------------------------------------------------
# 3. Sharded write-back buffer (per-shard accumulation)
# -----------------------------------------------------------------------------
# Give a State an explicit P('x') partition so each device keeps its own slice.
# The buffer is read and written in place; ``state_out_specs`` re-threads the new
# values back into the State so repeated calls accumulate.
buffer = brainstate.State(jnp.zeros(N * 2))


def accumulate(data):
    buffer.value = buffer.value + data
    return data


f_accum = brainstate.transform.shard_map(
    accumulate, mesh, in_specs=(P('x'),), out_specs=P('x'),
    state_in_specs={buffer: P('x')}, state_out_specs={buffer: P('x')},
)
ones = jnp.ones(N * 2)
for step in range(3):
    f_accum(ones)
print("3. Sharded write-back buffer")
print(f"   buffer after 3 accumulations of ones = {buffer.value}")
assert jnp.allclose(buffer.value, 3.0)
print("   each shard accumulated in place\n")


# -----------------------------------------------------------------------------
# 4. Cross-shard communication with a collective (psum)
# -----------------------------------------------------------------------------
# Devices are otherwise independent; ``jax.lax.psum`` sums a value across the
# named mesh axis so every shard ends up with the global total. Here each device
# contributes a partial sum, and the reduced total is replicated back (out_specs
# is the empty PartitionSpec).
def global_sum(data):
    partial = jnp.sum(data, keepdims=True)
    return jax.lax.psum(partial, axis_name='x')


f_sum = brainstate.transform.shard_map(
    global_sum, mesh, in_specs=(P('x'),), out_specs=P(),
)
data = jnp.arange(N * 4, dtype=jnp.float32)
out = f_sum(data)
print("4. Collective psum (global reduction)")
print(f"   sum over all shards = {out} (expected {jnp.sum(data)})")
assert jnp.allclose(out, jnp.sum(data))
print()


# -----------------------------------------------------------------------------
# 5. Data-parallel evaluation of a neural-network layer
# -----------------------------------------------------------------------------
# A ``brainstate.nn`` layer's parameters are ordinary States; by default they
# are replicated, so every device holds the full weights and applies the layer
# to its own slice of the batch. The result equals a single-device forward pass.
layer = brainstate.nn.Linear(8, 4)
batch = brainstate.random.randn(N * 2, 8)


def forward(x):
    return layer(x)


f_forward = brainstate.transform.shard_map(
    forward, mesh, in_specs=(P('x'),), out_specs=P('x'),
)
sharded_out = f_forward(batch)
reference = layer(batch)
print("5. Data-parallel layer over a batch")
print(f"   batch {tuple(batch.shape)} -> output {tuple(sharded_out.shape)}")
assert jnp.allclose(sharded_out, reference, atol=1e-5)
print("   sharded forward matches single-device forward\n")


# -----------------------------------------------------------------------------
# 6. Composition with jax.jit
# -----------------------------------------------------------------------------
# ``shard_map`` re-traces ``fun`` on every call to discover its state usage.
# Wrapping the sharded function in ``jax.jit`` amortises that on the hot path.
bias = brainstate.ParamState(jnp.array(5.0))


def add_bias(data):
    return data + bias.value


f_bias = brainstate.transform.shard_map(
    add_bias, mesh, in_specs=(P('x'),), out_specs=P('x'),
)
jit_f = jax.jit(f_bias)
data = jnp.arange(N * 2, dtype=jnp.float32)
out = jit_f(data)
print("6. Composition with jax.jit")
print(f"   jit(shard_map(...)) out = {out}")
assert jnp.allclose(out, data + 5.0)
print()


# -----------------------------------------------------------------------------
# 7. 2-D mesh: data + model (tensor) parallelism
# -----------------------------------------------------------------------------
# A 2-D mesh names two axes: 'data' (split the batch) and 'model' (split the
# contraction dimension of a weight matrix). We compute ``y = x @ W`` where the
# shared inner dimension is sharded across 'model'; each shard produces a partial
# product, and ``psum`` over the 'model' axis assembles the full result.
if N >= 2 and N % 2 == 0:
    d = N // 2
    mesh2d = jax.make_mesh((d, 2), ('data', 'model'))

    batch_size, d_in, d_out = d * 3, 2 * 4, 5  # divisible by ('data', 'model')
    x_full = brainstate.random.randn(batch_size, d_in)
    w_full = brainstate.random.randn(d_in, d_out)

    def tensor_parallel_matmul(x, w):
        # x: (batch/data, d_in/model)   w: (d_in/model, d_out)
        partial = x @ w                       # (batch/data, d_out) partial sum
        return jax.lax.psum(partial, axis_name='model')

    f_tp = brainstate.transform.shard_map(
        tensor_parallel_matmul, mesh2d,
        # batch over 'data', contraction dim over 'model'; W rows over 'model'.
        in_specs=(P('data', 'model'), P('model', None)),
        out_specs=P('data', None),
    )
    y = f_tp(x_full, w_full)
    reference = x_full @ w_full
    print("7. 2-D mesh: data + model parallelism")
    print(f"   mesh {dict(mesh2d.shape)}: x{tuple(x_full.shape)} @ W{tuple(w_full.shape)}"
          f" -> y{tuple(y.shape)}")
    assert jnp.allclose(y, reference, atol=1e-4)
    print("   tensor-parallel matmul matches the dense result\n")
else:
    print("7. 2-D mesh example skipped (needs an even device count >= 2)\n")

print("All shard_map examples passed.")
