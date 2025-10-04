# The file is adapted from the Flax library (https://github.com/google/flax).
# The credit should go to the Flax authors.
#
# Copyright 2024 The Flax Authors & 2024 BrainX Ecosystem.
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


import jax
import jax.numpy as jnp
import brainstate


class Block(brainstate.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = brainstate.nn.Linear(dim, dim)
        self.bn = brainstate.nn.BatchNorm0d([dim])
        self.dropout = brainstate.nn.Dropout(0.5)

    def __call__(self, x: jax.Array):
        return jax.nn.gelu(self.dropout(self.bn(self.linear(x))))


class ScanMLP(brainstate.nn.Module):
    """
    An MLP that uses `vmap` during `__init__` to create a Block instance
    with an additional `layer` axis, and `scan` during `__call__` to apply
    the sequence of layers iteratively over the input / output `x`.
    """

    def __init__(self, dim: int, *, n_layers: int):
        super().__init__()
        self.n_layers = n_layers

        self.layers = [Block(dim) for _ in range(n_layers)]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x



x = jnp.ones((3, 10))
model = ScanMLP(10, n_layers=5)
with brainstate.environ.context(fit=True):
    y = model(x)

print(jax.tree.map(jnp.shape, brainstate.graph.treefy_states(model)))
print(y.shape)
