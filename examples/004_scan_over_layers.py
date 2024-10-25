# The file is adapted from the Flax library (https://github.com/google/flax).
# The credit should go to the Flax authors.
#
# Copyright 2024 The Flax Authors & 2024 BDP Ecosystem.
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

import brainstate as bst


class Block(bst.nn.Module):
  def __init__(self, dim: int):
    super().__init__()
    self.linear = bst.nn.Linear(dim, dim)
    self.bn = bst.nn.BatchNorm0d([dim])
    self.dropout = bst.nn.Dropout(0.5)

  def __call__(self, x: jax.Array):
    return jax.nn.gelu(self.dropout(self.bn(self.linear(x))))


class ScanMLP(bst.nn.Module):
  """
  An MLP that uses `vmap` during `__init__` to create a Block instance
  with an additional `layer` axis, and `scan` during `__call__` to apply
  the sequence of layers iteratively over the input / output `x`.
  """

  def __init__(self, dim: int, *, n_layers: int):
    super().__init__()
    self.n_layers = n_layers

    @bst.augment.restore_rngs
    @bst.augment.vmap
    def create_block(key):
      bst.random.set_key(key)
      return Block(dim)

    self.layers = create_block(bst.random.split_key(n_layers))

  def __call__(self, x: jax.Array) -> jax.Array:
    activation = bst.ShortTermState(x)

    def loop_fn(block_tree):
      # Feed the output of the previous layer to the next layer
      block: Block = bst.graph.treefy_merge(graphdef, block_tree)
      activation.value = block(activation.value)

    # Loop over each layer in the block tree
    graphdef, statetree = bst.graph.treefy_split(self.layers)
    bst.compile.for_loop(loop_fn, statetree)
    return activation.value


model = ScanMLP(10, n_layers=5)

x = jnp.ones((3, 10))

with bst.environ.context(fit=True):
  y = model(x)

print(jax.tree.map(jnp.shape, bst.graph.treefy_states(model)))
print(y.shape)
