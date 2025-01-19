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


from tempfile import TemporaryDirectory
import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax

import brainstate as bst


class MLP(bst.nn.Module):
    def __init__(self, din: int, dmid: int, dout: int):
        super().__init__()
        self.dense1 = bst.nn.Linear(din, dmid)
        self.dense2 = bst.nn.Linear(dmid, dout)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        return x


def create_model(seed: int):
    bst.random.seed(seed)
    return MLP(10, 20, 30)


def create_and_save(seed: int, path: str):
  model = create_model(seed)
  state_tree = bst.graph.treefy_states(model)
  # Save the parameters
  checkpointer = orbax.PyTreeCheckpointer()
  checkpointer.save(os.path.join(path, 'state'), state_tree)


def load_model(path: str) -> MLP:
  # create that model with abstract shapes
  model = bst.augment.abstract_init(lambda: create_model(0))
  state_tree = bst.graph.treefy_states(model)
  # Load the parameters
  checkpointer = orbax.PyTreeCheckpointer()
  state_tree = checkpointer.restore(os.path.join(path, 'state'), item=state_tree)
  # update the model with the loaded state
  bst.graph.update_states(model, state_tree)
  return model


with TemporaryDirectory() as tmpdir:
    # create a checkpoint
    create_and_save(42, tmpdir)
    # load model from checkpoint
    model = load_model(tmpdir)
    # run the model
    y = model(jnp.ones((1, 10)))
    print(model)
    print(y)
