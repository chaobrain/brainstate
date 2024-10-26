# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


import unittest

import jax
import jax.numpy as jnp

import brainstate as bst


class TestStateSourceInfo(unittest.TestCase):

  def test_state_source_info(self):
    state = bst.State(bst.random.randn(10))
    print(state._source_info)

  def test_state_value_tree(self):
    state = bst.ShortTermState(jnp.zeros((2, 3)))

    with bst.check_state_value_tree():
      state.value = jnp.zeros((2, 3))

      with self.assertRaises(ValueError):
        state.value = (jnp.zeros((2, 3)), jnp.zeros((2, 3)))

  def test_check_jax_tracer(self):
    a = bst.ShortTermState(jnp.zeros((2, 3)))

    @jax.jit
    def run_state(b):
      a.value = b
      return a.value

    # The following code will not raise an error, since the state is valid to trace.
    run_state(jnp.ones((2, 3)))

    with bst.check_state_jax_tracer():
      # The line below will not raise an error.
      with self.assertRaises(bst.util.TraceContextError):
        # recompile the function
        run_state(jnp.ones((2, 4)))


class TestStateRepr(unittest.TestCase):

  def test_state_repr(self):
    print()

    state = bst.State(bst.random.randn(10))
    print(state)

    state2 = bst.State({'a': bst.random.randn(10), 'b': bst.random.randn(10)})
    print(state2)

    state3 = bst.State([bst.random.randn(10), bst.random.randn(10)])
    print(state3)
