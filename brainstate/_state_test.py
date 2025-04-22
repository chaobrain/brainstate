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


class TestStateRepr(unittest.TestCase):

    def test_state_repr(self):
        print()

        state = bst.State(bst.random.randn(10))
        print(state)

        state2 = bst.State({'a': bst.random.randn(10), 'b': bst.random.randn(10)})
        print(state2)

        state3 = bst.State([bst.random.randn(10), bst.random.randn(10)])
        print(state3)
