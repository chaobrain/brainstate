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

import jax
import jax.numpy as jnp

import brainstate
from brainstate.graph import states as graph_states


class TestEvalShape:
    def test_eval_shape_simple(self):
        out = brainstate.transform.eval_shape(lambda x: x * 2.0, jnp.ones(3))
        assert out.shape == (3,)
        assert out.dtype == jnp.float32

    def test_eval_shape_with_node(self):
        model = brainstate.transform.eval_shape(lambda: brainstate.nn.LSTMCell(3, 4))
        assert isinstance(model, brainstate.nn.LSTMCell)
