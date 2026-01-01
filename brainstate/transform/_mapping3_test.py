# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from __future__ import annotations

import unittest

import jax.core
import jax.numpy as jnp

import brainstate
from brainstate.transform._mapping3 import model_vmap


class TestVmap(unittest.TestCase):
    def test_vmap_return_keep_reference_return(self):
        @model_vmap(in_axes=0, out_axes=0)
        def create_model(key):
            brainstate.random.set_key(key)
            m1 = brainstate.nn.Linear(2, 3)

            m2 = brainstate.nn.Linear(3, 4)
            m2.a = m1
            m3 = brainstate.nn.Linear(3, 5)
            m3.a = m1
            self.assertTrue(id(m2.a) == id(m3.a))
            return m2, m3

        m2, m3 = create_model(brainstate.random.split_key(10, backup=True))
        self.assertTrue(id(m2.a) == id(m3.a))
        self.assertTrue(isinstance(brainstate.random.DEFAULT.value, jax.core.Tracer))
        brainstate.random.restore_key()
        jax.core.concrete_or_error(None, brainstate.random.DEFAULT.value)

    def test_vmap_return_keep_reference_pass_into_fun(self):
        @model_vmap(in_axes=(None, None, 0), out_axes=0)
        def run_model(m2, m3, x):
            self.assertTrue(id(m2.a) == id(m3.a))
            self.assertTrue(id(m2) != m2_id)
            self.assertTrue(id(m3) != m3_id)
            return m2(x), m3(x)

        m1 = brainstate.nn.Linear(2, 3)
        m2 = brainstate.nn.Linear(4, 3)
        m2.a = m1
        m3 = brainstate.nn.Linear(4, 5)
        m3.a = m1
        m3_id = id(m3)
        m2_id = id(m2)
        r1, r2 = run_model(m2, m3, jnp.ones((4, 3, 4)))

    def test_vmap_set_key(self):
        @model_vmap(in_axes=0, out_axes=0)
        def create_model(key):
            brainstate.random.set_key(key)
            return brainstate.nn.Linear(2, 3)

        model = create_model(brainstate.random.split_keys(10))
        print(model.weight.value_call(jnp.shape))
        self.assertTrue(isinstance(brainstate.random.DEFAULT.value, jax.core.Tracer))
        model.weight.value_call(lambda x: jax.core.concrete_or_error(None, x))
        brainstate.random.seed()

    def test_vmap_input(self):
        model = brainstate.nn.Linear(2, 3)
        print(id(model), id(model.weight))
        model_id = id(model)
        weight_id = id(model.weight)

        x = jnp.ones((5, 2))

        @model_vmap
        def forward(x):
            self.assertTrue(id(model) == model_id)
            self.assertTrue(id(model.weight) == weight_id)
            return model(x)

        y = forward(x)
        self.assertTrue(y.shape == (5, 3))
        print(y.shape)
        print(model.weight.value_call(jnp.shape))
        print(model.weight.value)

    def test_vmap_model(self):
        model = brainstate.nn.Linear(2, 3)
        model_id = id(model)
        weight_id = id(model.weight)
        print(id(model), id(model.weight))
        x = jnp.ones((5, 2))

        @model_vmap(in_axes=(None, 0), out_axes=0)
        def forward(model, x):
            self.assertTrue(id(model) != model_id)
            self.assertTrue(id(model.weight) != weight_id)
            print(id(model), id(model.weight))
            return model(x)

        y = forward(model, x)
        print(y.shape)
        print(model.weight.value_call(jnp.shape))
        print(model.weight.value)
