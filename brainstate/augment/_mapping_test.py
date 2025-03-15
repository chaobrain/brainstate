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

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import brainstate as bst
import brainstate.augment
from brainstate.augment._mapping import BatchAxisError
from brainstate.augment._mapping import _remove_axis


class TestVmap(unittest.TestCase):
    def test_vmap_1(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.State(bst.random.randn(5))
                self.b = bst.State(bst.random.randn(5))

            def __call__(self, *args, **kwargs):
                return self.a.value * self.b.value

        model = Model()
        r1 = model.a.value * model.b.value
        r2 = bst.augment.vmap(model, in_states=model.states())()
        self.assertTrue(jnp.allclose(r1, r2))

    def test_vmap_2(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.ShortTermState(bst.random.randn(5))
                self.b = bst.ShortTermState(bst.random.randn(5))
                self.c = bst.State(bst.random.randn(1))

            def __call__(self, *args, **kwargs):
                self.c.value = self.a.value * self.b.value
                return self.c.value + 1.

        model = Model()
        with self.assertRaises(BatchAxisError):
            r2 = bst.augment.vmap(model, in_states=model.states(bst.ShortTermState))()

        model = Model()
        r2 = bst.augment.vmap(model, in_states=model.states(bst.ShortTermState), out_states=model.c)()

    def test_vmap_3(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.State(bst.random.randn(5))
                self.b = bst.State(bst.random.randn(5))

            def __call__(self, *args, **kwargs):
                return self.a.value * self.b.value

        model = Model()
        with self.assertRaises(BatchAxisError):
            r2 = bst.augment.vmap(model, in_states=model.states(), out_states={1: model.states()})()

    def test_vmap_with_random(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.ShortTermState(bst.random.randn(5))
                self.b = bst.ShortTermState(bst.random.randn(5))
                self.c = bst.State(bst.random.randn(1))

            def __call__(self, key):
                bst.random.set_key(key)
                self.c.value = self.a.value * self.b.value
                return self.c.value + bst.random.randn(1)

        model = Model()
        r2 = bst.augment.vmap(
            model,
            in_states=model.states(bst.ShortTermState),
            out_states=model.c
        )(
            bst.random.split_key(5)
        )
        print(bst.random.DEFAULT)

    def test_vmap_with_random_v3(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.ShortTermState(bst.random.randn(5))
                self.b = bst.ShortTermState(bst.random.randn(5))
                self.c = bst.State(bst.random.randn(1))

            def __call__(self):
                self.c.value = self.a.value * self.b.value
                return self.c.value + bst.random.randn(1)

        model = Model()
        r2 = bst.augment.vmap(
            model,
            in_states=model.states(bst.ShortTermState),
            out_states=model.c
        )()
        print(bst.random.DEFAULT)

    def test_vmap_with_random_2(self):
        class Model(bst.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = bst.ShortTermState(bst.random.randn(5))
                self.b = bst.ShortTermState(bst.random.randn(5))
                self.c = bst.State(bst.random.randn(1))
                self.rng = bst.random.RandomState(1)

            def __call__(self, key):
                self.rng.set_key(key)
                self.c.value = self.a.value * self.b.value
                return self.c.value + bst.random.randn(1)

        model = Model()
        r2 = bst.augment.vmap(
            model,
            in_states=model.states(bst.ShortTermState),
            out_states=model.c
        )(
            bst.random.split_key(5)
        )

    def test_vmap_input(self):
        model = bst.nn.Linear(2, 3)
        print(id(model), id(model.weight))
        model_id = id(model)
        weight_id = id(model.weight)

        x = jnp.ones((5, 2))

        @bst.augment.vmap
        def forward(x):
            self.assertTrue(id(model) == model_id)
            self.assertTrue(id(model.weight) == weight_id)
            return model(x)

        y = forward(x)
        self.assertTrue(y.shape == (5, 3))
        print(y.shape)
        print(model.weight.value_call(jnp.shape))
        print(model.weight.value)

    def test_vmap_states_and_input_1(self):
        gru = bst.nn.GRUCell(2, 3)
        gru.init_state(5)

        @bst.augment.vmap(in_states=gru.states(bst.HiddenState))
        def forward(x):
            return gru(x)

        xs = bst.random.randn(5, 2)
        y = forward(xs)
        self.assertTrue(y.shape == (5, 3))

    def test_vmap_jit(self):
        class Foo(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = bst.ParamState(jnp.arange(4))
                self.b = bst.ShortTermState(jnp.arange(4))

            def __call__(self):
                self.b.value = self.a.value * self.b.value

        foo = Foo()

        @bst.augment.vmap(in_states=foo.states())
        def mul():
            foo()

        @bst.compile.jit
        def mul_jit(inp):
            mul()
            foo.a.value += inp

        with bst.StateTraceStack() as trace:
            mul_jit(1.)

        print(foo.a.value)
        print(foo.b.value)
        self.assertTrue(jnp.allclose(foo.a.value, jnp.arange(4) + 1.))
        self.assertTrue(jnp.allclose(foo.b.value, jnp.arange(4) * jnp.arange(4)))

        write_state_ids = [id(st) for st in trace.get_write_states()]
        read_state_ids = [id(st) for st in trace.get_read_states()]

        assert id(foo.a) in write_state_ids
        assert id(foo.b) in write_state_ids

        print(trace.get_write_states())
        print(trace.get_read_states())

    def test_vmap_jit_2(self):
        class Foo(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = bst.ParamState(jnp.arange(4))
                self.b = bst.ShortTermState(jnp.arange(4))

            def __call__(self):
                self.b.value = self.a.value * self.b.value

        foo = Foo()

        @bst.augment.vmap(in_states=foo.states())
        def mul():
            foo()

        @bst.compile.jit
        def mul_jit(inp):
            mul()
            foo.b.value += inp

        with bst.StateTraceStack() as trace:
            mul_jit(1.)

        print(foo.a.value)
        print(foo.b.value)
        self.assertTrue(jnp.allclose(foo.a.value, jnp.arange(4)))
        self.assertTrue(jnp.allclose(foo.b.value, jnp.arange(4) * jnp.arange(4) + 1.))

        write_state_ids = [id(st) for st in trace.get_write_states()]
        read_state_ids = [id(st) for st in trace.get_read_states()]

        assert id(foo.a) in read_state_ids
        assert id(foo.b) in write_state_ids

        print(trace.get_write_states())
        print(trace.get_read_states())

    def test_auto_rand_key_split(self):
        def f():
            return bst.random.rand(1)

        res = bst.augment.vmap(f, axis_size=10)()
        self.assertTrue(jnp.all(~(res[0] == res[1:])))

        res2 = jax.vmap(f, axis_size=10)()
        self.assertTrue(jnp.all((res2[0] == res2[1:])))

    def test_axis(self):
        def f(x):
            return x - jax.lax.pmean(x, 'i')

        r = jax.vmap(f, axis_name='i')(jnp.arange(10))
        print(r)

        r2 = brainstate.augment.vmap(f, axis_name='i')(jnp.arange(10))
        print(r2)
        self.assertTrue(jnp.allclose(r, r2))

    def test_vmap_init(self):
        class Foo(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = bst.ParamState(jnp.arange(4))
                self.b = bst.ShortTermState(jnp.arange(4))

            def init_state_v1(self, *args, **kwargs):
                self.c = bst.State(jnp.arange(4))

            def init_state_v2(self):
                self.d = bst.State(self.c.value * 2.)

        foo = Foo()

        @brainstate.augment.vmap_new_states(state_tag='new1', axis_size=5)
        def init1():
            foo.init_state_v1()

        init1()
        print(foo.c.value)

        @brainstate.augment.vmap_new_states(state_tag='new2', axis_size=5, in_states=foo.states('new1'))
        def init2():
            foo.init_state_v2()

        init2()
        print(foo.c.value)
        print(foo.d.value)

        self.assertTrue(
            jnp.allclose(
                foo.d.value,
                foo.c.value * 2.
            )
        )



class TestMap(unittest.TestCase):
    def test_map(self):
        for dim in [(10,), (10, 10), (10, 10, 10)]:
            x = bst.random.rand(*dim)
            r1 = bst.augment.map(lambda a: a + 1, x, batch_size=None)
            r2 = bst.augment.map(lambda a: a + 1, x, batch_size=2)
            r3 = bst.augment.map(lambda a: a + 1, x, batch_size=4)
            r4 = bst.augment.map(lambda a: a + 1, x, batch_size=5)
            true_r = x + 1

            self.assertTrue(jnp.allclose(r1, true_r))
            self.assertTrue(jnp.allclose(r2, true_r))
            self.assertTrue(jnp.allclose(r3, true_r))
            self.assertTrue(jnp.allclose(r4, true_r))


class TestRemoveAxis:

    def test_remove_axis_2d_array_axis_0(self):
        input_array = np.array([[1, 2, 3], [4, 5, 6]])
        expected_output = np.array([1, 2, 3])

        result = _remove_axis(input_array, axis=0)

        np.testing.assert_array_equal(result, expected_output)

    def test_remove_axis_3d_array(self):
        # Create a 3D array
        x = np.arange(24).reshape((2, 3, 4))

        # Remove axis 1
        result = _remove_axis(x, axis=1)

        # Expected result: a 2D array with shape (2, 4)
        expected = x[:, 0, :]

        np.testing.assert_array_equal(result, expected)
        assert result.shape == (2, 4)

    def test_remove_axis_1d_array(self):
        # Create a 1D array
        x = np.array([1, 2, 3, 4, 5])

        # Remove axis 0 (the only axis in a 1D array)
        result = _remove_axis(x, axis=0)

        # Check that the result is a scalar (0D array) and equal to the first element
        assert np.isscalar(result), "Result should be a scalar"
        assert result == 1, "Result should be equal to the first element of the input array"

    def test_remove_axis_out_of_bounds(self):
        x = jnp.array([[1, 2], [3, 4]])
        with unittest.TestCase().assertRaises(IndexError):
            _remove_axis(x, axis=2)

    def test_remove_axis_negative(self):
        x = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = _remove_axis(x, -1)
        expected = jnp.array([[1, 3], [5, 7]])
        np.testing.assert_array_equal(result, expected)

    def test_remove_axis_with_nan_and_inf(self):
        x = jnp.array([[1.0, jnp.nan, 3.0], [4.0, 5.0, jnp.inf]])
        result = _remove_axis(x, axis=0)
        expected = jnp.array([1.0, jnp.nan, 3.0])
        np.testing.assert_array_equal(result, expected)
        assert jnp.isnan(result[1])

    def test_remove_axis_different_dtypes(self):
        # Test with integer array
        int_array = jnp.array([[1, 2, 3], [4, 5, 6]])
        int_result = _remove_axis(int_array, 0)
        assert jnp.array_equal(int_result, jnp.array([1, 2, 3]))

        # Test with float array
        float_array = jnp.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        float_result = _remove_axis(float_array, 1)
        assert jnp.allclose(float_result, jnp.array([1.1, 4.4]))

        # Test with complex array
        complex_array = jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        complex_result = _remove_axis(complex_array, 0)
        assert jnp.allclose(complex_result, jnp.array([1 + 1j, 2 + 2j]))
