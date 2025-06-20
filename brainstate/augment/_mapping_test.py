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
import numpy as np
from jax import vmap
from jax.lax import psum, pmean, pmax

import brainstate
import brainstate.augment
from brainstate.augment._mapping import BatchAxisError
from brainstate.augment._mapping import _remove_axis


class TestVmap(unittest.TestCase):
    def test_vmap_1(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = brainstate.State(brainstate.random.randn(5))
                self.b = brainstate.State(brainstate.random.randn(5))

            def __call__(self, *args, **kwargs):
                return self.a.value * self.b.value

        model = Model()
        r1 = model.a.value * model.b.value
        r2 = brainstate.augment.vmap(model, in_states=model.states())()
        self.assertTrue(jnp.allclose(r1, r2))

    def test_vmap_2(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = brainstate.ShortTermState(brainstate.random.randn(5))
                self.b = brainstate.ShortTermState(brainstate.random.randn(5))
                self.c = brainstate.State(brainstate.random.randn(1))

            def __call__(self, *args, **kwargs):
                self.c.value = self.a.value * self.b.value
                return self.c.value + 1.

        model = Model()
        with self.assertRaises(BatchAxisError):
            r2 = brainstate.augment.vmap(model, in_states=model.states(brainstate.ShortTermState))()

        model = Model()
        r2 = brainstate.augment.vmap(model, in_states=model.states(brainstate.ShortTermState), out_states=model.c)()

    def test_vmap_3(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = brainstate.State(brainstate.random.randn(5))
                self.b = brainstate.State(brainstate.random.randn(5))

            def __call__(self, *args, **kwargs):
                return self.a.value * self.b.value

        model = Model()
        with self.assertRaises(BatchAxisError):
            r2 = brainstate.augment.vmap(model, in_states=model.states(), out_states={1: model.states()})()

    def test_vmap_with_random(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = brainstate.ShortTermState(brainstate.random.randn(5))
                self.b = brainstate.ShortTermState(brainstate.random.randn(5))
                self.c = brainstate.State(brainstate.random.randn(1))

            def __call__(self, key):
                brainstate.random.set_key(key)
                self.c.value = self.a.value * self.b.value
                return self.c.value + brainstate.random.randn(1)

        model = Model()
        r2 = brainstate.augment.vmap(
            model,
            in_states=model.states(brainstate.ShortTermState),
            out_states=model.c
        )(
            brainstate.random.split_key(5)
        )
        print(brainstate.random.DEFAULT)

    def test_vmap_with_random_v3(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = brainstate.ShortTermState(brainstate.random.randn(5))
                self.b = brainstate.ShortTermState(brainstate.random.randn(5))
                self.c = brainstate.State(brainstate.random.randn(1))

            def __call__(self):
                self.c.value = self.a.value * self.b.value
                return self.c.value + brainstate.random.randn(1)

        model = Model()
        r2 = brainstate.augment.vmap(
            model,
            in_states=model.states(brainstate.ShortTermState),
            out_states=model.c
        )()
        print(brainstate.random.DEFAULT)

    def test_vmap_with_random_2(self):
        class Model(brainstate.nn.Module):
            def __init__(self):
                super().__init__()

                self.a = brainstate.ShortTermState(brainstate.random.randn(5))
                self.b = brainstate.ShortTermState(brainstate.random.randn(5))
                self.c = brainstate.State(brainstate.random.randn(1))
                self.rng = brainstate.random.RandomState(1)

            def __call__(self, key):
                self.rng.set_key(key)
                self.c.value = self.a.value * self.b.value
                return self.c.value + brainstate.random.randn(1)

        model = Model()
        r2 = brainstate.augment.vmap(
            model,
            in_states=model.states(brainstate.ShortTermState),
            out_states=model.c
        )(
            brainstate.random.split_key(5)
        )

    def test_vmap_input(self):
        model = brainstate.nn.Linear(2, 3)
        print(id(model), id(model.weight))
        model_id = id(model)
        weight_id = id(model.weight)

        x = jnp.ones((5, 2))

        @brainstate.augment.vmap
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
        gru = brainstate.nn.GRUCell(2, 3)
        gru.init_state(5)

        @brainstate.augment.vmap(in_states=gru.states(brainstate.HiddenState))
        def forward(x):
            return gru(x)

        xs = brainstate.random.randn(5, 2)
        y = forward(xs)
        self.assertTrue(y.shape == (5, 3))

    def test_vmap_jit(self):
        class Foo(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.ParamState(jnp.arange(4))
                self.b = brainstate.ShortTermState(jnp.arange(4))

            def __call__(self):
                self.b.value = self.a.value * self.b.value

        foo = Foo()

        @brainstate.augment.vmap(in_states=foo.states())
        def mul():
            foo()

        @brainstate.compile.jit
        def mul_jit(inp):
            mul()
            foo.a.value += inp

        with brainstate.StateTraceStack() as trace:
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
        class Foo(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.ParamState(jnp.arange(4))
                self.b = brainstate.ShortTermState(jnp.arange(4))

            def __call__(self):
                self.b.value = self.a.value * self.b.value

        foo = Foo()

        @brainstate.augment.vmap(in_states=foo.states())
        def mul():
            foo()

        @brainstate.compile.jit
        def mul_jit(inp):
            mul()
            foo.b.value += inp

        with brainstate.StateTraceStack() as trace:
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
            return brainstate.random.rand(1)

        res = brainstate.augment.vmap(f, axis_size=10)()
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
        class Foo(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = brainstate.ParamState(jnp.arange(4))
                self.b = brainstate.ShortTermState(jnp.arange(4))

            def init_state_v1(self, *args, **kwargs):
                self.c = brainstate.State(jnp.arange(4))

            def init_state_v2(self):
                self.d = brainstate.State(self.c.value * 2.)

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
            x = brainstate.random.rand(*dim)
            r1 = brainstate.augment.map(lambda a: a + 1, x, batch_size=None)
            r2 = brainstate.augment.map(lambda a: a + 1, x, batch_size=2)
            r3 = brainstate.augment.map(lambda a: a + 1, x, batch_size=4)
            r4 = brainstate.augment.map(lambda a: a + 1, x, batch_size=5)
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


class TestVMAPNewStatesEdgeCases(unittest.TestCase):

    def test_axis_size_zero(self):
        foo = brainstate.nn.LIF(3)
        # Testing that axis_size of 0 raises an error.
        with self.assertRaises(ValueError):
            @brainstate.augment.vmap_new_states(state_tag='new1', axis_size=0)
            def faulty_init():
                foo.init_state()

            # Call the decorated function to trigger validation
            faulty_init()

    def test_axis_size_negative(self):
        foo = brainstate.nn.LIF(3)
        # Testing that a negative axis_size raises an error.
        with self.assertRaises(ValueError):
            @brainstate.augment.vmap_new_states(state_tag='new1', axis_size=-3)
            def faulty_init():
                foo.init_state()

            faulty_init()

    def test_incompatible_shapes(self):
        foo = brainstate.nn.LIF(3)

        # Simulate an incompatible shapes scenario:
        # We intentionally assign a state with a different shape than expected.
        @brainstate.augment.vmap_new_states(state_tag='new1', axis_size=5)
        def faulty_init():
            # Modify state to produce an incompatible shape
            foo.c = brainstate.State(jnp.arange(3))  # Original expected shape is (4,)

        faulty_init()


class TestAxisName:
    def test1(self):
        def compute_stats_with_axis_name(x):
            """Compute statistics using named axis operations"""
            # Sum across the named axis 'batch'
            total_sum = psum(x, axis_name='batch')

            # Mean across the named axis 'batch'
            mean_val = pmean(x, axis_name='batch')

            # Max across the named axis 'batch'
            max_val = pmax(x, axis_name='batch')

            return {
                'sum': total_sum,
                'mean': mean_val,
                'max': max_val,
                'original': x
            }

        batch_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        print("Input batch data:", batch_data)

        # vmap with axis name 'batch'
        vectorized_stats_jax = jax.jit(vmap(compute_stats_with_axis_name, axis_name='batch'))
        result_jax = vectorized_stats_jax(batch_data)

        # vmap with axis name 'batch'
        vectorized_stats = brainstate.transform.vmap(compute_stats_with_axis_name, axis_name='batch')
        result = vectorized_stats(batch_data)

        # vmap with axis name 'batch'
        vectorized_stats_v2 = brainstate.transform.jit(
            brainstate.transform.vmap(compute_stats_with_axis_name, axis_name='batch')
        )
        result_v2 = vectorized_stats_v2(batch_data)

        for key in result_jax.keys():
            print(f"  {key}: {result_jax[key]}")
            assert jnp.allclose(result_jax[key], result[key]), f"Mismatch in {key}"
            assert jnp.allclose(result_jax[key], result_v2[key]), f"Mismatch in {key}"

    def test_nested_vmap(self):
        def nested_computation(x):
            """Computation with multiple named axes"""
            # Sum over 'inner' axis, then mean over 'outer' axis
            inner_sum = psum(x, axis_name='inner')
            outer_mean = pmean(inner_sum, axis_name='outer')
            return outer_mean

        # Create 2D batch data
        data_2d = jnp.arange(12.0).reshape(3, 4)  # Shape: [outer_batch=3, inner_batch=4]
        print("Input 2D data shape:", data_2d.shape)
        print("Input 2D data:\n", data_2d)

        # Nested vmap: first over inner dimension, then outer dimension
        inner_vmap = vmap(nested_computation, axis_name='inner')
        nested_vmap = vmap(inner_vmap, axis_name='outer')

        result_2d = nested_vmap(data_2d)
        print("Result after nested vmap:", result_2d)

        inner_vmap_bst = brainstate.transform.vmap(nested_computation, axis_name='inner')
        nested_vmap_bst = brainstate.transform.vmap(inner_vmap_bst, axis_name='outer')
        result_2d_bst = nested_vmap_bst(data_2d)
        print("Result after nested vmap:", result_2d_bst)

        assert jnp.allclose(result_2d, result_2d_bst)

    def _gradient_averaging_simulation_bst(self):
        def loss_function(params, x, y):
            """Simple quadratic loss"""
            pred = params * x
            return (pred - y) ** 2

        def compute_gradients_with_averaging(params, batch_x, batch_y):
            """Compute gradients and average them across the batch"""
            # Compute per-sample gradients
            grad_fn = jax.grad(loss_function, argnums=0)
            per_sample_grads = vmap(grad_fn, in_axes=(None, 0, 0))(params, batch_x, batch_y)

            # Average gradients across batch using named axis
            def average_grads(grads):
                return pmean(grads, axis_name='batch')

            # Apply averaging with named axis
            averaged_grads = vmap(average_grads, axis_name='batch')(per_sample_grads)
            return averaged_grads

        # Example data
        params = 2.0
        batch_x = jnp.array([1.0, 2.0, 3.0, 4.0])
        batch_y = jnp.array([2.0, 4.0, 7.0, 8.0])

        print("Parameters:", params)
        print("Batch X:", batch_x)
        print("Batch Y:", batch_y)

        # Compute individual gradients first
        grad_fn = jax.grad(loss_function, argnums=0)
        individual_grads = vmap(grad_fn, in_axes=(None, 0, 0))(params, batch_x, batch_y)
        print("Individual gradients:", individual_grads)

        # Now compute averaged gradients using axis names
        averaged_grads = compute_gradients_with_averaging(params, batch_x, batch_y)
        print("Averaged gradients:", averaged_grads)

        return individual_grads, averaged_grads

    def _gradient_averaging_simulation_jax(self):
        def loss_function(params, x, y):
            """Simple quadratic loss"""
            pred = params * x
            return (pred - y) ** 2

        def compute_gradients_with_averaging(params, batch_x, batch_y):
            """Compute gradients and average them across the batch"""
            # Compute per-sample gradients
            grad_fn = jax.grad(loss_function, argnums=0)
            per_sample_grads = brainstate.transform.vmap(grad_fn, in_axes=(None, 0, 0))(params, batch_x, batch_y)

            # Average gradients across batch using named axis
            def average_grads(grads):
                return pmean(grads, axis_name='batch')

            # Apply averaging with named axis
            averaged_grads = brainstate.transform.vmap(average_grads, axis_name='batch')(per_sample_grads)
            return averaged_grads

        # Example data
        params = 2.0
        batch_x = jnp.array([1.0, 2.0, 3.0, 4.0])
        batch_y = jnp.array([2.0, 4.0, 7.0, 8.0])

        print("Parameters:", params)
        print("Batch X:", batch_x)
        print("Batch Y:", batch_y)

        # Compute individual gradients first
        grad_fn = jax.grad(loss_function, argnums=0)
        individual_grads = brainstate.transform.vmap(grad_fn, in_axes=(None, 0, 0))(params, batch_x, batch_y)
        print("Individual gradients:", individual_grads)

        # Now compute averaged gradients using axis names
        averaged_grads = compute_gradients_with_averaging(params, batch_x, batch_y)
        print("Averaged gradients:", averaged_grads)

        return individual_grads, averaged_grads

    def test_gradient_averaging_simulation(self):
        individual_grads, averaged_grads = self._gradient_averaging_simulation_bst()
        individual_grads_jax, averaged_grads_jax = self._gradient_averaging_simulation_jax()
        assert jnp.allclose(individual_grads, individual_grads_jax)
        assert jnp.allclose(averaged_grads, averaged_grads_jax)




