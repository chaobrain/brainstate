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


import unittest

import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import psum, pmean, pmax

import brainstate
import brainstate.transform
from brainstate._error import BatchAxisError



class TestMap(unittest.TestCase):
    def test_map(self):
        for dim in [(10,), (10, 10), (10, 10, 10)]:
            x = brainstate.random.rand(*dim)
            r1 = brainstate.transform.map(lambda a: a + 1, x, batch_size=None)
            r2 = brainstate.transform.map(lambda a: a + 1, x, batch_size=2)
            r3 = brainstate.transform.map(lambda a: a + 1, x, batch_size=4)
            r4 = brainstate.transform.map(lambda a: a + 1, x, batch_size=5)
            true_r = x + 1

            self.assertTrue(jnp.allclose(r1, true_r))
            self.assertTrue(jnp.allclose(r2, true_r))
            self.assertTrue(jnp.allclose(r3, true_r))
            self.assertTrue(jnp.allclose(r4, true_r))


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

