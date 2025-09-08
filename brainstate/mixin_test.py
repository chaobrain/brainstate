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

import brainstate


class TestMixin(unittest.TestCase):
    def test_mixin(self):
        self.assertTrue(brainstate.mixin.Mixin)
        self.assertTrue(brainstate.mixin.ParamDesc)
        self.assertTrue(brainstate.mixin.ParamDescriber)
        self.assertTrue(brainstate.mixin.JointTypes)
        self.assertTrue(brainstate.mixin.OneOfTypes)
        self.assertTrue(brainstate.mixin.Mode)
        self.assertTrue(brainstate.mixin.Batching)
        self.assertTrue(brainstate.mixin.Training)


class TestMode(unittest.TestCase):
    def test_JointMode(self):
        a = brainstate.mixin.JointMode(brainstate.mixin.Batching(), brainstate.mixin.Training())
        self.assertTrue(a.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Batching, brainstate.mixin.Training]))
        self.assertTrue(a.has(brainstate.mixin.Batching))
        self.assertTrue(a.has(brainstate.mixin.Training))
        b = brainstate.mixin.JointMode(brainstate.mixin.Batching())
        self.assertTrue(b.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Batching]))
        self.assertTrue(b.is_a(brainstate.mixin.Batching))
        self.assertTrue(b.has(brainstate.mixin.Batching))

    def test_Training(self):
        a = brainstate.mixin.Training()
        self.assertTrue(a.is_a(brainstate.mixin.Training))
        self.assertTrue(a.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Training]))
        self.assertTrue(a.has(brainstate.mixin.Training))
        self.assertTrue(a.has(brainstate.mixin.JointTypes[brainstate.mixin.Training]))
        self.assertFalse(a.is_a(brainstate.mixin.Batching))
        self.assertFalse(a.has(brainstate.mixin.Batching))

    def test_Batching(self):
        a = brainstate.mixin.Batching()
        self.assertTrue(a.is_a(brainstate.mixin.Batching))
        self.assertTrue(a.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Batching]))
        self.assertTrue(a.has(brainstate.mixin.Batching))
        self.assertTrue(a.has(brainstate.mixin.JointTypes[brainstate.mixin.Batching]))

        self.assertFalse(a.is_a(brainstate.mixin.Training))
        self.assertFalse(a.has(brainstate.mixin.Training))

    def test_Mode(self):
        a = brainstate.mixin.Mode()
        self.assertTrue(a.is_a(brainstate.mixin.Mode))
        self.assertTrue(a.is_a(brainstate.mixin.JointTypes[brainstate.mixin.Mode]))
        self.assertTrue(a.has(brainstate.mixin.Mode))
        self.assertTrue(a.has(brainstate.mixin.JointTypes[brainstate.mixin.Mode]))

        self.assertFalse(a.is_a(brainstate.mixin.Training))
        self.assertFalse(a.has(brainstate.mixin.Training))
        self.assertFalse(a.is_a(brainstate.mixin.Batching))
        self.assertFalse(a.has(brainstate.mixin.Batching))


class Array(brainstate.State, brainstate.mixin.ArrayImpl):
    pass


class TestArray(unittest.TestCase):
    def setUp(self):
        # Basic arrays for testing
        self.np_array = np.array([1, 2, 3])
        self.jax_array = jnp.array([1, 2, 3])
        # Create ArrayImpl instances
        self.array_impl_np = Array(self.np_array)
        self.array_impl_np.value = self.np_array
        self.array_impl_jax = Array(self.jax_array)
        self.array_impl_jax.value = self.jax_array
        # More complex arrays for advanced testing
        self.ones_2d = np.ones((2, 3))
        self.array_impl_2d = Array(self.ones_2d)
        self.array_impl_2d.value = self.ones_2d
        # Scalar array
        self.scalar = np.array(5.0)
        self.array_impl_scalar = Array(self.scalar)
        self.array_impl_scalar.value = self.scalar

    def test_basic_properties(self):
        # Test dtype property
        self.assertEqual(self.array_impl_np.dtype, self.np_array.dtype)
        self.assertEqual(self.array_impl_jax.dtype, self.jax_array.dtype)

        # Test shape property
        self.assertEqual(self.array_impl_np.shape, self.np_array.shape)
        self.assertEqual(self.array_impl_jax.shape, self.jax_array.shape)

        # Test ndim property
        self.assertEqual(self.array_impl_np.ndim, self.np_array.ndim)
        self.assertEqual(self.array_impl_2d.ndim, self.ones_2d.ndim)

        # Test size property
        self.assertEqual(self.array_impl_np.size, self.np_array.size)
        self.assertEqual(self.array_impl_2d.size, self.ones_2d.size)

        # Test real property (should work for real arrays)
        np.testing.assert_array_equal(self.array_impl_np.real, self.np_array.real)

        # Test T property (transpose)
        np.testing.assert_array_equal(self.array_impl_2d.T, self.ones_2d.T)

    def test_unary_operations(self):
        # Test __neg__
        np.testing.assert_array_equal(-self.array_impl_np, -self.np_array)

        # Test __pos__
        np.testing.assert_array_equal(+self.array_impl_np, +self.np_array)

        # Test __abs__
        neg_array = Array(np.array([-1, 2, -3]))
        np.testing.assert_array_equal(abs(neg_array), np.abs(neg_array.value))

        # Test __invert__ (bitwise NOT)
        uint_array = Array(np.array([1, 2, 3], dtype=np.uint8))
        np.testing.assert_array_equal(~uint_array, ~uint_array.value)

    def test_binary_operations(self):
        # Test __add__ and __radd__
        np.testing.assert_array_equal(self.array_impl_np + 2, self.np_array + 2)
        np.testing.assert_array_equal(2 + self.array_impl_np, 2 + self.np_array)

        # Test __sub__ and __rsub__
        np.testing.assert_array_equal(self.array_impl_np - 1, self.np_array - 1)
        np.testing.assert_array_equal(5 - self.array_impl_np, 5 - self.np_array)

        # Test __mul__ and __rmul__
        np.testing.assert_array_equal(self.array_impl_np * 2, self.np_array * 2)
        np.testing.assert_array_equal(2 * self.array_impl_np, 2 * self.np_array)

        # Test __truediv__ and __rtruediv__
        np.testing.assert_array_almost_equal(self.array_impl_np / 2, self.np_array / 2)
        np.testing.assert_array_almost_equal(6 / self.array_impl_np, 6 / self.np_array)

        # Test __pow__ and __rpow__
        np.testing.assert_array_equal(self.array_impl_np ** 2, self.np_array ** 2)
        np.testing.assert_array_equal(2 ** self.array_impl_np, 2 ** self.np_array)

        # Test __matmul__
        a = Array(np.array([[1, 2], [3, 4]]))
        b = Array(np.array([[5, 6], [7, 8]]))
        np.testing.assert_array_equal(a @ b, a.value @ b.value)

    def test_inplace_operations(self):
        # Test __iadd__
        test_array = Array(np.array([1, 2, 3]))
        test_array += 1
        np.testing.assert_array_equal(test_array.value, np.array([2, 3, 4]))

        # Test __isub__
        test_array = Array(np.array([1, 2, 3]))
        test_array -= 1
        np.testing.assert_array_equal(test_array.value, np.array([0, 1, 2]))

        # Test __imul__
        test_array = Array(np.array([1, 2, 3]))
        test_array *= 2
        np.testing.assert_array_equal(test_array.value, np.array([2, 4, 6]))

        # Test __itruediv__
        test_array = Array(np.array([2, 4, 6]))
        test_array /= 2
        np.testing.assert_array_equal(test_array.value, np.array([1, 2, 3]))

    def test_iterator(self):
        # Test __iter__
        values = [x for x in self.array_impl_np]
        expected_values = [x for x in self.np_array]
        self.assertEqual(values, expected_values)

    def test_misc_methods(self):
        # Test fill method
        test_array = Array(np.array([1, 2, 3]))
        test_array.fill(5)
        np.testing.assert_array_equal(test_array.value, np.array([5, 5, 5]))

        # Test flatten method
        flattened = self.array_impl_2d.flatten()
        np.testing.assert_array_equal(flattened, self.ones_2d.flatten())

        # Test item method
        scalar_impl = Array(np.array(5.0))
        self.assertEqual(scalar_impl.item(), 5.0)

        # Test view method
        view_array = Array(np.array([1, 2, 3], dtype=np.int32))
        int_view = view_array.view(np.int32)
        self.assertEqual(int_view.dtype, np.int32)


class TestArray2(unittest.TestCase):
    def setUp(self):
        # Basic arrays for testing
        self.np_array = np.array([1, 2, 3])
        self.jax_array = jnp.array([1, 2, 3])
        # Create ArrayImpl instances
        self.array_impl_np = Array(self.np_array)
        self.array_impl_jax = Array(self.jax_array)
        # More complex arrays for advanced testing
        self.ones_2d = np.ones((2, 3))
        self.array_impl_2d = Array(self.ones_2d)
        # Scalar array
        self.scalar = np.array(5.0)
        self.array_impl_scalar = Array(self.scalar)

    def test_basic_properties(self):
        # Test dtype property
        self.assertEqual(self.array_impl_np.dtype, self.np_array.dtype)
        self.assertEqual(self.array_impl_jax.dtype, self.jax_array.dtype)

        # Test shape property
        self.assertEqual(self.array_impl_np.shape, self.np_array.shape)
        self.assertEqual(self.array_impl_jax.shape, self.jax_array.shape)

        # Test ndim property
        self.assertEqual(self.array_impl_np.ndim, self.np_array.ndim)
        self.assertEqual(self.array_impl_2d.ndim, self.ones_2d.ndim)

        # Test real property (should work for real arrays)
        np.testing.assert_array_equal(self.array_impl_np.real, self.np_array.real)

    def test_unary_operations(self):
        # Test __neg__
        np.testing.assert_array_equal(-self.array_impl_np, -self.np_array)

        # Test __pos__
        np.testing.assert_array_equal(+self.array_impl_np, +self.np_array)

        # Test __abs__
        neg_array = Array(np.array([-1, 2, -3]))
        np.testing.assert_array_equal(abs(neg_array), np.abs(neg_array.value))

        # Test __invert__ (bitwise NOT)
        uint_array = Array(np.array([1, 2, 3], dtype=np.uint8))
        np.testing.assert_array_equal(~uint_array, ~uint_array.value)

    def test_binary_operations(self):
        # Test __add__ and __radd__
        np.testing.assert_array_equal(self.array_impl_np + 2, self.np_array + 2)
        np.testing.assert_array_equal(2 + self.array_impl_np, 2 + self.np_array)

        # Test __sub__ and __rsub__
        np.testing.assert_array_equal(self.array_impl_np - 1, self.np_array - 1)
        np.testing.assert_array_equal(5 - self.array_impl_np, 5 - self.np_array)

        # Test __mul__ and __rmul__
        np.testing.assert_array_equal(self.array_impl_np * 2, self.np_array * 2)
        np.testing.assert_array_equal(2 * self.array_impl_np, 2 * self.np_array)

        # Test __truediv__ and __rtruediv__
        np.testing.assert_array_almost_equal(self.array_impl_np / 2, self.np_array / 2)
        np.testing.assert_array_almost_equal(6 / self.array_impl_np, 6 / self.np_array)

        # Test __pow__ and __rpow__
        np.testing.assert_array_equal(self.array_impl_np ** 2, self.np_array ** 2)
        np.testing.assert_array_equal(2 ** self.array_impl_np, 2 ** self.np_array)

        # Test __matmul__
        a = Array(np.array([[1, 2], [3, 4]]))
        b = Array(np.array([[5, 6], [7, 8]]))
        np.testing.assert_array_equal(a @ b, a.value @ b.value)

    def test_inplace_operations(self):
        # Test __iadd__
        test_array = Array(np.array([1, 2, 3]))
        test_array += 1
        np.testing.assert_array_equal(test_array.value, np.array([2, 3, 4]))

        # Test __isub__
        test_array = Array(np.array([1, 2, 3]))
        test_array -= 1
        np.testing.assert_array_equal(test_array.value, np.array([0, 1, 2]))

        # Test __imul__
        test_array = Array(np.array([1, 2, 3]))
        test_array *= 2
        np.testing.assert_array_equal(test_array.value, np.array([2, 4, 6]))

        # Test __itruediv__
        test_array = Array(np.array([2, 4, 6]))
        test_array /= 2
        np.testing.assert_array_equal(test_array.value, np.array([1, 2, 3]))

    def test_iterator(self):
        # Test __iter__
        values = [x for x in self.array_impl_np]
        expected_values = [x for x in self.np_array]
        self.assertEqual(values, expected_values)

    def test_misc_methods(self):
        # Test fill method
        test_array = Array(np.array([1, 2, 3]))
        test_array.fill(5)
        np.testing.assert_array_equal(test_array.value, np.array([5, 5, 5]))

        # Test flatten method
        flattened = self.array_impl_2d.flatten()
        np.testing.assert_array_equal(flattened, self.ones_2d.flatten())

        # Test item method
        scalar_impl = Array(np.array(5.0))
        self.assertEqual(scalar_impl.item(), 5.0)

        # Test view method
        view_array = Array(np.array([1, 2, 3], dtype=np.int32))
        int_view = view_array.view(np.int32)
        self.assertEqual(int_view.dtype, np.int32)

    def test_advanced_indexing(self):
        # Test basic indexing
        arr = Array(np.array([1, 2, 3, 4, 5]))
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[-1], 5)
        np.testing.assert_array_equal(arr[1:4], np.array([2, 3, 4]))

        # Test boolean indexing
        bool_mask = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(arr[bool_mask], np.array([1, 3, 5]))

        # Test integer array indexing
        idx = np.array([0, 2, 4])
        np.testing.assert_array_equal(arr[idx], np.array([1, 3, 5]))

        # Test assignment via indexing
        arr = Array(np.array([1, 2, 3, 4, 5]))
        arr[1:4] = 10
        np.testing.assert_array_equal(arr.value, np.array([1, 10, 10, 10, 5]))

    def test_comparison_operators(self):
        arr = Array(np.array([1, 2, 3]))

        # Test ==, !=, <, <=, >, >=
        np.testing.assert_array_equal(arr == 2, np.array([False, True, False]))
        np.testing.assert_array_equal(arr != 2, np.array([True, False, True]))
        np.testing.assert_array_equal(arr < 2, np.array([True, False, False]))
        np.testing.assert_array_equal(arr <= 2, np.array([True, True, False]))
        np.testing.assert_array_equal(arr > 2, np.array([False, False, True]))
        np.testing.assert_array_equal(arr >= 2, np.array([False, True, True]))

        # Test array vs array comparisons
        arr2 = Array(np.array([2, 2, 2]))
        np.testing.assert_array_equal(arr == arr2, np.array([False, True, False]))

    def test_jax_integration(self):
        # Test JAX transformations with Array
        arr = Array(jnp.array([1.0, 2.0, 3.0]))

        # Test jit compilation
        @jax.jit
        def square(x):
            return x * x

        result = square(arr)
        self.assertIsInstance(result, jax.Array)
        np.testing.assert_array_equal(result, jnp.array([1.0, 4.0, 9.0]))

        # Test grad with Array
        @jax.grad
        def sum_squares(x):
            return jnp.sum(x * x)

        grad_result = sum_squares(arr)
        self.assertIsInstance(grad_result, jax.Array)
        np.testing.assert_array_equal(grad_result, jnp.array([2.0, 4.0, 6.0]))

    def test_edge_cases(self):
        # Test empty array
        empty_arr = Array(np.array([]))
        self.assertEqual(empty_arr.shape, (0,))
        self.assertEqual(empty_arr.ndim, 1)

        # Test very large array
        large_arr = Array(np.ones((1000,)))
        self.assertEqual(large_arr.shape, (1000,))

        # Test high-dimensional array
        high_dim = Array(np.zeros((2, 3, 4, 5)))
        self.assertEqual(high_dim.ndim, 4)
        self.assertEqual(high_dim.shape, (2, 3, 4, 5))

    def test_error_handling(self):
        # Test setting value with different tree structure
        arr = Array(np.array([1, 2, 3]))

        # Test invalid operations
        with self.assertRaises(TypeError):
            # Attempt to add incompatible types
            arr + "string"

    def test_copy_and_clone(self):
        # Test copy method
        arr = Array(np.array([1, 2, 3]))
        arr_copy = arr.copy()

        # Check they have the same value but are different objects
        np.testing.assert_array_equal(arr.value, arr_copy.value)
        self.assertIsNot(arr, arr_copy)

        # Modify the copy and check original is unchanged
        arr_copy.value = np.array([4, 5, 6])
        np.testing.assert_array_equal(arr.value, np.array([1, 2, 3]))

        # Test replace method
        arr_replaced = arr.replace(value=np.array([7, 8, 9]))
        np.testing.assert_array_equal(arr_replaced.value, np.array([7, 8, 9]))
        # Original should remain unchanged
        np.testing.assert_array_equal(arr.value, np.array([1, 2, 3]))

    def test_state_integration(self):
        # Test State methods
        arr = Array(np.array([1, 2, 3]), name="test_array")
        self.assertEqual(arr.name, "test_array")

        # Test numel method
        self.assertEqual(arr.numel(), 3)
        self.assertEqual(self.array_impl_2d.numel(), 6)  # 2x3 array

        # Test stack level operations
        original_level = arr.stack_level
        arr.increase_stack_level()
        self.assertEqual(arr.stack_level, original_level + 1)
        arr.decrease_stack_level()
        self.assertEqual(arr.stack_level, original_level)

    def test_brainstate_integration(self):
        # Test value_call method
        arr = Array(np.array([1, 2, 3]))
        result = arr.value_call(lambda x: x * 2)
        np.testing.assert_array_equal(result, np.array([2, 4, 6]))

        # Test with brainstate functions
        import brainstate as bs

        # Test check_state_value_tree context manager
        with bs.check_state_value_tree():
            # This should work since tree structure is the same
            arr.value = np.array([4, 5, 6])

            # This should fail
            with self.assertRaises(ValueError):
                arr.value = (np.array([1, 2, 3]),)


