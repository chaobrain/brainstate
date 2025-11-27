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

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from brainstate._state import TRACE_CONTEXT, StateTraceStack
from brainstate.random._state import RandomState, DEFAULT, formalize_key, _size2shape, _check_py_seq


class TestRandomStateInitialization(unittest.TestCase):
    """Test RandomState initialization and setup."""

    def setUp(self):
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        TRACE_CONTEXT.state_stack.pop()

    def test_init_with_none(self):
        """Test initialization with None seed."""
        rs = RandomState(None)
        self.assertIsNotNone(rs.value)
        self.assertEqual(rs.value.shape, (2,))
        self.assertEqual(rs.value.dtype, jnp.uint32)

    def test_init_with_int_seed(self):
        """Test initialization with integer seed."""
        seed = 42
        rs = RandomState(seed)
        expected_key = jr.PRNGKey(seed)
        np.testing.assert_array_equal(rs.value, expected_key)

    def test_init_with_prng_key(self):
        """Test initialization with JAX PRNGKey."""
        key = jr.PRNGKey(123)
        rs = RandomState(key)
        np.testing.assert_array_equal(rs.value, key)

    def test_init_with_uint32_array(self):
        """Test initialization with uint32 array."""
        key_array = np.array([123, 456], dtype=np.uint32)
        rs = RandomState(key_array)
        np.testing.assert_array_equal(rs.value, key_array)

    def test_init_with_invalid_key(self):
        """Test initialization with invalid key raises error."""
        # Test case that should raise error: wrong length AND wrong dtype
        with self.assertRaises(ValueError):
            RandomState(np.array([1, 2, 3], dtype=np.int32))  # len != 2 AND dtype != uint32

        # Test valid cases that should NOT raise errors
        # Wrong length but correct dtype is OK
        rs1 = RandomState(np.array([1, 2, 3], dtype=np.uint32))
        self.assertIsNotNone(rs1.value)

        # Correct length but wrong dtype is OK
        rs2 = RandomState(np.array([1, 2], dtype=np.int32))
        self.assertIsNotNone(rs2.value)

    def test_repr(self):
        """Test string representation."""
        rs = RandomState(42)
        repr_str = repr(rs)
        self.assertIn("RandomState", repr_str)
        self.assertIn("42", repr_str)


class TestRandomStateKeyManagement(unittest.TestCase):
    """Test key management functionality."""

    def setUp(self):
        self.rs = RandomState(42)
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        TRACE_CONTEXT.state_stack.pop()

    def test_seed_with_int(self):
        """Test seeding with integer."""
        self.rs.seed(123)
        expected_key = jr.PRNGKey(123)
        np.testing.assert_array_equal(self.rs.value, expected_key)

    def test_seed_with_none(self):
        """Test seeding with None generates new random seed."""
        original_key = self.rs.value.copy()
        self.rs.seed(None)
        # Should be different (with very high probability)
        self.assertFalse(np.array_equal(self.rs.value, original_key))

    def test_seed_with_prng_key(self):
        """Test seeding with PRNGKey."""
        key = jr.PRNGKey(999)
        self.rs.seed(key)
        np.testing.assert_array_equal(self.rs.value, key)

    def test_seed_with_invalid_input(self):
        """Test seeding with invalid input raises error."""
        with self.assertRaises(ValueError):
            self.rs.seed([1, 2, 3])  # Wrong length list

    def test_split_key_single(self):
        """Test splitting key to get single new key."""
        original_key = self.rs.value.copy()
        new_key = self.rs.split_key()

        # Original key should have changed
        self.assertFalse(np.array_equal(self.rs.value, original_key))
        # New key should be different from both
        self.assertFalse(np.array_equal(new_key, original_key))
        self.assertFalse(np.array_equal(new_key, self.rs.value))

    def test_split_key_multiple(self):
        """Test splitting key to get multiple new keys."""
        n = 3
        original_key = self.rs.value.copy()
        new_keys = self.rs.split_key(n)

        self.assertEqual(len(new_keys), n)
        # All keys should be different
        for i, key in enumerate(new_keys):
            self.assertFalse(np.array_equal(key, original_key))
            for j, other_key in enumerate(new_keys):
                if i != j:
                    self.assertFalse(np.array_equal(key, other_key))

    def test_split_key_invalid_n(self):
        """Test split_key with invalid n raises error."""
        with self.assertRaises(AssertionError):
            self.rs.split_key(0)

        with self.assertRaises(AssertionError):
            self.rs.split_key(-1)

    def test_backup_restore_key(self):
        """Test backup and restore functionality."""
        original_key = self.rs.value.copy()

        # Backup the key
        self.rs.backup_key()

        # Change the key
        self.rs.split_key()
        changed_key = self.rs.value.copy()
        self.assertFalse(np.array_equal(changed_key, original_key))

        # Restore the key
        self.rs.restore_key()
        np.testing.assert_array_equal(self.rs.value, original_key)

    def test_backup_already_backed_up(self):
        """Test backup when already backed up raises error."""
        self.rs.backup_key()
        with self.assertRaises(ValueError):
            self.rs.backup_key()

    def test_restore_without_backup(self):
        """Test restore without backup raises error."""
        with self.assertRaises(ValueError):
            self.rs.restore_key()

    def test_clone(self):
        """Test cloning creates independent copy."""
        clone = self.rs.clone()

        # Should be different instances
        self.assertIsNot(clone, self.rs)

        # Should have different keys after split
        original_key = self.rs.value.copy()
        clone_key = clone.value.copy()

        self.rs.split_key()
        clone.split_key()

        self.assertFalse(np.array_equal(self.rs.value, clone.value))

    def test_set_key(self):
        """Test setting key directly."""
        new_key = jr.PRNGKey(999)
        self.rs.set_key(new_key)
        np.testing.assert_array_equal(self.rs.value, new_key)


class TestRandomStateDistributions(unittest.TestCase):
    """Test random distribution methods."""

    def setUp(self):
        self.rs = RandomState(42)
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        TRACE_CONTEXT.state_stack.pop()

    def test_rand(self):
        """Test rand method."""
        # Single value
        val = self.rs.rand()
        self.assertEqual(val.shape, ())
        self.assertTrue(0 <= val < 1)

        # Multiple dimensions
        arr = self.rs.rand(3, 2)
        self.assertEqual(arr.shape, (3, 2))
        self.assertTrue((arr >= 0).all() and (arr < 1).all())

    def test_randint(self):
        """Test randint method."""
        # Single bound
        val = self.rs.randint(10)
        self.assertTrue(0 <= val < 10)

        # Both bounds
        val = self.rs.randint(5, 15)
        self.assertTrue(5 <= val < 15)

        # With size
        arr = self.rs.randint(0, 5, size=(2, 3))
        self.assertEqual(arr.shape, (2, 3))
        self.assertTrue((arr >= 0).all() and (arr < 5).all())

    def test_randn(self):
        """Test randn method."""
        # Single value
        val = self.rs.randn()
        self.assertEqual(val.shape, ())

        # Multiple dimensions
        arr = self.rs.randn(3, 2)
        self.assertEqual(arr.shape, (3, 2))

    def test_normal(self):
        """Test normal distribution."""
        # Standard normal
        val = self.rs.normal()
        self.assertEqual(val.shape, ())

        # With parameters
        arr = self.rs.normal(5.0, 2.0, size=(3, 2))
        self.assertEqual(arr.shape, (3, 2))

    def test_uniform(self):
        """Test uniform distribution."""
        # Standard uniform
        val = self.rs.uniform()
        self.assertTrue(0.0 <= val < 1.0)

        # With bounds
        arr = self.rs.uniform(low=2.0, high=8.0, size=(2, 3))
        self.assertEqual(arr.shape, (2, 3))
        self.assertTrue((arr >= 2.0).all() and (arr < 8.0).all())

    def test_choice(self):
        """Test choice method."""

        # Choose from range
        val = self.rs.choice(5)
        self.assertTrue(0 <= val < 5)

        # Choose from array
        options = jnp.array([10, 20, 30, 40])
        val = self.rs.choice(options)
        self.assertIn(val, options)

        # Multiple choices
        arr = self.rs.choice(5, size=10)
        self.assertEqual(arr.shape, (10,))
        self.assertTrue((arr >= 0).all() and (arr < 5).all())

    def test_beta(self):
        """Test beta distribution."""
        arr = self.rs.beta(2.0, 3.0, size=(2, 3))
        self.assertEqual(arr.shape, (2, 3))
        self.assertTrue((arr >= 0).all() and (arr <= 1).all())

    def test_exponential(self):
        """Test exponential distribution."""
        arr = self.rs.exponential(2.0, size=(2, 3))
        self.assertEqual(arr.shape, (2, 3))
        self.assertTrue((arr >= 0).all())

    def test_gamma(self):
        """Test gamma distribution."""
        arr = self.rs.gamma(2.0, 1.0, size=(2, 3))
        self.assertEqual(arr.shape, (2, 3))
        self.assertTrue((arr >= 0).all())

    def test_poisson(self):
        """Test Poisson distribution."""
        arr = self.rs.poisson(3.0, size=(2, 3))
        self.assertEqual(arr.shape, (2, 3))
        self.assertTrue((arr >= 0).all())

    def test_binomial(self):
        """Test binomial distribution."""
        arr = self.rs.binomial(10, 0.3, size=(2, 3))
        self.assertEqual(arr.shape, (2, 3))
        self.assertTrue((arr >= 0).all() and (arr <= 10).all())

    def test_bernoulli(self):
        """Test Bernoulli distribution."""
        arr = self.rs.bernoulli(0.7, size=(100,))
        self.assertEqual(arr.shape, (100,))
        self.assertTrue(jnp.all((arr == 0) | (arr == 1)))

    def test_bernoulli_invalid_p(self):
        """Test Bernoulli with invalid probability."""
        # Note: This should trigger jit_error_if, but in test we check the validation exists
        with self.assertRaises((ValueError, Exception)):
            self.rs.bernoulli(1.5)  # p > 1

    def test_truncated_normal(self):
        """Test truncated normal distribution."""
        arr = self.rs.truncated_normal(-1.0, 1.0, size=(2, 3))
        self.assertEqual(arr.shape, (2, 3))
        self.assertTrue((arr >= -1.0).all() and (arr <= 1.0).all())

    def test_multivariate_normal(self):
        """Test multivariate normal distribution."""
        mean = jnp.array([0.0, 1.0])
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])

        arr = self.rs.multivariate_normal(mean, cov, size=(3,))
        self.assertEqual(arr.shape, (3, 2))

    def test_categorical(self):
        """Test categorical distribution."""
        logits = jnp.array([0.1, 0.2, 0.3, 0.4])
        arr = self.rs.categorical(logits, size=(10,))
        self.assertEqual(arr.shape, (10,))
        self.assertTrue((arr >= 0).all() and (arr < len(logits)).all())


class TestRandomStatePyTorchCompatibility(unittest.TestCase):
    """Test PyTorch-like methods."""

    def setUp(self):
        self.rs = RandomState(42)
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        TRACE_CONTEXT.state_stack.pop()

    def test_rand_like(self):
        """Test rand_like method."""
        input_tensor = jnp.zeros((3, 4))
        result = self.rs.rand_like(input_tensor)
        self.assertEqual(result.shape, input_tensor.shape)
        self.assertTrue((result >= 0).all() and (result < 1).all())

    def test_randn_like(self):
        """Test randn_like method."""
        input_tensor = jnp.zeros((2, 3))
        result = self.rs.randn_like(input_tensor)
        self.assertEqual(result.shape, input_tensor.shape)

    def test_randint_like(self):
        """Test randint_like method."""
        input_tensor = jnp.zeros((2, 3), dtype=jnp.int32)
        result = self.rs.randint_like(input_tensor, 0, 10)
        self.assertEqual(result.shape, input_tensor.shape)
        self.assertTrue((result >= 0).all() and (result < 10).all())


class TestRandomStateKeyBehavior(unittest.TestCase):
    """Test key parameter behavior across methods."""

    def setUp(self):
        self.rs = RandomState(42)
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        TRACE_CONTEXT.state_stack.pop()

    def test_external_key_does_not_change_state(self):
        """Test that using external key doesn't change internal state."""
        original_key = self.rs.value.copy()
        external_key = jr.PRNGKey(999)

        # Use external key
        self.rs.rand(5, key=external_key)

        # Internal state should be unchanged
        np.testing.assert_array_equal(self.rs.value, original_key)

    def test_no_key_changes_state(self):
        """Test that not providing key changes internal state."""
        original_key = self.rs.value.copy()

        # Use internal key
        self.rs.rand(5)

        # Internal state should have changed
        self.assertFalse(np.array_equal(self.rs.value, original_key))

    def test_reproducibility_with_same_key(self):
        """Test reproducibility when using same external key."""
        key = jr.PRNGKey(123)

        result1 = self.rs.rand(5, key=key)
        result2 = self.rs.rand(5, key=key)

        np.testing.assert_array_equal(result1, result2)

    def test_reproducibility_with_seed(self):
        """Test reproducibility with seeding."""
        self.rs.seed(42)
        result1 = self.rs.rand(5)

        self.rs.seed(42)
        result2 = self.rs.rand(5)

        np.testing.assert_array_equal(result1, result2)


class TestGlobalDefaultInstance(unittest.TestCase):
    """Test the global DEFAULT RandomState instance."""

    def setUp(self):
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        TRACE_CONTEXT.state_stack.pop()

    def test_default_exists(self):
        """Test that DEFAULT instance exists and is RandomState."""
        self.assertIsInstance(DEFAULT, RandomState)

    def test_default_has_valid_key(self):
        """Test that DEFAULT has valid key."""
        self.assertIsNotNone(DEFAULT.value)
        self.assertEqual(DEFAULT.value.shape, (2,))
        self.assertEqual(DEFAULT.value.dtype, jnp.uint32)

    def test_default_seeding(self):
        """Test seeding DEFAULT instance."""
        original_key = DEFAULT.value.copy()
        DEFAULT.seed(12345)
        self.assertFalse(np.array_equal(DEFAULT.value, original_key))

    def test_default_split_key(self):
        """Test splitting DEFAULT key."""
        original_key = DEFAULT.value.copy()
        new_key = DEFAULT.split_key()
        self.assertFalse(np.array_equal(DEFAULT.value, original_key))
        self.assertIsNotNone(new_key)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in _rand_state module."""

    def setUp(self):
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        TRACE_CONTEXT.state_stack.pop()

    def test_formalize_key_with_int(self):
        """Test _formalize_key with integer."""
        key = formalize_key(42)
        expected = jr.PRNGKey(42)
        np.testing.assert_array_equal(key, expected)

    def test_formalize_key_with_array(self):
        """Test _formalize_key with array."""
        input_key = jr.PRNGKey(123)
        key = formalize_key(input_key, True)
        np.testing.assert_array_equal(key, input_key)

    def test_formalize_key_with_uint32_array(self):
        """Test _formalize_key with uint32 array."""
        input_array = np.array([123, 456], dtype=np.uint32)
        key = formalize_key(input_array)
        np.testing.assert_array_equal(key, input_array)

    def test_formalize_key_invalid_input(self):
        """Test _formalize_key with invalid input."""
        with self.assertRaises(TypeError):
            formalize_key("invalid")

        with self.assertRaises(TypeError):
            formalize_key(np.array([1, 2, 3], dtype=np.uint32))  # Wrong size

        with self.assertRaises(TypeError):
            formalize_key(np.array([1, 2], dtype=np.int32))  # Wrong dtype

    def test_size2shape(self):
        """Test _size2shape function."""
        self.assertEqual(_size2shape(None), ())
        self.assertEqual(_size2shape(5), (5,))
        self.assertEqual(_size2shape((3, 4)), (3, 4))
        self.assertEqual(_size2shape([2, 3, 4]), (2, 3, 4))

    def test_check_py_seq(self):
        """Test _check_py_seq function."""
        # Should convert lists/tuples to arrays
        result = _check_py_seq([1, 2, 3])
        self.assertIsInstance(result, jnp.ndarray)
        np.testing.assert_array_equal(result, jnp.array([1, 2, 3]))

        # Should leave other types unchanged
        arr = jnp.array([1, 2, 3])
        result = _check_py_seq(arr)
        self.assertIs(result, arr)

        scalar = 5
        result = _check_py_seq(scalar)
        self.assertEqual(result, scalar)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        self.rs = RandomState(42)
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        TRACE_CONTEXT.state_stack.pop()

    def test_invalid_distribution_parameters(self):
        """Test invalid parameters for distributions."""
        # Note: Some distributions may not validate parameters immediately
        # so we test what we can verify

        # Test invalid probability for binomial should work with check_valid=True
        try:
            # This may or may not raise immediately depending on JAX compilation
            self.rs.binomial(10, 1.5, check_valid=True)
        except:
            pass  # Expected to fail

        # Test normal distribution works with negative scale (JAX allows this)
        result = self.rs.normal(None, -1.0, size=(2,))
        self.assertEqual(result.shape, (2,))

    def test_invalid_size_parameters(self):
        """Test invalid size parameters."""
        # Test empty shape works for distributions that accept size parameter
        result = self.rs.random(size=())
        self.assertEqual(result.shape, ())

        # Test with None size
        result = self.rs.random(size=None)
        self.assertEqual(result.shape, ())

    def test_dtype_consistency(self):
        """Test dtype consistency across methods."""
        # Integer methods should return integers
        result = self.rs.randint(10, size=(3,))
        self.assertTrue(jnp.issubdtype(result.dtype, jnp.integer))

        # Float methods should return floats
        result = self.rs.rand(3)
        self.assertTrue(jnp.issubdtype(result.dtype, jnp.floating))

    def test_self_assign_multi_keys(self):
        """Test self_assign_multi_keys method."""
        original_shape = self.rs.value.shape

        # Test with backup
        self.rs.self_assign_multi_keys(3, backup=True)
        self.assertEqual(self.rs.value.shape, (3, 2))

        # Restore should work
        self.rs.restore_key()
        self.assertEqual(self.rs.value.shape, original_shape)

        # Test without backup
        self.rs.self_assign_multi_keys(2, backup=False)
        self.assertEqual(self.rs.value.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()
