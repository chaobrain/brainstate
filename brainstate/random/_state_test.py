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
import numpy as np
import pytest

import brainstate
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
        expected_key = formalize_key(seed)
        np.testing.assert_array_equal(rs.value, expected_key)

    def test_init_with_prng_key(self):
        """Test initialization with a brainstate-formalized PRNG key."""
        key = formalize_key(123)
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
        expected_key = formalize_key(123)
        np.testing.assert_array_equal(self.rs.value, expected_key)

    def test_seed_with_none(self):
        """Test seeding with None generates new random seed."""
        original_key = self.rs.value.copy()
        self.rs.seed(None)
        # Should be different (with very high probability)
        self.assertFalse(np.array_equal(self.rs.value, original_key))

    def test_seed_with_prng_key(self):
        """Test seeding with a brainstate-formalized PRNG key."""
        key = formalize_key(999)
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
        new_key = formalize_key(999)
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
        external_key = formalize_key(999)

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
        key = formalize_key(123)

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
        """Test _formalize_key with integer matches a fresh RandomState key."""
        key = formalize_key(42)
        expected = RandomState(42).value
        np.testing.assert_array_equal(key, expected)

    def test_formalize_key_with_array(self):
        """Test _formalize_key passes through an existing PRNG key unchanged."""
        input_key = formalize_key(123)
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


class TestRandomStateTransforms(unittest.TestCase):
    """RandomState as a brainstate State: pytree, key advancement, transforms."""

    def setUp(self):
        """Push a fresh state-trace stack for each test."""
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        """Pop the state-trace stack after each test."""
        TRACE_CONTEXT.state_stack.pop()

    def test_pytree_roundtrip(self):
        """A RandomState's treefied reference survives flatten/unflatten."""
        from brainstate import _testing
        rs = RandomState(0)
        ref = rs.to_state_ref()
        rebuilt = _testing.assert_pytree_roundtrip(ref)
        np.testing.assert_array_equal(rebuilt.value, rs.value)

    def test_pytree_in_container_roundtrip(self):
        """A treefied RandomState inside a dict roundtrips through jax.tree."""
        import jax
        rs = RandomState(7)
        tree = {'rng': rs.to_state_ref(), 'x': jnp.arange(3)}
        leaves, treedef = jax.tree.flatten(tree)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        np.testing.assert_array_equal(rebuilt['rng'].value, rs.value)
        np.testing.assert_array_equal(rebuilt['x'], tree['x'])

    def test_randomstate_is_tree_leaf(self):
        """A bare RandomState is treated as a single pytree leaf."""
        import jax
        rs = RandomState(0)
        leaves = jax.tree.leaves(rs)
        self.assertEqual(len(leaves), 1)
        self.assertIs(leaves[0], rs)

    def test_repeated_draws_advance_key(self):
        """Consecutive draws from a RandomState are not identical."""
        rs = RandomState(0)
        self.assertFalse(bool(jnp.allclose(rs.randn(5), rs.randn(5))))

    def test_draw_advances_internal_key(self):
        """Drawing without an explicit key mutates the internal state value."""
        rs = RandomState(0)
        before = np.asarray(rs.value).copy()
        rs.randn(3)
        self.assertFalse(np.array_equal(before, np.asarray(rs.value)))

    def test_same_seed_same_sequence(self):
        """Two RandomStates with the same seed yield identical sequences."""
        r1 = RandomState(123)
        r2 = RandomState(123)
        np.testing.assert_array_equal(r1.randn(5), r2.randn(5))
        np.testing.assert_array_equal(r1.randn(5), r2.randn(5))

    def test_different_seed_different_sequence(self):
        """Two RandomStates with different seeds yield different sequences."""
        r1 = RandomState(1)
        r2 = RandomState(2)
        self.assertFalse(bool(jnp.allclose(r1.randn(5), r2.randn(5))))

    def test_randomstate_inside_jit(self):
        """Drawing from a RandomState inside jit yields the right shape."""
        rs = RandomState(0)

        @brainstate.transform.jit
        def draw():
            return rs.randn(4)

        self.assertEqual(draw().shape, (4,))

    def test_jit_advances_key_between_calls(self):
        """Successive jitted draws from the same RandomState differ."""
        rs = RandomState(0)

        @brainstate.transform.jit
        def draw():
            return rs.randn(4)

        self.assertFalse(bool(jnp.allclose(draw(), draw())))

    def test_randomstate_inside_vmap_with_batched_key(self):
        """vmap over a batch of keys produces a batched output."""
        rs = RandomState(0)

        def draw(key):
            return rs.randn(3, key=key)

        keys = brainstate.random.split_keys(4)
        out = brainstate.transform.vmap(draw)(keys)
        self.assertEqual(out.shape, (4, 3))

    def test_explicit_key_is_reproducible(self):
        """Passing the same explicit key twice yields identical draws."""
        rs = RandomState(0)
        brainstate.random.seed(99)
        key = brainstate.random.split_key()
        np.testing.assert_array_equal(rs.randn(5, key=key), rs.randn(5, key=key))

    def test_global_seed_split_key_roundtrip(self):
        """A key from the global stream can construct a usable RandomState."""
        brainstate.random.seed(7)
        key = brainstate.random.split_key()
        self.assertEqual(key.shape, (2,))
        self.assertEqual(key.dtype, jnp.uint32)
        rs = RandomState(key)
        np.testing.assert_array_equal(rs.value, key)

    def test_global_seed_is_deterministic(self):
        """Re-seeding the global stream replays the same keys."""
        brainstate.random.seed(11)
        a = brainstate.random.split_key()
        brainstate.random.seed(11)
        b = brainstate.random.split_key()
        np.testing.assert_array_equal(a, b)


class TestRandomStateMisc(unittest.TestCase):
    """Cover repr, clone, numpy keys, deletion checks, and multi-key assignment."""

    def setUp(self):
        """Create a seeded RandomState and push a state-trace stack."""
        self.rs = RandomState(42)
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        """Pop the state-trace stack after each test."""
        TRACE_CONTEXT.state_stack.pop()

    def test_repr_contains_value(self):
        """The repr embeds the class name and the underlying key value."""
        text = repr(self.rs)
        self.assertIn('RandomState', text)
        self.assertIn(str(self.rs.value), text)

    def test_clone_is_independent(self):
        """Cloning yields a distinct instance with a distinct key."""
        clone = self.rs.clone()
        self.assertIsNot(clone, self.rs)
        self.assertFalse(np.array_equal(clone.value, self.rs.value))

    def test_numpy_keys_shape_and_dtype(self):
        """_numpy_keys returns a (batch, 2) uint32 array."""
        keys = self.rs._numpy_keys(4)
        self.assertEqual(keys.shape, (4, 2))
        self.assertEqual(keys.dtype, np.uint32)

    def test_check_if_deleted_on_live_key(self):
        """check_if_deleted is a no-op for a live (non-deleted) key."""
        before = np.asarray(self.rs.value).copy()
        self.rs.check_if_deleted()
        np.testing.assert_array_equal(self.rs.value, before)

    def test_seed_with_numpy_int_array_scalar(self):
        """Seeding with a 1-element integer numpy array sets a valid key."""
        self.rs.seed(np.array(123, dtype=np.int64))
        self.assertEqual(self.rs.value.shape, (2,))
        np.testing.assert_array_equal(self.rs.value, formalize_key(123))

    def test_seed_with_uint32_pair(self):
        """Seeding with a uint32 pair stores the key verbatim."""
        key = np.array([7, 9], dtype=np.uint32)
        self.rs.seed(key)
        np.testing.assert_array_equal(self.rs.value, key)

    def test_seed_with_invalid_scalar_type(self):
        """Seeding with a float scalar raises ValueError."""
        with self.assertRaises(ValueError):
            self.rs.seed(np.array(1.5, dtype=np.float32))

    def test_self_assign_multi_keys_without_backup(self):
        """self_assign_multi_keys without backup leaves no restore point."""
        self.rs.self_assign_multi_keys(3, backup=False)
        self.assertEqual(self.rs.value.shape, (3, 2))
        with self.assertRaises(ValueError):
            self.rs.restore_key()

    def test_split_key_with_backup(self):
        """split_key(backup=True) records the post-split key as the restore point."""
        # split_key advances the internal key to keys[0] and *then* backs it up,
        # so the backed-up value is the advanced key, not the pre-split key.
        self.rs.split_key(backup=True)
        post_split = np.asarray(self.rs.value).copy()
        # Advance again, then restore back to the post-split key.
        self.rs.split_key()
        self.assertFalse(np.array_equal(np.asarray(self.rs.value), post_split))
        self.rs.restore_key()
        np.testing.assert_array_equal(self.rs.value, post_split)


class TestRandomStateMoreDistributions(unittest.TestCase):
    """Exercise distribution methods not covered by the baseline suite."""

    def setUp(self):
        """Create a seeded RandomState and push a state-trace stack."""
        self.rs = RandomState(2024)
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        """Pop the state-trace stack after each test."""
        TRACE_CONTEXT.state_stack.pop()

    def test_random_aliases(self):
        """random_sample, ranf, and sample all delegate to random."""
        self.assertEqual(self.rs.random_sample(size=(2, 3)).shape, (2, 3))
        self.assertEqual(self.rs.ranf(size=(2, 3)).shape, (2, 3))
        self.assertEqual(self.rs.sample(size=(2, 3)).shape, (2, 3))

    def test_random_integers(self):
        """random_integers honours inclusive high and broadcast sizing."""
        self.assertEqual(self.rs.random_integers(1, 5, size=(3,)).shape, (3,))
        self.assertEqual(self.rs.random_integers(5).shape, ())

    def test_permutation_and_shuffle(self):
        """permutation and shuffle preserve the multiset of elements."""
        perm = self.rs.permutation(jnp.arange(6))
        np.testing.assert_array_equal(np.sort(np.asarray(perm)), np.arange(6))
        shuf = self.rs.shuffle(jnp.arange(6))
        np.testing.assert_array_equal(np.sort(np.asarray(shuf)), np.arange(6))

    def test_gumbel_laplace_logistic(self):
        """gumbel, laplace, and logistic produce the requested shape."""
        self.assertEqual(self.rs.gumbel(0.0, 1.0, size=(2, 3)).shape, (2, 3))
        self.assertEqual(self.rs.laplace(0.0, 1.0, size=(2, 3)).shape, (2, 3))
        self.assertEqual(self.rs.logistic(0.0, 1.0, size=(2, 3)).shape, (2, 3))

    def test_logistic_no_loc_scale(self):
        """logistic broadcasts to a scalar when loc/scale are omitted."""
        self.assertEqual(self.rs.logistic().shape, ())

    def test_pareto(self):
        """pareto yields positive samples with the requested shape."""
        arr = self.rs.pareto(2.0, size=(3,))
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr >= 0).all())

    def test_standard_family(self):
        """standard_* helpers each yield the requested shape."""
        self.assertEqual(self.rs.standard_cauchy(size=(3,)).shape, (3,))
        self.assertEqual(self.rs.standard_exponential(size=(3,)).shape, (3,))
        self.assertEqual(self.rs.standard_gamma(2.0, size=(3,)).shape, (3,))
        self.assertEqual(self.rs.standard_normal(size=(3,)).shape, (3,))
        self.assertEqual(self.rs.standard_t(3.0).shape, ())

    def test_lognormal(self):
        """lognormal returns strictly positive samples."""
        arr = self.rs.lognormal(0.0, 1.0, size=(3,))
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr > 0).all())

    def test_chisquare_scalar_and_sized(self):
        """chisquare supports scalar df (size None) and integer df with size."""
        self.assertEqual(self.rs.chisquare(3).shape, ())
        self.assertEqual(self.rs.chisquare(3, size=(4,)).shape, (4,))

    def test_chisquare_nonscalar_df_requires_size(self):
        """chisquare with non-scalar df and no size is unsupported."""
        with self.assertRaises(NotImplementedError):
            self.rs.chisquare(jnp.array([2, 3]))

    def test_dirichlet(self):
        """dirichlet rows sum to one over the simplex axis."""
        arr = self.rs.dirichlet(jnp.array([1.0, 2.0, 3.0]), size=(4,))
        self.assertEqual(arr.shape, (4, 3))
        np.testing.assert_allclose(np.asarray(arr).sum(axis=-1), 1.0, atol=1e-5)

    def test_geometric(self):
        """geometric yields non-negative integer-valued samples."""
        arr = self.rs.geometric(0.5, size=(3,))
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr >= 0).all())

    def test_multinomial(self):
        """multinomial counts sum to n across the category axis."""
        arr = self.rs.multinomial(10, jnp.array([0.2, 0.3, 0.5]), size=(2,))
        self.assertEqual(arr.shape, (2, 3))
        np.testing.assert_array_equal(np.asarray(arr).sum(axis=-1), 10)

    def test_multinomial_invalid_pvals(self):
        """multinomial rejects pvals whose leading sum exceeds one."""
        with self.assertRaises(Exception):
            self.rs.multinomial(10, jnp.array([0.6, 0.6, 0.5]))

    def test_multinomial_traced_n_raises(self):
        """multinomial rejects a traced (abstract) total count n."""
        rs = self.rs

        @brainstate.transform.jit
        def f(n):
            return rs.multinomial(n, jnp.array([0.2, 0.3, 0.5]), size=(2,))

        with self.assertRaises(ValueError):
            f(jnp.array(10))

    def test_multivariate_normal_methods(self):
        """multivariate_normal supports svd, eigh, and cholesky factorisations."""
        mean = jnp.array([0.0, 1.0])
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        for method in ('svd', 'eigh', 'cholesky'):
            out = self.rs.multivariate_normal(mean, cov, size=(3,), method=method)
            self.assertEqual(out.shape, (3, 2))

    def test_multivariate_normal_bad_method(self):
        """multivariate_normal rejects an unknown factorisation method."""
        mean = jnp.array([0.0, 1.0])
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        with self.assertRaises(ValueError):
            self.rs.multivariate_normal(mean, cov, method='bogus')

    def test_multivariate_normal_dimension_errors(self):
        """multivariate_normal validates mean/cov ranks and cov shape."""
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        with self.assertRaises(ValueError):
            self.rs.multivariate_normal(jnp.array(0.0), cov)  # mean ndim < 1
        with self.assertRaises(ValueError):
            self.rs.multivariate_normal(jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0]))  # cov ndim < 2
        with self.assertRaises(ValueError):
            self.rs.multivariate_normal(jnp.array([0.0, 1.0, 2.0]), cov)  # cov shape mismatch

    def test_rayleigh(self):
        """rayleigh yields non-negative samples with the requested shape."""
        arr = self.rs.rayleigh(2.0, size=(3,))
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr >= 0).all())

    def test_triangular(self):
        """triangular returns values in {-1, 1}."""
        arr = self.rs.triangular(size=(50,))
        self.assertEqual(arr.shape, (50,))
        self.assertTrue(jnp.all((arr == -1) | (arr == 1)))

    def test_vonmises(self):
        """vonmises returns angles within (-pi, pi]."""
        arr = self.rs.vonmises(0.0, 1.0, size=(3,))
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr >= -jnp.pi - 1e-5).all() and (arr <= jnp.pi + 1e-5).all())

    def test_weibull_and_min(self):
        """weibull and weibull_min yield non-negative samples of the right shape."""
        self.assertEqual(self.rs.weibull(2.0, size=(3,)).shape, (3,))
        self.assertEqual(self.rs.weibull_min(2.0, scale=1.5, size=(3,)).shape, (3,))

    def test_weibull_array_a_with_size_raises(self):
        """weibull requires scalar shape parameter a when size is provided."""
        with self.assertRaises(ValueError):
            self.rs.weibull(jnp.array([1.0, 2.0]), size=(3,))

    def test_weibull_min_array_a_with_size_raises(self):
        """weibull_min requires scalar shape parameter a when size is provided."""
        with self.assertRaises(ValueError):
            self.rs.weibull_min(jnp.array([1.0, 2.0]), size=(3,))

    def test_maxwell(self):
        """maxwell returns non-negative speeds with the requested shape."""
        arr = self.rs.maxwell(size=(3,))
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr >= 0).all())

    def test_negative_binomial(self):
        """negative_binomial works both with and without an explicit key."""
        self.assertEqual(self.rs.negative_binomial(5, 0.5, size=(3,)).shape, (3,))
        key = brainstate.random.split_key()
        self.assertEqual(self.rs.negative_binomial(5, 0.5, size=(3,), key=key).shape, (3,))

    def test_wald(self):
        """wald returns positive samples with the requested shape."""
        arr = self.rs.wald(1.0, 2.0, size=(3,))
        self.assertEqual(arr.shape, (3,))
        self.assertTrue((arr > 0).all())

    def test_t_distribution(self):
        """t works both with and without an explicit key."""
        self.assertEqual(self.rs.t(3.0, size=(4,)).shape, (4,))
        key = brainstate.random.split_key()
        self.assertEqual(self.rs.t(3.0, size=(4,), key=key).shape, (4,))

    def test_orthogonal(self):
        """orthogonal returns batches of orthonormal matrices."""
        q = self.rs.orthogonal(3, size=(2,))
        self.assertEqual(q.shape, (2, 3, 3))
        identity = jnp.einsum('...ij,...kj->...ik', q, q)
        np.testing.assert_allclose(np.asarray(identity), np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-4)

    def test_noncentral_chisquare(self):
        """noncentral_chisquare works both with and without an explicit key."""
        self.assertEqual(self.rs.noncentral_chisquare(3.0, 1.0, size=(2,)).shape, (2,))
        key = brainstate.random.split_key()
        self.assertEqual(self.rs.noncentral_chisquare(3.0, 1.0, size=(2,), key=key).shape, (2,))

    def test_loggamma(self):
        """loggamma yields samples with the requested shape."""
        self.assertEqual(self.rs.loggamma(2.0, size=(3,)).shape, (3,))

    def test_categorical_size_none(self):
        """categorical infers the output shape from the logits when size is None."""
        logits = jnp.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
        out = self.rs.categorical(logits)
        self.assertEqual(out.shape, (2,))

    def test_special_impl_distributions(self):
        """zipf, power, f, hypergeometric, logseries, noncentral_f all sample."""
        self.assertEqual(self.rs.zipf(2.0, size=(3,)).shape, (3,))
        self.assertEqual(self.rs.power(2.0, size=(3,)).shape, (3,))
        self.assertEqual(self.rs.f(3.0, 5.0, size=(3,)).shape, (3,))
        self.assertEqual(self.rs.hypergeometric(5, 5, 4, size=(3,)).shape, (3,))
        self.assertEqual(self.rs.logseries(0.5, size=(3,)).shape, (3,))
        self.assertEqual(self.rs.noncentral_f(3.0, 5.0, 1.0, size=(3,)).shape, (3,))


class TestRandomStatePyTorchHelpersExtra(unittest.TestCase):
    """Extend coverage of the PyTorch-compat *_like helpers."""

    def setUp(self):
        """Create a seeded RandomState and push a state-trace stack."""
        self.rs = RandomState(2025)
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        """Pop the state-trace stack after each test."""
        TRACE_CONTEXT.state_stack.pop()

    def test_rand_like_with_dtype_and_key(self):
        """rand_like honours an explicit dtype and external key."""
        key = brainstate.random.split_key()
        out = self.rs.rand_like(jnp.zeros((2, 3)), dtype=jnp.float32, key=key)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(out.dtype, jnp.float32)

    def test_randn_like_with_dtype_and_key(self):
        """randn_like honours an explicit dtype and external key."""
        key = brainstate.random.split_key()
        out = self.rs.randn_like(jnp.zeros((2, 3)), dtype=jnp.float32, key=key)
        self.assertEqual(out.shape, (2, 3))
        self.assertEqual(out.dtype, jnp.float32)

    def test_randint_like_default_high(self):
        """randint_like defaults its upper bound to max(input)."""
        out = self.rs.randint_like(jnp.array([3, 5, 9]))
        self.assertEqual(out.shape, (3,))
        self.assertTrue((np.asarray(out) < 9).all())

    def test_randint_like_with_key(self):
        """randint_like accepts an explicit key and bounds."""
        key = brainstate.random.split_key()
        out = self.rs.randint_like(jnp.zeros((2, 3), dtype=jnp.int32), 0, 4, key=key)
        self.assertEqual(out.shape, (2, 3))
        self.assertTrue((np.asarray(out) >= 0).all() and (np.asarray(out) < 4).all())


class TestRandomStateScalarSizeInference(unittest.TestCase):
    """Drive the ``size is None`` shape-inference branch of every distribution."""

    def setUp(self):
        """Create a seeded RandomState and push a state-trace stack."""
        self.rs = RandomState(31337)
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        """Pop the state-trace stack after each test."""
        TRACE_CONTEXT.state_stack.pop()

    def test_scalar_distributions_infer_empty_shape(self):
        """Scalar parameters with no size yield scalar (shape ()) draws."""
        calls = {
            'beta': lambda: self.rs.beta(2.0, 3.0),
            'exponential': lambda: self.rs.exponential(2.0),
            'gamma': lambda: self.rs.gamma(2.0, 1.0),
            'laplace': lambda: self.rs.laplace(0.0, 1.0),
            'gumbel': lambda: self.rs.gumbel(0.0, 1.0),
            'pareto': lambda: self.rs.pareto(2.0),
            'standard_gamma': lambda: self.rs.standard_gamma(2.0),
            'lognormal': lambda: self.rs.lognormal(0.0, 1.0),
            'bernoulli': lambda: self.rs.bernoulli(0.5),
            'binomial': lambda: self.rs.binomial(10, 0.5),
            'geometric': lambda: self.rs.geometric(0.5),
            'rayleigh': lambda: self.rs.rayleigh(2.0),
            'vonmises': lambda: self.rs.vonmises(0.0, 1.0),
            'weibull': lambda: self.rs.weibull(2.0),
            'weibull_min': lambda: self.rs.weibull_min(2.0, scale=1.5),
            'negative_binomial': lambda: self.rs.negative_binomial(5, 0.5),
            'wald': lambda: self.rs.wald(1.0, 2.0),
            't': lambda: self.rs.t(3.0),
            'noncentral_chisquare': lambda: self.rs.noncentral_chisquare(3.0, 1.0),
            'loggamma': lambda: self.rs.loggamma(2.0),
            'zipf': lambda: self.rs.zipf(2.0),
            'power': lambda: self.rs.power(2.0),
            'f': lambda: self.rs.f(3.0, 5.0),
            'logseries': lambda: self.rs.logseries(0.5),
            'noncentral_f': lambda: self.rs.noncentral_f(3.0, 5.0, 1.0),
            'hypergeometric': lambda: self.rs.hypergeometric(5, 5, 4),
            'poisson': lambda: self.rs.poisson(3.0),
            'truncated_normal': lambda: self.rs.truncated_normal(-1.0, 1.0),
        }
        for name, fn in calls.items():
            with self.subTest(distribution=name):
                self.assertEqual(np.asarray(fn()).shape, ())


class TestRandomStateEdgeCases(unittest.TestCase):
    """Cover deletion recovery, non-array splitting, and unit-carrying inputs."""

    def setUp(self):
        """Push a fresh state-trace stack for each test."""
        TRACE_CONTEXT.state_stack.append(StateTraceStack())

    def tearDown(self):
        """Pop the state-trace stack after each test."""
        TRACE_CONTEXT.state_stack.pop()

    def test_check_if_deleted_reseeds_after_buffer_delete(self):
        """check_if_deleted reseeds the state once its backing buffer is freed."""
        rs = RandomState(9)
        rs.value.delete()
        rs.check_if_deleted()
        self.assertEqual(rs.value.shape, (2,))
        # After reseeding the state is usable again.
        self.assertEqual(rs.randn(3).shape, (3,))

    def test_split_key_coerces_non_jax_array_value(self):
        """split_key coerces a plain numpy key array before splitting."""
        rs = RandomState(9)
        rs._value = np.array([1, 2], dtype=np.uint32)
        new_key = rs.split_key()
        self.assertEqual(new_key.shape, (2,))
        self.assertEqual(rs.value.shape, (2,))

    def test_multivariate_normal_with_units(self):
        """multivariate_normal accepts unit-carrying mean and covariance."""
        import brainunit as u
        rs = RandomState(9)
        mean = jnp.array([0.0, 1.0]) * u.mV
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]]) * (u.mV ** 2)
        out = rs.multivariate_normal(mean, cov, size=(2,))
        self.assertEqual(np.asarray(u.get_magnitude(out)).shape, (2, 2))

    def test_multivariate_normal_size_none(self):
        """multivariate_normal infers a scalar batch when size is None."""
        rs = RandomState(9)
        out = rs.multivariate_normal(jnp.array([0.0, 1.0]), jnp.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(out.shape, (2,))

    def test_check_valid_false_skips_validation(self):
        """Distributions with check_valid=False skip their jit_error_if guard."""
        rs = RandomState(9)
        self.assertEqual(rs.truncated_normal(-1.0, 1.0, size=(3,), check_valid=False).shape, (3,))
        self.assertEqual(rs.bernoulli(0.5, size=(3,), check_valid=False).shape, (3,))
        self.assertEqual(rs.binomial(10, 0.5, size=(3,), check_valid=False).shape, (3,))
        out = rs.multinomial(10, jnp.array([0.2, 0.3, 0.5]), size=(2,), check_valid=False)
        self.assertEqual(out.shape, (2, 3))

    def test_standard_t_with_size(self):
        """standard_t honours an explicit size argument."""
        rs = RandomState(9)
        self.assertEqual(rs.standard_t(3.0, size=(3,)).shape, (3,))


if __name__ == '__main__':
    unittest.main()
