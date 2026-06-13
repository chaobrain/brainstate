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
import numpy as np
import pytest

import brainstate


class TestRandom(unittest.TestCase):
    """Smoke tests for seeding the global random state."""

    def test_seed2(self):
        """seed accepts a key array, an int, and None at top level. Inside a
        raw jax.jit trace, re-seeding the global RNG with a traced key is
        rejected (the key tracer would outlive the trace and corrupt the
        global random state); brainstate.transform.jit tracks it properly."""
        test_seed = 299
        brainstate.random.seed(test_seed)
        key = brainstate.random.get_key()
        brainstate.random.seed(key)
        brainstate.random.seed(1)
        brainstate.random.seed(None)

        @jax.jit
        def jit_seed(key):
            brainstate.random.seed(key)
            with brainstate.random.seed_context(key):
                return brainstate.random.DEFAULT.value

        with self.assertRaises(brainstate.TraceContextError):
            jit_seed(key)

        @brainstate.transform.jit
        def bst_jit_seed(key):
            brainstate.random.seed(key)
            return brainstate.random.DEFAULT.value

        bst_jit_seed(key)
        brainstate.random.seed(1)

    def test_seed(self):
        test_seed = 299
        brainstate.random.seed(test_seed)
        a = brainstate.random.rand(3)
        brainstate.random.seed(test_seed)
        b = brainstate.random.rand(3)
        self.assertTrue(jnp.array_equal(a, b))


class TestSeedUtilities(unittest.TestCase):
    """Cover the seed and key management utilities of ``brainstate.random._seed``."""

    def setUp(self):
        """Reset the global random state to a known seed before each test."""
        brainstate.random.seed(1234)

    # ------------------------------------------------------------------ #
    # seed                                                               #
    # ------------------------------------------------------------------ #

    def test_seed_makes_draws_reproducible(self):
        """Seed with an integer then draw reproduces the same sequence."""
        brainstate.random.seed(7)
        a = brainstate.random.randn(5)
        brainstate.random.seed(7)
        b = brainstate.random.randn(5)
        self.assertTrue(bool(jnp.allclose(a, b)))

    def test_seed_none_generates_random_state(self):
        """Seed with None auto-generates a usable random state."""
        brainstate.random.seed(None)
        drawn = brainstate.random.randn(3)
        self.assertEqual(drawn.shape, (3,))

    def test_seed_accepts_jax_key(self):
        """Seed accepts a typed JAX PRNG key (interop path)."""
        brainstate.random.seed(7)
        key = brainstate.random.split_key()
        # feeding a typed key exercises the ``_is_typed_key`` branch
        brainstate.random.seed(key)
        drawn = brainstate.random.randn(2)
        self.assertEqual(drawn.shape, (2,))

    def test_seed_accepts_legacy_uint32_key(self):
        """Seed accepts a legacy ``uint32[2]`` key (auto-wrapped)."""
        brainstate.random.seed(7)
        raw = brainstate.random.get_key_data()
        self.assertEqual(raw.shape, (2,))
        self.assertEqual(raw.dtype, jnp.uint32)
        brainstate.random.seed(raw)
        drawn = brainstate.random.randn(2)
        self.assertEqual(drawn.shape, (2,))

    def test_seed_rejects_oversized_array(self):
        """Seed rejects an array whose size exceeds two.

        Validation is delegated to ``_format_key`` (which raises ``TypeError``
        for a uint32 array that is not size 2); previously a hand-rolled
        ``np.size`` branch raised ``ValueError``. Accept either so the test
        documents the rejection rather than the precise error class."""
        bad = np.array([1, 2, 3], dtype=np.uint32)
        with self.assertRaises((TypeError, ValueError)):
            brainstate.random.seed(bad)

    # ------------------------------------------------------------------ #
    # get_key / set_key                                                  #
    # ------------------------------------------------------------------ #

    def test_get_key_returns_current_value(self):
        """get_key returns the current global typed-key snapshot."""
        brainstate.random.seed(42)
        key = brainstate.random.get_key()
        self.assertEqual(key.shape, ())
        self.assertTrue(jnp.issubdtype(key.dtype, jax.dtypes.prng_key))

    def test_get_key_data_returns_raw_uint32(self):
        """get_key_data returns the raw ``uint32[2]`` representation."""
        brainstate.random.seed(42)
        data = brainstate.random.get_key_data()
        self.assertEqual(data.shape, (2,))
        self.assertEqual(data.dtype, jnp.uint32)
        # get_key_data is the raw view of get_key.
        self.assertTrue(bool(jnp.array_equal(
            data, jax.random.key_data(brainstate.random.get_key()))))

    def test_set_get_key_roundtrip_with_integer(self):
        """set_key with an integer seed restores a reproducible sequence."""
        brainstate.random.set_key(42)
        a = brainstate.random.rand(3)
        brainstate.random.set_key(42)
        b = brainstate.random.rand(3)
        self.assertTrue(bool(jnp.allclose(a, b)))

    def test_set_key_with_raw_uint32_key(self):
        """set_key accepts a legacy raw uint32 size-2 key array (auto-wrapped)."""
        brainstate.random.seed(5)
        raw_key = brainstate.random.get_key_data()
        self.assertEqual(raw_key.dtype, jnp.uint32)
        brainstate.random.set_key(raw_key)
        # The wrapped key reproduces the same raw data.
        self.assertTrue(bool(jnp.array_equal(
            brainstate.random.get_key_data(), raw_key)))

    def test_set_key_with_typed_prng_key(self):
        """set_key accepts a typed JAX PRNG key (interop path)."""
        typed_key = jax.random.key(7)
        self.assertTrue(jnp.issubdtype(typed_key.dtype, jax.dtypes.prng_key))
        brainstate.random.set_key(typed_key)
        self.assertTrue(jnp.issubdtype(brainstate.random.get_key().dtype, jax.dtypes.prng_key))

    def test_set_key_roundtrip_via_get_key(self):
        """A key captured by get_key can be restored via set_key."""
        brainstate.random.seed(99)
        snapshot = brainstate.random.get_key()
        _ = brainstate.random.rand(4)  # advance state
        brainstate.random.set_key(snapshot)
        self.assertTrue(bool(jnp.array_equal(brainstate.random.get_key(), snapshot)))

    def test_set_key_rejects_wrong_shaped_array(self):
        """set_key raises TypeError for a non-key uint32 array."""
        bad = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        with self.assertRaises(TypeError):
            brainstate.random.set_key(bad)

    def test_set_key_rejects_float_array(self):
        """set_key raises TypeError for a float array."""
        bad = np.array([1.0, 2.0], dtype=np.float32)
        with self.assertRaises(TypeError):
            brainstate.random.set_key(bad)

    def test_set_key_rejects_non_array(self):
        """set_key raises TypeError for a non-array, non-integer input."""
        with self.assertRaises(TypeError):
            brainstate.random.set_key("not-a-key")

    # ------------------------------------------------------------------ #
    # split_key / split_keys                                             #
    # ------------------------------------------------------------------ #

    def test_split_key_single(self):
        """split_key with no argument returns one scalar typed key."""
        brainstate.random.seed(0)
        key = brainstate.random.split_key()
        self.assertEqual(key.shape, ())
        self.assertTrue(jnp.issubdtype(key.dtype, jax.dtypes.prng_key))

    def test_split_key_advances_state(self):
        """split_key advances the global state, yielding distinct keys."""
        brainstate.random.seed(0)
        k1 = brainstate.random.split_key()
        k2 = brainstate.random.split_key()
        self.assertFalse(bool(jnp.array_equal(k1, k2)))

    def test_split_key_multiple(self):
        """split_key with n returns a batch of n typed keys."""
        brainstate.random.seed(0)
        keys = brainstate.random.split_key(3)
        self.assertEqual(keys.shape, (3,))
        self.assertTrue(jnp.issubdtype(keys.dtype, jax.dtypes.prng_key))

    def test_split_keys_count(self):
        """split_keys(n) returns exactly n keys."""
        brainstate.random.seed(0)
        keys = brainstate.random.split_keys(4)
        self.assertEqual(keys.shape[0], 4)

    def test_split_keys_one(self):
        """split_keys(1) returns a single key in a batch of one."""
        brainstate.random.seed(0)
        keys = brainstate.random.split_keys(1)
        self.assertEqual(keys.shape[0], 1)

    def test_split_keys_rejects_zero(self):
        """split_keys raises ValueError for n == 0."""
        with self.assertRaises(ValueError):
            brainstate.random.split_keys(0)

    def test_split_keys_rejects_negative(self):
        """split_keys raises ValueError for a negative n."""
        with self.assertRaises(ValueError):
            brainstate.random.split_keys(-2)

    def test_split_keys_rejects_non_integer(self):
        """split_keys raises ValueError for a non-integer n."""
        with self.assertRaises(ValueError):
            brainstate.random.split_keys(2.5)

    def test_split_key_with_backup_then_restore(self):
        """split_key(backup=True) backs up the pre-split key so restore_key rewinds."""
        brainstate.random.seed(5)
        original = brainstate.random.get_key()  # capture BEFORE the split
        keys = brainstate.random.split_key(2, backup=True)
        self.assertEqual(keys.shape, (2,))
        _ = brainstate.random.rand(2)  # advance state
        brainstate.random.restore_key()
        # restore rewinds to the original pre-split key (documented contract),
        # not the already-advanced key.
        self.assertTrue(bool(jnp.array_equal(brainstate.random.get_key(), original)))

    # ------------------------------------------------------------------ #
    # restore_key                                                        #
    # ------------------------------------------------------------------ #

    def test_restore_key_without_backup_raises(self):
        """restore_key raises ValueError when no backup exists."""
        brainstate.random.seed(0)
        with self.assertRaises(ValueError):
            brainstate.random.restore_key()

    # ------------------------------------------------------------------ #
    # self_assign_multi_keys                                             #
    # ------------------------------------------------------------------ #

    def test_self_assign_multi_keys_with_backup(self):
        """self_assign_multi_keys(backup=True) stores n keys and backs up a single key."""
        brainstate.random.seed(9)
        brainstate.random.self_assign_multi_keys(3, backup=True)
        # the working value now holds the n multi-keys (batch of typed keys)
        self.assertEqual(brainstate.random.get_key().shape, (3,))
        # restore_key recovers the original pre-assignment single key
        brainstate.random.restore_key()
        restored = brainstate.random.get_key()
        self.assertEqual(restored.shape, ())
        # the restored single key supports normal draws again
        self.assertEqual(brainstate.random.randn(3).shape, (3,))

    def test_self_assign_multi_keys_without_backup(self):
        """self_assign_multi_keys(backup=False) stores n keys without a backup."""
        brainstate.random.seed(9)
        brainstate.random.self_assign_multi_keys(2, backup=False)
        self.assertEqual(brainstate.random.get_key().shape, (2,))
        with self.assertRaises(ValueError):
            brainstate.random.restore_key()

    def test_self_assign_multi_keys_rejects_zero(self):
        """self_assign_multi_keys raises ValueError for n == 0."""
        with self.assertRaises(ValueError):
            brainstate.random.self_assign_multi_keys(0)

    def test_self_assign_multi_keys_rejects_negative(self):
        """self_assign_multi_keys raises ValueError for a negative n."""
        with self.assertRaises(ValueError):
            brainstate.random.self_assign_multi_keys(-1)

    def test_self_assign_multi_keys_rejects_non_integer(self):
        """self_assign_multi_keys raises ValueError for a non-integer n."""
        with self.assertRaises(ValueError):
            brainstate.random.self_assign_multi_keys(1.5)

    # ------------------------------------------------------------------ #
    # clone_rng                                                          #
    # ------------------------------------------------------------------ #

    def test_clone_rng_default_is_independent_copy(self):
        """clone_rng() returns a RandomState distinct from the global default."""
        brainstate.random.seed(3)
        cloned = brainstate.random.clone_rng()
        self.assertIsInstance(cloned, brainstate.random.RandomState)
        self.assertIsNot(cloned, brainstate.random.DEFAULT)

    def test_clone_rng_no_clone_returns_global(self):
        """clone_rng(clone=False) returns the global default state directly."""
        same = brainstate.random.clone_rng(clone=False)
        self.assertIs(same, brainstate.random.DEFAULT)

    def test_clone_rng_with_seed_creates_new_state(self):
        """clone_rng(seed) builds a fresh independent RandomState."""
        rng = brainstate.random.clone_rng(123)
        self.assertIsInstance(rng, brainstate.random.RandomState)
        self.assertIsNot(rng, brainstate.random.DEFAULT)

    def test_clone_rng_clones_are_independent(self):
        """Two clones of the same state draw identical sequences independently."""
        brainstate.random.seed(7)
        r1 = brainstate.random.clone_rng(321)
        brainstate.random.seed(7)
        r2 = brainstate.random.clone_rng(321)
        self.assertTrue(bool(jnp.allclose(r1.randn(4), r2.randn(4))))

    # ------------------------------------------------------------------ #
    # default_rng                                                        #
    # ------------------------------------------------------------------ #

    def test_default_rng_returns_global(self):
        """default_rng() returns the global default state."""
        self.assertIs(brainstate.random.default_rng(), brainstate.random.DEFAULT)

    def test_default_rng_with_seed_creates_new_state(self):
        """default_rng(seed) builds a fresh independent RandomState."""
        rng = brainstate.random.default_rng(456)
        self.assertIsInstance(rng, brainstate.random.RandomState)
        self.assertIsNot(rng, brainstate.random.DEFAULT)

    def test_default_rng_seed_is_reproducible(self):
        """default_rng(seed) produces reproducible draws for the same seed."""
        a = brainstate.random.default_rng(99).randn(5)
        b = brainstate.random.default_rng(99).randn(5)
        self.assertTrue(bool(jnp.allclose(a, b)))

    # ------------------------------------------------------------------ #
    # seed_context                                                       #
    # ------------------------------------------------------------------ #

    def test_seed_context_is_reproducible(self):
        """seed_context yields identical draws for the same temporary seed."""
        with brainstate.random.seed_context(42):
            a = brainstate.random.rand(3)
        with brainstate.random.seed_context(42):
            b = brainstate.random.rand(3)
        self.assertTrue(bool(jnp.allclose(a, b)))

    def test_seed_context_restores_outer_state(self):
        """seed_context restores the prior global key on exit."""
        brainstate.random.seed(7)
        before = brainstate.random.get_key()
        with brainstate.random.seed_context(1000):
            _ = brainstate.random.rand(2)
        after = brainstate.random.get_key()
        self.assertTrue(bool(jnp.array_equal(before, after)))

    def test_seed_context_restores_on_exception(self):
        """seed_context restores the prior key even if the block raises."""
        brainstate.random.seed(7)
        before = brainstate.random.get_key()
        with self.assertRaises(ValueError):
            with brainstate.random.seed_context(1000):
                _ = brainstate.random.rand(2)
                raise ValueError("boom")
        after = brainstate.random.get_key()
        self.assertTrue(bool(jnp.array_equal(before, after)))

    def test_seed_context_nested(self):
        """Nested seed_context blocks each restore their own outer state."""
        brainstate.random.seed(7)
        before = brainstate.random.get_key()
        with brainstate.random.seed_context(123):
            outer_after_enter = brainstate.random.get_key()
            with brainstate.random.seed_context(456):
                _ = brainstate.random.rand(2)
            self.assertTrue(bool(jnp.array_equal(brainstate.random.get_key(), outer_after_enter)))
        self.assertTrue(bool(jnp.array_equal(brainstate.random.get_key(), before)))


class TestSeedGlobalStateEdgeCases(unittest.TestCase):
    """Regression tests for seed()/seed_context() global-state correctness.

    These tests deliberately exercise and inspect the global NumPy + JAX random
    state, so each one restores a known state on exit (``tearDown``) to avoid
    perturbing other tests in the suite.
    """

    def setUp(self):
        brainstate.random.seed(1234)

    def tearDown(self):
        # Re-establish a deterministic global state for subsequent tests.
        brainstate.random.seed(1234)

    # ------------------------------------------------------------------ #
    # M25: integer seeds outside [0, 2**32-1]                            #
    # ------------------------------------------------------------------ #

    def test_seed_large_int_does_not_raise(self):
        """seed(2**33) succeeds (no ValueError from np.random.seed) and is
        reproducible. Regression test for M25."""
        brainstate.random.seed(2 ** 33)
        a = brainstate.random.randn(5)
        brainstate.random.seed(2 ** 33)
        b = brainstate.random.randn(5)
        self.assertTrue(bool(jnp.allclose(a, b)))

    def test_seed_negative_int_does_not_raise(self):
        """seed(-1) succeeds and is reproducible. Regression test for M25."""
        brainstate.random.seed(-1)
        a = brainstate.random.randn(5)
        brainstate.random.seed(-1)
        b = brainstate.random.randn(5)
        self.assertTrue(bool(jnp.allclose(a, b)))

    def test_seed_numpy_int64_out_of_range_does_not_raise(self):
        """seed(np.int64(2**33)) succeeds via the size-1 array path.
        Regression test for M25."""
        brainstate.random.seed(np.int64(2 ** 33))
        drawn = brainstate.random.randn(3)
        self.assertEqual(drawn.shape, (3,))

    def test_seed_out_of_range_int_matches_mod_2pow32_for_jax(self):
        """An out-of-range int seed produces the same JAX key as its value
        reduced modulo 2**32 (JAX uses the low 32 bits). Regression test for
        M25."""
        brainstate.random.seed(2 ** 33 + 7)
        k1 = brainstate.random.get_key_data()
        brainstate.random.seed((2 ** 33 + 7) % (2 ** 32))
        k2 = brainstate.random.get_key_data()
        self.assertTrue(bool(jnp.array_equal(k1, k2)))

    def test_seed_out_of_range_int_seeds_numpy_in_range(self):
        """After a large int seed, NumPy's state matches seeding NumPy with the
        value reduced modulo 2**32. Regression test for M25."""
        brainstate.random.seed(2 ** 32 + 7)
        a = float(np.random.rand())
        np.random.seed((2 ** 32 + 7) % (2 ** 32))
        b = float(np.random.rand())
        self.assertEqual(a, b)

    # ------------------------------------------------------------------ #
    # M26: size-2 tuple/list inputs must not partially seed NumPy        #
    # ------------------------------------------------------------------ #

    def test_seed_tuple_raises_and_leaves_numpy_state_unchanged(self):
        """seed((5, 9)) raises and leaves NumPy's global state untouched (no
        partial mutation before the JAX-key validation fails). Regression test
        for M26."""
        np.random.seed(777)
        before = np.random.get_state()
        before_key = brainstate.random.get_key()
        with self.assertRaises((TypeError, ValueError)):
            brainstate.random.seed((5, 9))
        after = np.random.get_state()
        # NumPy state vector unchanged.
        self.assertEqual(before[0], after[0])
        self.assertTrue(bool(np.array_equal(before[1], after[1])))
        self.assertEqual(before[2], after[2])
        # JAX key unchanged too.
        self.assertTrue(bool(jnp.array_equal(before_key, brainstate.random.get_key())))

    def test_seed_float_size2_array_raises_without_mutating_numpy(self):
        """seed(np.array([5., 9.])) raises (float, not uint32) and does not
        mutate NumPy's state. Regression test for M26."""
        np.random.seed(777)
        before = np.random.get_state()
        with self.assertRaises((TypeError, ValueError)):
            brainstate.random.seed(np.array([5.0, 9.0]))
        after = np.random.get_state()
        self.assertTrue(bool(np.array_equal(before[1], after[1])))
        self.assertEqual(before[2], after[2])

    def test_seed_int64_size2_array_raises_without_mutating_numpy(self):
        """seed(np.array([5, 9], dtype=int64)) raises (not uint32) and does not
        mutate NumPy's state. Regression test for M26."""
        np.random.seed(777)
        before = np.random.get_state()
        with self.assertRaises((TypeError, ValueError)):
            brainstate.random.seed(np.array([5, 9], dtype=np.int64))
        after = np.random.get_state()
        self.assertTrue(bool(np.array_equal(before[1], after[1])))
        self.assertEqual(before[2], after[2])

    # ------------------------------------------------------------------ #
    # L15: seed(None) honors NumPy state and uses full entropy range     #
    # ------------------------------------------------------------------ #

    def test_seed_none_does_not_advance_numpy_global_state(self):
        """seed(None) must NOT consume/advance NumPy's global RNG state, per the
        docstring. Regression test for L15."""
        np.random.seed(999)
        before = np.random.get_state()
        brainstate.random.seed(None)
        after = np.random.get_state()
        self.assertEqual(before[2], after[2])  # position counter unchanged
        self.assertTrue(bool(np.array_equal(before[1], after[1])))
        # The very next np.random draw matches what it would have been without
        # the brainstate.random.seed(None) call.
        np.random.seed(999)
        expected = float(np.random.rand())
        np.random.seed(999)
        brainstate.random.seed(None)
        got = float(np.random.rand())
        self.assertEqual(expected, got)

    def test_seed_none_yields_usable_full_uint32_key(self):
        """seed(None) produces a usable typed key derived from a full
        uint32[2] auto-key. Regression test for L15."""
        brainstate.random.seed(None)
        key = brainstate.random.get_key()
        self.assertEqual(key.shape, ())
        self.assertTrue(jnp.issubdtype(key.dtype, jax.dtypes.prng_key))
        self.assertEqual(brainstate.random.randn(3).shape, (3,))

    # ------------------------------------------------------------------ #
    # M27: seed_context also seeds & restores NumPy's global state        #
    # ------------------------------------------------------------------ #

    def test_seed_context_makes_numpy_reproducible(self):
        """Inside seed_context(seed), NumPy-backed randomness is reproducible
        across two identically-seeded contexts. Regression test for M27."""
        with brainstate.random.seed_context(42):
            a = float(np.random.rand())
        with brainstate.random.seed_context(42):
            b = float(np.random.rand())
        self.assertEqual(a, b)

    def test_seed_context_restores_numpy_state_on_exit(self):
        """seed_context restores NumPy's full global state on exit, so a draw
        after the block matches the pre-context stream. Regression test for
        M27."""
        np.random.seed(2024)
        before = np.random.get_state()
        expected_next = float(np.random.rand())
        # Re-establish the pre-draw state, run a context that perturbs NumPy,
        # and confirm the next draw is unaffected.
        np.random.set_state(before)
        with brainstate.random.seed_context(123):
            _ = np.random.rand(5)  # perturb NumPy inside the context
        after = np.random.get_state()
        self.assertEqual(before[2], after[2])
        self.assertTrue(bool(np.array_equal(before[1], after[1])))
        self.assertEqual(expected_next, float(np.random.rand()))

    def test_seed_context_restores_numpy_state_on_exception(self):
        """seed_context restores NumPy's state even when the block raises.
        Regression test for M27."""
        np.random.seed(2024)
        before = np.random.get_state()
        with self.assertRaises(ValueError):
            with brainstate.random.seed_context(123):
                _ = np.random.rand(5)
                raise ValueError("boom")
        after = np.random.get_state()
        self.assertTrue(bool(np.array_equal(before[1], after[1])))
        self.assertEqual(before[2], after[2])
