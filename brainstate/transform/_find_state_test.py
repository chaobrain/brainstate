import unittest

import jax.numpy as jnp

import brainstate as bst
from brainstate.transform import StateFinder


class TestStateFinder(unittest.TestCase):
    def test_default_dictionary_output(self):
        read_state = bst.State(jnp.array(0.0), name='read_state')
        param_state = bst.ParamState(jnp.array(1.0), name='param_state')

        def fn(scale):
            _ = read_state.value
            param_state.value = param_state.value * scale
            return param_state.value + _

        finder = StateFinder(fn)
        result = finder(2.0)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result.values()), {read_state, param_state})

    def test_filter_and_usage_read(self):
        buffer_state = bst.State(jnp.array(1.0), name='buffer')
        param_state = bst.ParamState(jnp.array(3.0), name='param')

        def fn(offset):
            _ = buffer_state.value
            param_state.value = param_state.value + offset
            return param_state.value

        read_finder = StateFinder(fn, usage='read', return_type='list')
        reads = read_finder(1.0)
        self.assertEqual(reads, [buffer_state])

        param_finder = StateFinder(fn, filter=bst.ParamState, usage='all')
        param_states = param_finder(1.0)
        self.assertEqual(list(param_states.values()), [param_state])

    def test_usage_write_with_custom_key(self):
        param_state = bst.ParamState(jnp.array(5.0), name='param')

        def fn(scale):
            param_state.value = param_state.value * scale
            return param_state.value

        finder = StateFinder(fn, usage='write', return_type='dict', key_fn=lambda idx, st: f"w_{idx}")
        write_states = finder(2.0)
        self.assertIn('w_0', write_states)
        self.assertIs(write_states['w_0'], param_state)

    def test_usage_both_returns_separated_collections(self):
        read_state = bst.State(jnp.array(2.0), name='read')
        write_state = bst.ParamState(jnp.array(4.0), name='write')

        def fn(delta):
            _ = read_state.value
            write_state.value = write_state.value + delta
            return write_state.value

        finder = StateFinder(fn, usage='both', return_type='tuple')
        result = finder(1.5)
        self.assertEqual(set(result.keys()), {'read', 'write'})
        self.assertEqual(result['read'], (read_state,))
        self.assertEqual(result['write'], (write_state,))

    def test_duplicate_names_are_disambiguated(self):
        first = bst.State(jnp.array(0.0), name='dup')
        second = bst.State(jnp.array(1.0), name='dup')

        def fn():
            _ = first.value
            _ = second.value
            return None

        finder = StateFinder(fn)
        states = finder()
        self.assertEqual(len(states), 2)
        self.assertEqual(set(states.values()), {first, second})


class TestStateFinderValidation(unittest.TestCase):
    """Tests for invalid constructor arguments (lines 98, 100)."""

    def test_invalid_usage_raises(self):
        """Invalid usage string raises ValueError."""
        with self.assertRaises(ValueError):
            StateFinder(lambda: None, usage='invalid')

    def test_invalid_return_type_raises(self):
        """Invalid return_type string raises ValueError."""
        with self.assertRaises(ValueError):
            StateFinder(lambda: None, return_type='set')


class TestEnsureHashable(unittest.TestCase):
    """Tests for _ensure_hashable (lines 182, 185-186)."""

    def test_none_returns_none(self):
        """_ensure_hashable returns None unchanged."""
        result = StateFinder._ensure_hashable(None)
        self.assertIsNone(result)

    def test_unhashable_list_returns_str(self):
        """_ensure_hashable converts unhashable list to str."""
        result = StateFinder._ensure_hashable([1, 2, 3])
        self.assertEqual(result, str([1, 2, 3]))

    def test_hashable_int_returned_unchanged(self):
        """_ensure_hashable passes through a plain int."""
        self.assertEqual(StateFinder._ensure_hashable(42), 42)


class TestEnsureUniqueKey(unittest.TestCase):
    """Tests for _ensure_unique_key (lines 192-199)."""

    def test_none_key_uses_state_name(self):
        """None key falls back to state.name."""
        st = bst.State(jnp.array(0.0), name='mystate')
        result = StateFinder._ensure_unique_key(None, 0, st, set())
        self.assertEqual(result, 'mystate')

    def test_none_key_no_name_uses_state_idx(self):
        """None key with no state name falls back to state_{idx}."""
        st = bst.State(jnp.array(0.0))
        result = StateFinder._ensure_unique_key(None, 3, st, set())
        self.assertEqual(result, 'state_3')

    def test_duplicate_key_gets_suffix(self):
        """Key already in used set gets a numeric suffix."""
        st = bst.State(jnp.array(0.0), name='w')
        used = {'w'}
        result = StateFinder._ensure_unique_key('w', 0, st, used)
        self.assertEqual(result, 'w_1')

    def test_triple_duplicate_increments_suffix(self):
        """Key with two collisions gets suffix _2."""
        st = bst.State(jnp.array(0.0), name='w')
        used = {'w', 'w_1'}
        result = StateFinder._ensure_unique_key('w', 0, st, used)
        self.assertEqual(result, 'w_2')


class TestStateFinderKeyFnNone(unittest.TestCase):
    """Tests for key_fn returning None (triggers _ensure_unique_key None branch)."""

    def test_key_fn_returns_none_falls_back_to_name(self):
        """key_fn returning None causes the state name to be used as key."""
        st = bst.State(jnp.array(1.0), name='alpha')

        def fn():
            return st.value

        finder = StateFinder(fn, key_fn=lambda idx, s: None)
        result = finder()
        self.assertIn('alpha', result)
        self.assertIs(result['alpha'], st)

    def test_key_fn_returns_unhashable_falls_back(self):
        """key_fn returning an unhashable value is converted to str."""
        st = bst.State(jnp.array(1.0), name='beta')

        def fn():
            return st.value

        finder = StateFinder(fn, key_fn=lambda idx, s: [idx, 'x'])
        result = finder()
        self.assertEqual(len(result), 1)

    def test_filter_excludes_states(self):
        """Filter predicate removes non-matching states from the result."""
        s1 = bst.State(jnp.array(1.0), name='plain')
        p1 = bst.ParamState(jnp.array(2.0), name='param')

        def fn():
            _ = s1.value
            _ = p1.value
            return None

        finder = StateFinder(fn, filter=bst.ParamState, return_type='list')
        result = finder()
        self.assertEqual(result, [p1])

    def test_usage_write_list(self):
        """usage='write' with return_type='list' returns only written states."""
        w = bst.ParamState(jnp.array(3.0), name='w')

        def fn(x):
            w.value = w.value * x
            return w.value

        finder = StateFinder(fn, usage='write', return_type='list')
        result = finder(2.0)
        self.assertIn(w, result)


if __name__ == "__main__":
    unittest.main()
