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


if __name__ == "__main__":
    unittest.main()
