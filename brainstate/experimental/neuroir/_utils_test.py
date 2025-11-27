# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Comprehensive tests for neuroir/_utils.py using unittest framework."""

import unittest
from unittest.mock import Mock

from brainstate._state import State
from brainstate.experimental.neuroir._utils import (
    get_hidden_name,
    UnionFind,
    find_in_states,
    find_out_states,
)


# ============================================================================
# Test get_hidden_name()
# ============================================================================

class TestGetHiddenName(unittest.TestCase):
    """Test suite for get_hidden_name function."""

    def test_state_with_name(self):
        """Test that state with name returns the name."""
        state = Mock(spec=State)
        state.name = "test_state"

        result = get_hidden_name(state)
        self.assertEqual(result, "test_state")

    def test_state_without_name_attribute(self):
        """Test that state without name attribute returns State@{id}."""
        state = Mock(spec=State)
        if hasattr(state, 'name'):
            delattr(state, 'name')

        result = get_hidden_name(state)
        expected_id = id(state)
        expected = f'State@{expected_id:x}'

        self.assertEqual(result, expected)
        self.assertTrue(result.startswith('State@'))

    def test_state_with_none_name(self):
        """Test that state with None name returns State@{id}."""
        state = Mock(spec=State)
        state.name = None

        result = get_hidden_name(state)
        expected_id = id(state)
        expected = f'State@{expected_id:x}'

        self.assertEqual(result, expected)

    def test_state_with_empty_string_name(self):
        """Test that state with empty string name returns State@{id} (falsy)."""
        state = Mock(spec=State)
        state.name = ""

        result = get_hidden_name(state)
        # Empty string is falsy, so it uses the ID
        expected_id = id(state)
        expected = f'State@{expected_id:x}'

        self.assertEqual(result, expected)

    def test_id_format_is_hexadecimal(self):
        """Test that the ID format is hexadecimal."""
        state = Mock(spec=State)
        if hasattr(state, 'name'):
            delattr(state, 'name')

        result = get_hidden_name(state)
        hex_part = result.split('@')[1]

        # Should be valid hexadecimal
        is_hex = all(c in '0123456789abcdef' for c in hex_part.lower())
        self.assertTrue(is_hex, f"ID part '{hex_part}' should be valid hexadecimal")

    def test_different_states_have_different_ids(self):
        """Test that different states get different ID-based names."""
        state1 = Mock(spec=State)
        state2 = Mock(spec=State)
        if hasattr(state1, 'name'):
            delattr(state1, 'name')
        if hasattr(state2, 'name'):
            delattr(state2, 'name')

        name1 = get_hidden_name(state1)
        name2 = get_hidden_name(state2)

        self.assertNotEqual(name1, name2)
        self.assertTrue(name1.startswith('State@'))
        self.assertTrue(name2.startswith('State@'))


# ============================================================================
# Test UnionFind
# ============================================================================

class TestUnionFind(unittest.TestCase):
    """Test suite for UnionFind data structure."""

    def test_make_set_creates_singleton(self):
        """Test that make_set creates a singleton set."""
        uf = UnionFind()
        uf.make_set('a')

        self.assertIn('a', uf.parent)
        self.assertEqual(uf.parent['a'], 'a')
        self.assertEqual(uf.rank['a'], 0)

    def test_make_set_is_idempotent(self):
        """Test that calling make_set twice doesn't change anything."""
        uf = UnionFind()
        uf.make_set('a')
        original_parent = uf.parent['a']
        original_rank = uf.rank['a']

        uf.make_set('a')

        self.assertEqual(uf.parent['a'], original_parent)
        self.assertEqual(uf.rank['a'], original_rank)

    def test_find_existing_element_returns_itself(self):
        """Test that find returns the element itself for a singleton."""
        uf = UnionFind()
        uf.make_set('a')

        result = uf.find('a')
        self.assertEqual(result, 'a')

    def test_find_non_existing_element_creates_and_returns_itself(self):
        """Test that find auto-creates and returns element if not exists."""
        uf = UnionFind()
        result = uf.find('a')

        self.assertEqual(result, 'a')
        self.assertIn('a', uf.parent)
        self.assertEqual(uf.parent['a'], 'a')

    def test_find_after_union_returns_root(self):
        """Test that find returns the root after union."""
        uf = UnionFind()
        uf.make_set('a')
        uf.make_set('b')
        uf.union('a', 'b')

        root_a = uf.find('a')
        root_b = uf.find('b')

        self.assertEqual(root_a, root_b)

    def test_path_compression_in_find(self):
        """Test that find performs path compression."""
        uf = UnionFind()
        uf.make_set('a')
        uf.make_set('b')
        uf.make_set('c')

        # Create chain: a -> b -> c
        uf.parent['a'] = 'b'
        uf.parent['b'] = 'c'

        # First find should compress the path
        root = uf.find('a')
        self.assertEqual(root, 'c')

        # After compression, 'a' should point directly to 'c'
        self.assertEqual(uf.parent['a'], 'c')

    def test_union_two_singletons(self):
        """Test union of two singleton sets."""
        uf = UnionFind()
        uf.make_set('a')
        uf.make_set('b')

        uf.union('a', 'b')

        root_a = uf.find('a')
        root_b = uf.find('b')

        self.assertEqual(root_a, root_b)

    def test_union_same_set_is_noop(self):
        """Test that union of elements in same set does nothing."""
        uf = UnionFind()
        uf.make_set('a')
        uf.make_set('b')
        uf.union('a', 'b')

        # Get initial state
        root_before = uf.find('a')
        rank_before = uf.rank[root_before]

        # Union again
        uf.union('a', 'b')

        # Should be unchanged
        root_after = uf.find('a')
        rank_after = uf.rank[root_after]

        self.assertEqual(root_before, root_after)
        self.assertEqual(rank_before, rank_after)

    def test_union_by_rank_smaller_to_larger(self):
        """Test union by rank merges smaller rank to larger rank."""
        uf = UnionFind()
        uf.make_set('a')
        uf.make_set('b')

        # Make 'a' have rank 1
        uf.rank['a'] = 1
        uf.rank['b'] = 0

        uf.union('a', 'b')

        # 'b' should be merged to 'a'
        self.assertEqual(uf.find('b'), 'a')

    def test_union_by_rank_equal_ranks_increments(self):
        """Test union by rank with equal ranks increments the root's rank."""
        uf = UnionFind()
        uf.make_set('a')
        uf.make_set('b')

        # Both have rank 0
        self.assertEqual(uf.rank['a'], 0)
        self.assertEqual(uf.rank['b'], 0)

        uf.union('a', 'b')

        root = uf.find('a')
        # The root's rank should be incremented
        self.assertEqual(uf.rank[root], 1)

    def test_multiple_unions_create_larger_sets(self):
        """Test multiple unions create larger sets correctly."""
        uf = UnionFind()
        elements = ['a', 'b', 'c', 'd', 'e']

        for elem in elements:
            uf.make_set(elem)

        # Union a-b, c-d, then a-c
        uf.union('a', 'b')
        uf.union('c', 'd')
        uf.union('a', 'c')

        # All of a, b, c, d should be in same set
        root = uf.find('a')
        self.assertEqual(uf.find('b'), root)
        self.assertEqual(uf.find('c'), root)
        self.assertEqual(uf.find('d'), root)

        # 'e' should be in its own set
        self.assertNotEqual(uf.find('e'), root)

    def test_get_groups_empty(self):
        """Test get_groups on empty UnionFind returns empty list."""
        uf = UnionFind()
        groups = uf.get_groups()

        self.assertEqual(groups, [])

    def test_get_groups_single_element(self):
        """Test get_groups with single element."""
        uf = UnionFind()
        uf.make_set('a')

        groups = uf.get_groups()

        self.assertEqual(len(groups), 1)
        self.assertIn('a', groups[0])

    def test_get_groups_multiple_disjoint(self):
        """Test get_groups with multiple disjoint elements."""
        uf = UnionFind()
        uf.make_set('a')
        uf.make_set('b')
        uf.make_set('c')

        groups = uf.get_groups()

        self.assertEqual(len(groups), 3)
        group_sets = {frozenset(g) for g in groups}
        expected = {frozenset({'a'}), frozenset({'b'}), frozenset({'c'})}
        self.assertEqual(group_sets, expected)

    def test_get_groups_after_unions(self):
        """Test get_groups after performing unions."""
        uf = UnionFind()
        elements = ['a', 'b', 'c', 'd', 'e']

        for elem in elements:
            uf.make_set(elem)

        # Create two groups: {a, b, c} and {d, e}
        uf.union('a', 'b')
        uf.union('b', 'c')
        uf.union('d', 'e')

        groups = uf.get_groups()

        self.assertEqual(len(groups), 2)

        # Convert to sets for easier comparison
        group_sets = [set(g) for g in groups]

        self.assertIn({'a', 'b', 'c'}, group_sets)
        self.assertIn({'d', 'e'}, group_sets)

    def test_get_groups_complex_scenario(self):
        """Test get_groups with complex union pattern."""
        uf = UnionFind()

        # Create 10 elements
        for i in range(10):
            uf.make_set(i)

        # Create groups: {0,1,2,3}, {4,5}, {6}, {7,8,9}
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(4, 5)
        uf.union(7, 8)
        uf.union(8, 9)

        groups = uf.get_groups()

        self.assertEqual(len(groups), 4)

        group_sets = [set(g) for g in groups]

        self.assertIn({0, 1, 2, 3}, group_sets)
        self.assertIn({4, 5}, group_sets)
        self.assertIn({6}, group_sets)
        self.assertIn({7, 8, 9}, group_sets)


# ============================================================================
# Test find_in_states and find_out_states
# ============================================================================

class TestFindInStates(unittest.TestCase):
    """Test suite for find_in_states function."""

    def test_empty_in_vars_returns_empty_list(self):
        """Test that empty in_vars returns empty list."""
        in_var_to_state = {}
        in_vars = []

        result = find_in_states(in_var_to_state, in_vars)

        self.assertEqual(result, [])

    def test_variables_in_mapping_returns_states(self):
        """Test that variables in mapping return their states."""
        var1 = Mock()
        var2 = Mock()

        state1 = Mock(spec=State)
        state2 = Mock(spec=State)

        in_var_to_state = {var1: state1, var2: state2}
        in_vars = [var1, var2]

        result = find_in_states(in_var_to_state, in_vars)

        self.assertEqual(len(result), 2)
        self.assertIn(state1, result)
        self.assertIn(state2, result)

    def test_variables_not_in_mapping_are_skipped(self):
        """Test that variables not in mapping are skipped."""
        var1 = Mock()
        var2 = Mock()
        var3 = Mock()

        state1 = Mock(spec=State)

        in_var_to_state = {var1: state1}
        in_vars = [var1, var2, var3]  # var2 and var3 not in mapping

        result = find_in_states(in_var_to_state, in_vars)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], state1)

    def test_duplicate_variables_same_state_appears_once(self):
        """Test that duplicate variables referencing same state only include state once."""
        var1 = Mock()
        var2 = Mock()

        state1 = Mock(spec=State)

        in_var_to_state = {var1: state1, var2: state1}
        in_vars = [var1, var2]

        result = find_in_states(in_var_to_state, in_vars)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], state1)

    def test_mix_of_mapped_and_unmapped_variables(self):
        """Test mix of mapped and unmapped variables."""
        var1 = Mock()
        var2 = Mock()
        var3 = Mock()
        var4 = Mock()

        state1 = Mock(spec=State)
        state2 = Mock(spec=State)

        in_var_to_state = {var1: state1, var3: state2}
        in_vars = [var1, var2, var3, var4]  # var2 and var4 unmapped

        result = find_in_states(in_var_to_state, in_vars)

        self.assertEqual(len(result), 2)
        self.assertIn(state1, result)
        self.assertIn(state2, result)

    def test_order_preservation_first_occurrence(self):
        """Test that order is preserved (first occurrence)."""
        var1 = Mock()
        var2 = Mock()
        var3 = Mock()

        state1 = Mock(spec=State)
        state2 = Mock(spec=State)
        state3 = Mock(spec=State)

        in_var_to_state = {var1: state1, var2: state2, var3: state3}
        in_vars = [var1, var2, var3]

        result = find_in_states(in_var_to_state, in_vars)

        self.assertEqual(result[0], state1)
        self.assertEqual(result[1], state2)
        self.assertEqual(result[2], state3)

    def test_multiple_states(self):
        """Test with multiple different states."""
        vars_list = [Mock() for _ in range(5)]
        states_list = [Mock(spec=State) for _ in range(5)]

        in_var_to_state = {v: s for v, s in zip(vars_list, states_list)}
        in_vars = vars_list

        result = find_in_states(in_var_to_state, in_vars)

        self.assertEqual(len(result), 5)
        for state in states_list:
            self.assertIn(state, result)


class TestFindOutStates(unittest.TestCase):
    """Test suite for find_out_states function (should behave like find_in_states)."""

    def test_empty_out_vars_returns_empty_list(self):
        """Test that empty out_vars returns empty list."""
        out_var_to_state = {}
        out_vars = []

        result = find_out_states(out_var_to_state, out_vars)

        self.assertEqual(result, [])

    def test_variables_in_mapping_returns_states(self):
        """Test that variables in mapping return their states."""
        var1 = Mock()
        var2 = Mock()

        state1 = Mock(spec=State)
        state2 = Mock(spec=State)

        out_var_to_state = {var1: state1, var2: state2}
        out_vars = [var1, var2]

        result = find_out_states(out_var_to_state, out_vars)

        self.assertEqual(len(result), 2)
        self.assertIn(state1, result)
        self.assertIn(state2, result)

    def test_variables_not_in_mapping_are_skipped(self):
        """Test that variables not in mapping are skipped."""
        var1 = Mock()
        var2 = Mock()
        var3 = Mock()

        state1 = Mock(spec=State)

        out_var_to_state = {var1: state1}
        out_vars = [var1, var2, var3]  # var2 and var3 not in mapping

        result = find_out_states(out_var_to_state, out_vars)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], state1)

    def test_duplicate_variables_same_state_appears_once(self):
        """Test that duplicate variables referencing same state only include state once."""
        var1 = Mock()
        var2 = Mock()

        state1 = Mock(spec=State)

        out_var_to_state = {var1: state1, var2: state1}
        out_vars = [var1, var2]

        result = find_out_states(out_var_to_state, out_vars)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], state1)

    def test_mix_of_mapped_and_unmapped_variables(self):
        """Test mix of mapped and unmapped variables."""
        var1 = Mock()
        var2 = Mock()
        var3 = Mock()
        var4 = Mock()

        state1 = Mock(spec=State)
        state2 = Mock(spec=State)

        out_var_to_state = {var1: state1, var3: state2}
        out_vars = [var1, var2, var3, var4]  # var2 and var4 unmapped

        result = find_out_states(out_var_to_state, out_vars)

        self.assertEqual(len(result), 2)
        self.assertIn(state1, result)
        self.assertIn(state2, result)

    def test_order_preservation_first_occurrence(self):
        """Test that order is preserved (first occurrence)."""
        var1 = Mock()
        var2 = Mock()
        var3 = Mock()

        state1 = Mock(spec=State)
        state2 = Mock(spec=State)
        state3 = Mock(spec=State)

        out_var_to_state = {var1: state1, var2: state2, var3: state3}
        out_vars = [var1, var2, var3]

        result = find_out_states(out_var_to_state, out_vars)

        self.assertEqual(result[0], state1)
        self.assertEqual(result[1], state2)
        self.assertEqual(result[2], state3)

    def test_multiple_states(self):
        """Test with multiple different states."""
        vars_list = [Mock() for _ in range(5)]
        states_list = [Mock(spec=State) for _ in range(5)]

        out_var_to_state = {v: s for v, s in zip(vars_list, states_list)}
        out_vars = vars_list

        result = find_out_states(out_var_to_state, out_vars)

        self.assertEqual(len(result), 5)
        for state in states_list:
            self.assertIn(state, result)


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
