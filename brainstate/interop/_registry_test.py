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

"""Unit tests for the framework-agnostic conversion registry."""

from absl.testing import absltest

from brainstate.interop import _registry as R
from brainstate.interop._registry import (LayerMapping, lookup_export,
                                          lookup_import,
                                          register_layer_mapping,
                                          register_unsupported_bst,
                                          register_unsupported_bst_for,
                                          register_unsupported_foreign,
                                          unsupported_bst_reason,
                                          unsupported_foreign_reason)


class _BstBase:
    pass


class _BstLayer(_BstBase):
    pass


class _ForeignBase:
    pass


class _ForeignLayer(_ForeignBase):
    pass


class _ForeignSubclass(_ForeignLayer):
    pass


class RegistryTest(absltest.TestCase):
    """Registration and lookup, using a private throwaway framework name."""

    FW = '_test_fw_registry'

    def tearDown(self):
        # Keep global registry state clean across tests.
        R._IMPORT.pop(self.FW, None)
        R._UNSUPPORTED_FOREIGN.pop(self.FW, None)
        for key in list(R._EXPORT):
            if key[1] == self.FW:
                R._EXPORT.pop(key)
        for key in list(R._UNSUPPORTED_BST_FW):
            if key[1] == self.FW:
                R._UNSUPPORTED_BST_FW.pop(key)
        R._UNSUPPORTED_BST.pop(_BstLayer, None)
        super().tearDown()

    def test_register_and_lookup_both_directions(self):
        m = LayerMapping(_BstLayer, self.FW, _ForeignLayer,
                         to_bst=lambda n, c: 'to_bst', to_foreign=lambda n, c: 'to_foreign')
        register_layer_mapping(m)
        self.assertIs(lookup_import(self.FW, _ForeignLayer), m)
        self.assertIs(lookup_export(_BstLayer, self.FW), m)

    def test_import_lookup_is_mro_aware(self):
        m = LayerMapping(_BstLayer, self.FW, _ForeignLayer,
                         to_bst=lambda n, c: None, to_foreign=lambda n, c: None)
        register_layer_mapping(m)
        # A subclass of the registered foreign type resolves to the base mapping.
        self.assertIs(lookup_import(self.FW, _ForeignSubclass), m)

    def test_export_lookup_is_mro_aware(self):
        m = LayerMapping(_BstBase, self.FW, _ForeignLayer,
                         to_bst=lambda n, c: None, to_foreign=lambda n, c: None)
        register_layer_mapping(m)
        # _BstLayer subclasses _BstBase -> resolves to base mapping.
        self.assertIs(lookup_export(_BstLayer, self.FW), m)

    def test_lookup_miss_returns_none(self):
        self.assertIsNone(lookup_import(self.FW, _ForeignLayer))
        self.assertIsNone(lookup_export(_BstLayer, self.FW))

    def test_unsupported_foreign(self):
        register_unsupported_foreign(self.FW, _ForeignLayer, 'nope-foreign')
        self.assertEqual(unsupported_foreign_reason(self.FW, _ForeignLayer), 'nope-foreign')
        # MRO-aware: subclass inherits the reason.
        self.assertEqual(unsupported_foreign_reason(self.FW, _ForeignSubclass), 'nope-foreign')
        self.assertIsNone(unsupported_foreign_reason(self.FW, _BstLayer))

    def test_unsupported_bst_framework_specific_and_agnostic(self):
        register_unsupported_bst_for(self.FW, _BstLayer, 'fw-specific')
        self.assertEqual(unsupported_bst_reason(_BstLayer, self.FW), 'fw-specific')
        # Without a framework, the framework-specific entry does not apply.
        self.assertIsNone(unsupported_bst_reason(_BstLayer))
        # Agnostic entry applies to any framework and to no-framework queries.
        register_unsupported_bst(_BstLayer, 'agnostic')
        self.assertEqual(unsupported_bst_reason(_BstLayer), 'agnostic')
        # framework-specific takes precedence when a framework is given.
        self.assertEqual(unsupported_bst_reason(_BstLayer, self.FW), 'fw-specific')

    def test_register_overrides_existing(self):
        m1 = LayerMapping(_BstLayer, self.FW, _ForeignLayer,
                          to_bst=lambda n, c: 1, to_foreign=lambda n, c: 1)
        m2 = LayerMapping(_BstLayer, self.FW, _ForeignLayer,
                          to_bst=lambda n, c: 2, to_foreign=lambda n, c: 2)
        register_layer_mapping(m1)
        register_layer_mapping(m2)
        self.assertIs(lookup_import(self.FW, _ForeignLayer), m2)
        self.assertIs(lookup_export(_BstLayer, self.FW), m2)


if __name__ == '__main__':
    absltest.main()
