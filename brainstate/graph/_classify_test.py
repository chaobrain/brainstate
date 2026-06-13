# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (see repo LICENSE).
import unittest

import jax.numpy as jnp
import numpy as np

import brainstate as bs
from brainstate import graph
from brainstate.graph import _walk
from brainstate.graph._walk import (
    classify, GRAPH_NODE, PYTREE, STATE, STATE_LEAF, STATIC,
    _clear_classification_cache, register_graph_node_type,
)


class Box(graph.Node):
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class TestClassify(unittest.TestCase):
    def setUp(self):
        _clear_classification_cache()

    def test_kinds(self):
        s = bs.ParamState(jnp.ones(3))
        self.assertEqual(classify(Box()), GRAPH_NODE)
        self.assertEqual(classify([1, 2]), PYTREE)
        self.assertEqual(classify({'a': 1}), PYTREE)
        self.assertEqual(classify((1, 2)), PYTREE)
        self.assertEqual(classify(s), STATE)
        self.assertEqual(classify(s.to_state_ref()), PYTREE)   # TreefyState is a registered jax pytree; must classify as PYTREE (checked before STATE_LEAF)
        self.assertEqual(classify(5), STATIC)
        self.assertEqual(classify(None), PYTREE)               # None is an empty pytree
        self.assertEqual(classify(np.ones(2)), STATIC)
        self.assertEqual(classify(jnp.ones(2)), STATIC)

    def test_cache_hit_is_stable(self):
        self.assertEqual(classify([1]), classify([2, 3]))      # same type -> same kind

    def test_cache_invalidated_on_registration(self):
        # Classify a plain class FIRST so it gets cached as STATIC, then
        # register it as a graph node type, and assert that classify now returns
        # GRAPH_NODE — which only happens if registration cleared the cache.
        class WillBeRegistered:
            pass

        # Classified before registration -> STATIC, and cached.
        self.assertEqual(classify(WillBeRegistered()), STATIC)

        # Registering must clear the cache so the stale STATIC entry is dropped.
        register_graph_node_type(
            WillBeRegistered,
            flatten=lambda n: ([], None),
            set_key=lambda n, k, v: None,
            pop_key=lambda n, k: None,
            create_empty=lambda aux: WillBeRegistered(),
            clear=lambda n: None,
        )

        # Must now be GRAPH_NODE, not the stale STATIC.
        self.assertEqual(classify(WillBeRegistered()), GRAPH_NODE)


if __name__ == '__main__':
    unittest.main()
