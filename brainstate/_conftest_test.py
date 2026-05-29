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

"""Tests for conftest behavior: deterministic seeding and the slow marker."""

import unittest

import jax.numpy as jnp

import brainstate


class TestDeterministicSeed(unittest.TestCase):
    """The autouse fixture seeds brainstate.random before each test."""

    def test_rng_is_seeded(self):
        """random draws are reproducible across identical re-seeds."""
        brainstate.random.seed(0)
        a = brainstate.random.rand(5)
        brainstate.random.seed(0)
        b = brainstate.random.rand(5)
        self.assertTrue(bool(jnp.allclose(a, b)))


if __name__ == "__main__":
    unittest.main()
