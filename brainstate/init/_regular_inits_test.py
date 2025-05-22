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

# -*- coding: utf-8 -*-

import unittest

import brainstate


class TestZeroInit(unittest.TestCase):
    def test_zero_init(self):
        init = brainstate.init.ZeroInit()
        for size in [(100,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestOneInit(unittest.TestCase):
    def test_one_init(self):
        for size in [(100,), (10, 20), (10, 20, 30)]:
            for value in [0., 1., -1.]:
                init = brainstate.init.Constant(value=value)
                weights = init(size)
                assert weights.shape == size
                assert (weights == value).all()


class TestIdentityInit(unittest.TestCase):
    def test_identity_init(self):
        for size in [(100,), (10, 20)]:
            for value in [0., 1., -1.]:
                init = brainstate.init.Identity(value=value)
                weights = init(size)
                if len(size) == 1:
                    assert weights.shape == (size[0], size[0])
                else:
                    assert weights.shape == size
