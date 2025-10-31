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

# -*- coding: utf-8 -*-

import unittest

import brainstate


class TestNormalInit(unittest.TestCase):

    def test_normal_init1(self):
        init = brainstate.nn.init.Normal()
        for size in [(10,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_normal_init2(self):
        init = brainstate.nn.init.Normal(scale=0.5)
        for size in [(10,), (10, 20)]:
            weights = init(size)
            assert weights.shape == size

    def test_normal_init3(self):
        init1 = brainstate.nn.init.Normal(scale=0.5, seed=10)
        init2 = brainstate.nn.init.Normal(scale=0.5, seed=10)
        size = (10,)
        weights1 = init1(size)
        weights2 = init2(size)
        assert weights1.shape == size
        assert (weights1 == weights2).all()


class TestUniformInit(unittest.TestCase):
    def test_uniform_init1(self):
        init = brainstate.nn.init.Normal()
        for size in [(10,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_uniform_init2(self):
        init = brainstate.nn.init.Uniform(min_val=10, max_val=20)
        for size in [(10,), (10, 20)]:
            weights = init(size)
            assert weights.shape == size


class TestVarianceScaling(unittest.TestCase):
    def test_var_scaling1(self):
        init = brainstate.nn.init.VarianceScaling(scale=1., mode='fan_in', distribution='truncated_normal')
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_var_scaling2(self):
        init = brainstate.nn.init.VarianceScaling(scale=2, mode='fan_out', distribution='normal')
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_var_scaling3(self):
        init = brainstate.nn.init.VarianceScaling(
            scale=2 / 4, mode='fan_avg', in_axis=0, out_axis=1, distribution='uniform'
        )
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestKaimingUniformUnit(unittest.TestCase):
    def test_kaiming_uniform_init(self):
        init = brainstate.nn.init.KaimingUniform()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestKaimingNormalUnit(unittest.TestCase):
    def test_kaiming_normal_init(self):
        init = brainstate.nn.init.KaimingNormal()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestXavierUniformUnit(unittest.TestCase):
    def test_xavier_uniform_init(self):
        init = brainstate.nn.init.XavierUniform()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestXavierNormalUnit(unittest.TestCase):
    def test_xavier_normal_init(self):
        init = brainstate.nn.init.XavierNormal()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestLecunUniformUnit(unittest.TestCase):
    def test_lecun_uniform_init(self):
        init = brainstate.nn.init.LecunUniform()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestLecunNormalUnit(unittest.TestCase):
    def test_lecun_normal_init(self):
        init = brainstate.nn.init.LecunNormal()
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestOrthogonalUnit(unittest.TestCase):
    def test_orthogonal_init1(self):
        init = brainstate.nn.init.Orthogonal()
        for size in [(20, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size

    def test_orthogonal_init2(self):
        init = brainstate.nn.init.Orthogonal(scale=2., axis=0)
        for size in [(10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestDeltaOrthogonalUnit(unittest.TestCase):
    def test_delta_orthogonal_init1(self):
        init = brainstate.nn.init.DeltaOrthogonal()
        for size in [(20, 20, 20), (10, 20, 30, 40), (50, 40, 30, 20, 20)]:
            weights = init(size)
            assert weights.shape == size


class TestZeroInit(unittest.TestCase):
    def test_zero_init(self):
        init = brainstate.nn.init.ZeroInit()
        for size in [(10,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size


class TestOneInit(unittest.TestCase):
    def test_one_init(self):
        for size in [(10,), (10, 20), (10, 20, 30)]:
            for value in [0., 1., -1.]:
                init = brainstate.nn.init.Constant(value=value)
                weights = init(size)
                assert weights.shape == size
                assert (weights == value).all()


class TestIdentityInit(unittest.TestCase):
    def test_identity_init(self):
        for size in [(10,), (10, 20)]:
            for value in [0., 1., -1.]:
                init = brainstate.nn.init.Identity(value=value)
                weights = init(size)
                if len(size) == 1:
                    assert weights.shape == (size[0], size[0])
                else:
                    assert weights.shape == size
