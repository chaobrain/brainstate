# -*- coding: utf-8 -*-

import jax
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import brainstate
import brainstate.nn as nn


class TestFlatten(parameterized.TestCase):
    def test_flatten1(self):
        for size in [
            (16, 32, 32, 8),
            (32, 8),
            (10, 20, 30),
        ]:
            arr = brainstate.random.rand(*size)
            f = nn.Flatten(start_axis=0)
            out = f(arr)
            self.assertTrue(out.shape == (np.prod(size),))

    def test_flatten2(self):
        for size in [
            (16, 32, 32, 8),
            (32, 8),
            (10, 20, 30),
        ]:
            arr = brainstate.random.rand(*size)
            f = nn.Flatten(start_axis=1)
            out = f(arr)
            self.assertTrue(out.shape == (size[0], np.prod(size[1:])))

    def test_flatten3(self):
        size = (16, 32, 32, 8)
        arr = brainstate.random.rand(*size)
        f = nn.Flatten(start_axis=0, in_size=(32, 8))
        out = f(arr)
        self.assertTrue(out.shape == (16, 32, 32 * 8))

    def test_flatten4(self):
        size = (16, 32, 32, 8)
        arr = brainstate.random.rand(*size)
        f = nn.Flatten(start_axis=1, in_size=(32, 32, 8))
        out = f(arr)
        self.assertTrue(out.shape == (16, 32, 32 * 8))


class TestUnflatten(parameterized.TestCase):
    pass


class TestPool(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_MaxPool2d_v1(self):
        arr = brainstate.random.rand(16, 32, 32, 8)

        out = nn.MaxPool2d(2, 2, channel_axis=-1)(arr)
        self.assertTrue(out.shape == (16, 16, 16, 8))

        out = nn.MaxPool2d(2, 2, channel_axis=None)(arr)
        self.assertTrue(out.shape == (16, 32, 16, 4))

        out = nn.MaxPool2d(2, 2, channel_axis=None, padding=1)(arr)
        self.assertTrue(out.shape == (16, 32, 17, 5))

        out = nn.MaxPool2d(2, 2, channel_axis=None, padding=(2, 1))(arr)
        self.assertTrue(out.shape == (16, 32, 18, 5))

        out = nn.MaxPool2d(2, 2, channel_axis=-1, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 17, 8))

        out = nn.MaxPool2d(2, 2, channel_axis=2, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 32, 5))

    def test_AvgPool2d_v1(self):
        arr = brainstate.random.rand(16, 32, 32, 8)

        out = nn.AvgPool2d(2, 2, channel_axis=-1)(arr)
        self.assertTrue(out.shape == (16, 16, 16, 8))

        out = nn.AvgPool2d(2, 2, channel_axis=None)(arr)
        self.assertTrue(out.shape == (16, 32, 16, 4))

        out = nn.AvgPool2d(2, 2, channel_axis=None, padding=1)(arr)
        self.assertTrue(out.shape == (16, 32, 17, 5))

        out = nn.AvgPool2d(2, 2, channel_axis=None, padding=(2, 1))(arr)
        self.assertTrue(out.shape == (16, 32, 18, 5))

        out = nn.AvgPool2d(2, 2, channel_axis=-1, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 17, 8))

        out = nn.AvgPool2d(2, 2, channel_axis=2, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 32, 5))

    @parameterized.named_parameters(
        dict(testcase_name=f'target_size={target_size}',
             target_size=target_size)
        for target_size in [10, 9, 8, 7, 6]
    )
    def test_adaptive_pool1d(self, target_size):
        from brainstate.nn._poolings import _adaptive_pool1d

        arr = brainstate.random.rand(100)
        op = jax.numpy.mean

        out = _adaptive_pool1d(arr, target_size, op)
        print(out.shape)
        self.assertTrue(out.shape == (target_size,))

        out = _adaptive_pool1d(arr, target_size, op)
        print(out.shape)
        self.assertTrue(out.shape == (target_size,))

    def test_AdaptiveAvgPool2d_v1(self):
        input = brainstate.random.randn(64, 8, 9)

        output = nn.AdaptiveAvgPool2d((5, 7), channel_axis=0)(input)
        self.assertTrue(output.shape == (64, 5, 7))

        output = nn.AdaptiveAvgPool2d((2, 3), channel_axis=0)(input)
        self.assertTrue(output.shape == (64, 2, 3))

        output = nn.AdaptiveAvgPool2d((2, 3), channel_axis=-1)(input)
        self.assertTrue(output.shape == (2, 3, 9))

        output = nn.AdaptiveAvgPool2d((2, 3), channel_axis=1)(input)
        self.assertTrue(output.shape == (2, 8, 3))

        output = nn.AdaptiveAvgPool2d((2, 3), channel_axis=None)(input)
        self.assertTrue(output.shape == (64, 2, 3))

    def test_AdaptiveAvgPool2d_v2(self):
        brainstate.random.seed()
        input = brainstate.random.randn(128, 64, 32, 16)

        output = nn.AdaptiveAvgPool2d((5, 7), channel_axis=0)(input)
        self.assertTrue(output.shape == (128, 64, 5, 7))

        output = nn.AdaptiveAvgPool2d((2, 3), channel_axis=0)(input)
        self.assertTrue(output.shape == (128, 64, 2, 3))

        output = nn.AdaptiveAvgPool2d((2, 3), channel_axis=-1)(input)
        self.assertTrue(output.shape == (128, 2, 3, 16))

        output = nn.AdaptiveAvgPool2d((2, 3), channel_axis=1)(input)
        self.assertTrue(output.shape == (128, 64, 2, 3))
        print()

    def test_AdaptiveAvgPool3d_v1(self):
        input = brainstate.random.randn(10, 128, 64, 32)
        net = nn.AdaptiveAvgPool3d(target_size=[6, 5, 3], channel_axis=0)
        output = net(input)
        self.assertTrue(output.shape == (10, 6, 5, 3))

    def test_AdaptiveAvgPool3d_v2(self):
        input = brainstate.random.randn(10, 20, 128, 64, 32)
        net = nn.AdaptiveAvgPool3d(target_size=[6, 5, 3])
        output = net(input)
        self.assertTrue(output.shape == (10, 6, 5, 3, 32))

    @parameterized.product(
        axis=(-1, 0, 1)
    )
    def test_AdaptiveMaxPool1d_v1(self, axis):
        input = brainstate.random.randn(32, 16)
        net = nn.AdaptiveMaxPool1d(target_size=4, channel_axis=axis)
        output = net(input)

    @parameterized.product(
        axis=(-1, 0, 1, 2)
    )
    def test_AdaptiveMaxPool1d_v2(self, axis):
        input = brainstate.random.randn(2, 32, 16)
        net = nn.AdaptiveMaxPool1d(target_size=4, channel_axis=axis)
        output = net(input)

    @parameterized.product(
        axis=(-1, 0, 1, 2)
    )
    def test_AdaptiveMaxPool2d_v1(self, axis):
        input = brainstate.random.randn(32, 16, 12)
        net = nn.AdaptiveAvgPool2d(target_size=[5, 4], channel_axis=axis)
        output = net(input)

    @parameterized.product(
        axis=(-1, 0, 1, 2, 3)
    )
    def test_AdaptiveMaxPool2d_v2(self, axis):
        input = brainstate.random.randn(2, 32, 16, 12)
        net = nn.AdaptiveAvgPool2d(target_size=[5, 4], channel_axis=axis)
        output = net(input)

    @parameterized.product(
        axis=(-1, 0, 1, 2, 3)
    )
    def test_AdaptiveMaxPool3d_v1(self, axis):
        input = brainstate.random.randn(2, 128, 64, 32)
        net = nn.AdaptiveMaxPool3d(target_size=[6, 5, 4], channel_axis=axis)
        output = net(input)
        print()

    @parameterized.product(
        axis=(-1, 0, 1, 2, 3, 4)
    )
    def test_AdaptiveMaxPool3d_v1(self, axis):
        input = brainstate.random.randn(2, 128, 64, 32, 16)
        net = nn.AdaptiveMaxPool3d(target_size=[6, 5, 4], channel_axis=axis)
        output = net(input)


if __name__ == '__main__':
    absltest.main()
