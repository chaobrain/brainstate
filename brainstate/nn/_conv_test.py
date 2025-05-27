# -*- coding: utf-8 -*-

import jax.numpy as jnp
import pytest
from absl.testing import absltest
from absl.testing import parameterized

import brainstate


class TestConv(parameterized.TestCase):
    def test_Conv2D_img(self):
        img = jnp.zeros((2, 200, 198, 4))
        for k in range(4):
            x = 30 + 60 * k
            y = 20 + 60 * k
            img = img.at[0, x:x + 10, y:y + 10, k].set(1.0)
            img = img.at[1, x:x + 20, y:y + 20, k].set(3.0)

        net = brainstate.nn.Conv2d((200, 198, 4), out_channels=32, kernel_size=(3, 3),
                                   stride=(2, 1), padding='VALID', groups=4)
        out = net(img)
        print("out shape: ", out.shape)
        self.assertEqual(out.shape, (2, 99, 196, 32))
        # print("First output channel:")
        # plt.figure(figsize=(10, 10))
        # plt.imshow(np.array(img)[0, :, :, 0])
        # plt.show()

    def test_conv1D(self):
        model = brainstate.nn.Conv1d((5, 3), out_channels=32, kernel_size=(3,))
        input = jnp.ones((2, 5, 3))
        out = model(input)
        print("out shape: ", out.shape)
        self.assertEqual(out.shape, (2, 5, 32))
        # print("First output channel:")
        # plt.figure(figsize=(10, 10))
        # plt.imshow(np.array(out)[0, :, :])
        # plt.show()

    def test_conv2D(self):
        model = brainstate.nn.Conv2d((5, 5, 3), out_channels=32, kernel_size=(3, 3))
        input = jnp.ones((2, 5, 5, 3))

        out = model(input)
        print("out shape: ", out.shape)
        self.assertEqual(out.shape, (2, 5, 5, 32))

    def test_conv3D(self):
        model = brainstate.nn.Conv3d((5, 5, 5, 3), out_channels=32, kernel_size=(3, 3, 3))
        input = jnp.ones((2, 5, 5, 5, 3))
        out = model(input)
        print("out shape: ", out.shape)
        self.assertEqual(out.shape, (2, 5, 5, 5, 32))


@pytest.mark.skip(reason="not implemented yet")
class TestConvTranspose1d(parameterized.TestCase):
    def test_conv_transpose(self):

        x = jnp.ones((1, 8, 3))
        for use_bias in [True, False]:
            conv_transpose_module = brainstate.nn.ConvTranspose1d(
                in_channels=3,
                out_channels=4,
                kernel_size=(3,),
                padding='VALID',
                w_initializer=brainstate.init.Constant(1.),
                b_initializer=brainstate.init.Constant(1.) if use_bias else None,
            )
            self.assertEqual(conv_transpose_module.w.shape, (3, 3, 4))
            y = conv_transpose_module(x)
            print(y.shape)
            correct_ans = jnp.array([[[4., 4., 4., 4.],
                                      [7., 7., 7., 7.],
                                      [10., 10., 10., 10.],
                                      [10., 10., 10., 10.],
                                      [10., 10., 10., 10.],
                                      [10., 10., 10., 10.],
                                      [10., 10., 10., 10.],
                                      [10., 10., 10., 10.],
                                      [7., 7., 7., 7.],
                                      [4., 4., 4., 4.]]])
            if not use_bias:
                correct_ans -= 1.
            self.assertTrue(jnp.allclose(y, correct_ans))

    def test_single_input_masked_conv_transpose(self):

        x = jnp.ones((1, 8, 3))
        m = jnp.tril(jnp.ones((3, 3, 4)))
        conv_transpose_module = brainstate.nn.ConvTranspose1d(
            in_channels=3,
            out_channels=4,
            kernel_size=(3,),
            padding='VALID',
            mask=m,
            w_initializer=brainstate.init.Constant(),
            b_initializer=brainstate.init.Constant(),
        )
        self.assertEqual(conv_transpose_module.w.shape, (3, 3, 4))
        y = conv_transpose_module(x)
        print(y.shape)
        correct_ans = jnp.array([[[4., 3., 2., 1.],
                                  [7., 5., 3., 1.],
                                  [10., 7., 4., 1.],
                                  [10., 7., 4., 1.],
                                  [10., 7., 4., 1.],
                                  [10., 7., 4., 1.],
                                  [10., 7., 4., 1.],
                                  [10., 7., 4., 1.],
                                  [7., 5., 3., 1.],
                                  [4., 3., 2., 1.]]])
        self.assertTrue(jnp.allclose(y, correct_ans))

    def test_computation_padding_same(self):

        data = jnp.ones([1, 3, 1])
        for use_bias in [True, False]:
            net = brainstate.nn.ConvTranspose1d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding="SAME",
                w_initializer=brainstate.init.Constant(),
                b_initializer=brainstate.init.Constant() if use_bias else None,
            )
            out = net(data)
            self.assertEqual(out.shape, (1, 3, 1))
            out = jnp.squeeze(out, axis=(0, 2))
            expected_out = jnp.asarray([2, 3, 2])
            if use_bias:
                expected_out += 1
            self.assertTrue(jnp.allclose(out, expected_out, rtol=1e-5))


@pytest.mark.skip(reason="not implemented yet")
class TestConvTranspose2d(parameterized.TestCase):
    def test_conv_transpose(self):

        x = jnp.ones((1, 8, 8, 3))
        for use_bias in [True, False]:
            conv_transpose_module = brainstate.nn.ConvTranspose2d(
                in_channels=3,
                out_channels=4,
                kernel_size=(3, 3),
                padding='VALID',
                w_initializer=brainstate.init.Constant(),
                b_initializer=brainstate.init.Constant() if use_bias else None,
            )
        self.assertEqual(conv_transpose_module.w.shape, (3, 3, 3, 4))
        y = conv_transpose_module(x)
        print(y.shape)

    def test_single_input_masked_conv_transpose(self):

        x = jnp.ones((1, 8, 8, 3))
        m = jnp.tril(jnp.ones((3, 3, 3, 4)))
        conv_transpose_module = brainstate.nn.ConvTranspose2d(
            in_channels=3,
            out_channels=4,
            kernel_size=(3, 3),
            padding='VALID',
            mask=m,
            w_initializer=brainstate.init.Constant(),
        )
        y = conv_transpose_module(x)
        print(y.shape)

    def test_computation_padding_same(self):

        x = jnp.ones((1, 8, 8, 3))
        for use_bias in [True, False]:
            conv_transpose_module = brainstate.nn.ConvTranspose2d(
                in_channels=3,
                out_channels=4,
                kernel_size=(3, 3),
                stride=1,
                padding='SAME',
                w_initializer=brainstate.init.Constant(),
                b_initializer=brainstate.init.Constant() if use_bias else None,
            )
        y = conv_transpose_module(x)
        print(y.shape)


@pytest.mark.skip(reason="not implemented yet")
class TestConvTranspose3d(parameterized.TestCase):
    def test_conv_transpose(self):

        x = jnp.ones((1, 8, 8, 8, 3))
        for use_bias in [True, False]:
            conv_transpose_module = brainstate.nn.ConvTranspose3d(
                in_channels=3,
                out_channels=4,
                kernel_size=(3, 3, 3),
                padding='VALID',
                w_initializer=brainstate.init.Constant(),
                b_initializer=brainstate.init.Constant() if use_bias else None,
            )
        y = conv_transpose_module(x)
        print(y.shape)

    def test_single_input_masked_conv_transpose(self):

        x = jnp.ones((1, 8, 8, 8, 3))
        m = jnp.tril(jnp.ones((3, 3, 3, 3, 4)))
        conv_transpose_module = brainstate.nn.ConvTranspose3d(
            in_channels=3,
            out_channels=4,
            kernel_size=(3, 3, 3),
            padding='VALID',
            mask=m,
            w_initializer=brainstate.init.Constant(),
        )
        y = conv_transpose_module(x)
        print(y.shape)

    def test_computation_padding_same(self):

        x = jnp.ones((1, 8, 8, 8, 3))
        for use_bias in [True, False]:
            conv_transpose_module = brainstate.nn.ConvTranspose3d(
                in_channels=3,
                out_channels=4,
                kernel_size=(3, 3, 3),
                stride=1,
                padding='SAME',
                w_initializer=brainstate.init.Constant(),
                b_initializer=brainstate.init.Constant() if use_bias else None,
            )
        y = conv_transpose_module(x)
        print(y.shape)


if __name__ == '__main__':
    absltest.main()
