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

import unittest

import jax
import jax.numpy as jnp

import brainstate


class TestJitError(unittest.TestCase):
    def test1(self):
        with self.assertRaises(Exception):
            brainstate.transform.jit_error_if(True, 'error')

        def err_f(x):
            raise ValueError(f'error: {x}')

        brainstate.transform.jit_error_if(False, err_f, 1.)
        with self.assertRaises(Exception):
            brainstate.transform.jit_error_if(True, err_f, 1.)

    def test_vmap(self):
        def f(x):
            brainstate.transform.jit_error_if(x, 'error: {x}', x=x)

        jax.vmap(f)(jnp.array([False, False, False]))
        with self.assertRaises(Exception):
            jax.vmap(f)(jnp.array([True, False, False]))

    def test_vmap_vmap(self):
        def f(x):
            brainstate.transform.jit_error_if(x, 'error: {x}', x=x)

        jax.vmap(jax.vmap(f))(jnp.array([[False, False, False],
                                         [False, False, False]]))
        with self.assertRaises(Exception):
            jax.vmap(jax.vmap(f))(jnp.array([[False, False, False],
                                             [True, False, False]]))


class TestErrorMsgDirect(unittest.TestCase):
    def test_positional_format_args_covers_line_40(self):
        """_error_msg with positional args exercises the %-format branch (line 40)."""
        from brainstate.transform._error_if import _error_msg
        with self.assertRaises(ValueError) as ctx:
            _error_msg('value is %d', 42)
        self.assertIn('42', str(ctx.exception))

    def test_kwargs_format_covers_line_42(self):
        """_error_msg with kwargs exercises the .format branch (line 42)."""
        from brainstate.transform._error_if import _error_msg
        with self.assertRaises(ValueError) as ctx:
            _error_msg('value is {x}', x=99)
        self.assertIn('99', str(ctx.exception))

    def test_no_args_no_kwargs_raises_plain(self):
        """_error_msg with no format args raises with the plain message (line 43)."""
        from brainstate.transform._error_if import _error_msg
        with self.assertRaises(ValueError) as ctx:
            _error_msg('plain message')
        self.assertIn('plain message', str(ctx.exception))
