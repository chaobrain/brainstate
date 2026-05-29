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


from unittest import TestCase

import jax
import jax.numpy as jnp

import brainstate


class TestWhileLoop(TestCase):
    def test1(self):
        a = brainstate.State(1.)
        b = brainstate.State(20.)

        def cond(_):
            return a.value < b.value

        def body(_):
            a.value += 1.

        brainstate.transform.while_loop(cond, body, None)

        print(a.value, b.value)

    def test2(self):
        a = brainstate.State(1.)
        b = brainstate.State(20.)

        def cond(x):
            return a.value < b.value

        def body(x):
            a.value += x
            return x

        r = brainstate.transform.while_loop(cond, body, 1.)

        print(a.value, b.value, r)


class TestWhileLoopCoverage(TestCase):
    """Validation, the disable-jit fast path, and the cond-write guard."""

    def test_non_callable_arguments_raise(self):
        """Non-callable ``cond_fun``/``body_fun`` raise ``TypeError``."""
        with self.assertRaises(TypeError):
            brainstate.transform.while_loop(1, lambda v: v, 0.0)

    def test_disable_jit_runs_python_loop(self):
        """Under ``disable_jit`` the loop runs eagerly in Python."""
        a = brainstate.State(jnp.array(0.0))

        def cond(v):
            return a.value < 5

        def body(v):
            a.value = a.value + 1.0
            return v

        with jax.disable_jit():
            brainstate.transform.while_loop(cond, body, 0.0)
        self.assertEqual(int(a.value), 5)

    def test_cond_writing_state_raises(self):
        """A ``cond_fun`` that writes a state raises ``ValueError``."""
        a = brainstate.State(jnp.array(0.0))

        def cond(v):
            a.value = a.value + 1.0  # illegal write in the condition
            return a.value < 5

        def body(v):
            return v

        with self.assertRaises(ValueError):
            brainstate.transform.while_loop(cond, body, 0.0)


class TestBoundedWhileLoop(TestCase):
    """``bounded_while_loop`` validation and execution with state."""

    def test_negative_max_steps_raises(self):
        """A negative ``max_steps`` raises ``ValueError``."""
        with self.assertRaises(ValueError):
            brainstate.transform.bounded_while_loop(
                lambda v: True, lambda v: v, 0.0, max_steps=-1
            )

    def test_zero_max_steps_returns_init(self):
        """``max_steps=0`` returns the (arrayified) initial value unchanged."""
        out = brainstate.transform.bounded_while_loop(
            lambda v: v < 10, lambda v: v + 1, 3.0, max_steps=0
        )
        self.assertTrue(bool(jnp.allclose(out, 3.0)))

    def test_runs_until_condition_false(self):
        """The loop advances state until the condition is false (within bound)."""
        i = brainstate.State(jnp.array(0))

        def cond(v):
            return i.value < 5

        def body(v):
            i.value = i.value + 1
            return v

        brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=32)
        self.assertEqual(int(i.value), 5)

    def test_max_steps_caps_iterations(self):
        """The loop stops at ``max_steps`` even if the condition stays true."""
        i = brainstate.State(jnp.array(0))

        def cond(v):
            return i.value < 1000  # never satisfied within the bound

        def body(v):
            i.value = i.value + 1
            return v

        brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=4, base=2)
        self.assertLessEqual(int(i.value), 4)

    def test_cond_writing_state_raises(self):
        """A ``cond_fun`` that writes a state raises ``ValueError``."""
        a = brainstate.State(jnp.array(0.0))

        def cond(v):
            a.value = a.value + 1.0
            return a.value < 5

        def body(v):
            return v

        with self.assertRaises(ValueError):
            brainstate.transform.bounded_while_loop(cond, body, 0.0, max_steps=10)
