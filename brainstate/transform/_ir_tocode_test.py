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

import unittest

import numpy as np
import jax
import jax.numpy as jnp

from brainstate.transform import fn_to_python_code, jaxpr_to_python_code


def _exec_generated(source, fn_name):
    """Exec generated code in a controlled namespace and return the function."""
    ns = {'jax': jax, 'jnp': jnp, 'np': np}
    exec(source, ns)
    for candidate in (fn_name, 'generated_function', 'unknown'):
        if candidate in ns and callable(ns[candidate]):
            return ns[candidate]
    # Fallback: the last def'd function in the namespace.
    funcs = [v for v in ns.values()
             if callable(v) and getattr(v, '__name__', '') not in ('', None)
             and v not in (jax, jnp, np)]
    if funcs:
        return funcs[-1]
    raise AssertionError("no generated function found in namespace")


def _roundtrip_check(testcase, f, *args, fn_name=None):
    fn_name = fn_name or getattr(f, '__name__', 'generated_function')
    if not fn_name.isidentifier():
        fn_name = 'generated_function'
    src = fn_to_python_code(f, *args)
    gen = _exec_generated(src, fn_name)
    got = gen(*args)
    ref = f(*args)
    got = got if isinstance(got, (tuple, list)) else (got,)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    testcase.assertEqual(len(got), len(ref))
    for a, b in zip(got, ref):
        testcase.assertTrue(
            np.allclose(np.asarray(a), np.asarray(b), equal_nan=True),
            f"mismatch in {fn_name}: {np.asarray(a)} vs {np.asarray(b)}\nsource:\n{src}",
        )


class TestToCodeRoundTrip(unittest.TestCase):
    def test_arithmetic(self):
        def f(x, y):
            return (x + y) * (x - y) / (y + 1.0)
        _roundtrip_check(self, f, jnp.float32(3.0), jnp.float32(2.0))

    def test_multiple_outputs(self):
        def f(x):
            return x + 1.0, x * 2.0
        _roundtrip_check(self, f, jnp.float32(4.0))

    def test_reduction_sum(self):
        def f(x):
            return jnp.sum(x)
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))

    def test_matmul(self):
        def f(a, b):
            return a @ b
        _roundtrip_check(self, f, jnp.float32([[1., 2.], [3., 4.]]), jnp.float32([[5.], [6.]]))


class TestToCodeExpandedPrimitives(unittest.TestCase):
    def test_unary_math(self):
        for op in (jnp.exp, jnp.log, jnp.sin, jnp.cos, jnp.tanh, jnp.sqrt, jnp.abs):
            with self.subTest(op=op.__name__):
                _roundtrip_check(self, (lambda x, _op=op: _op(x)), jnp.float32([0.5, 1.5]),
                                 fn_name='f')

    def test_where(self):
        def f(x):
            return jnp.where(x > 0, x, -x)
        _roundtrip_check(self, f, jnp.float32([-1., 2., -3.]))

    def test_concatenate(self):
        def f(a, b):
            return jnp.concatenate([a, b])
        _roundtrip_check(self, f, jnp.float32([1., 2.]), jnp.float32([3., 4.]))

    def test_reductions(self):
        for r in (jnp.max, jnp.min, jnp.prod):
            with self.subTest(r=r.__name__):
                _roundtrip_check(self, (lambda x, _r=r: _r(x)), jnp.float32([1., 2., 3.]),
                                 fn_name='f')

    def test_argmax(self):
        def f(x):
            return jnp.argmax(x)
        _roundtrip_check(self, f, jnp.float32([1., 3., 2.]))

    def test_integer_pow(self):
        def f(x):
            return x ** 3
        _roundtrip_check(self, f, jnp.float32(2.0))

    def test_expand_dims(self):
        def f(x):
            return jnp.expand_dims(x, 0)
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))

    def test_cumsum(self):
        def f(x):
            return jnp.cumsum(x)
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))


class TestToCodeEdgeCases(unittest.TestCase):
    def test_empty_jaxpr_generates_valid_code(self):
        def f(x):
            return x
        src = fn_to_python_code(f, jnp.float32(1.0))
        gen = _exec_generated(src, 'f')
        self.assertTrue(np.allclose(np.asarray(gen(jnp.float32(5.0))), 5.0))

    def test_lambda_without_name(self):
        f = lambda x: x + 1.0
        src = fn_to_python_code(f, jnp.float32(1.0))
        self.assertIsInstance(src, str)
        self.assertTrue(len(src) > 0)
        gen = _exec_generated(src, 'generated_function')
        self.assertTrue(np.allclose(np.asarray(gen(jnp.float32(2.0))), 3.0))


class TestToCodeUnsupported(unittest.TestCase):
    def test_unknown_primitive_raises(self):
        from brainstate.transform._ir_utils import UnsupportedPrimitiveError
        # Use a primitive with no registered handler.
        from brainstate.transform import _ir_tocode
        if 'cumlogsumexp' in _ir_tocode.prim_to_python:
            self.skipTest("cumlogsumexp now registered; pick another unregistered primitive")

        def f(x):
            return jax.lax.cumlogsumexp(x)

        with self.assertRaises(UnsupportedPrimitiveError):
            fn_to_python_code(f, jnp.float32([1., 2., 3.]))


class TestRegisterPrimHandlerPublic(unittest.TestCase):
    def test_register_prim_handler_is_public(self):
        import brainstate.transform as T
        self.assertTrue(hasattr(T, 'register_prim_handler'))
        self.assertTrue(callable(T.register_prim_handler))


class TestToCodeControlFlow(unittest.TestCase):
    def test_pjit_nested(self):
        @jax.jit
        def inner(x):
            return x * x

        def f(x):
            return inner(x) + 1.0
        _roundtrip_check(self, f, jnp.float32(3.0))

    def test_scan(self):
        def f(x):
            def body(c, _):
                return c + 1.0, c
            final, ys = jax.lax.scan(body, x, None, length=4)
            return final
        _roundtrip_check(self, f, jnp.float32(0.0))


if __name__ == '__main__':
    unittest.main()
