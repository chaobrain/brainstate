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

import ast
import enum
import unittest
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from brainstate.transform import fn_to_python_code, jaxpr_to_python_code
from brainstate.transform import _ir_tocode


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


class TestBoxedLiteralArrays(unittest.TestCase):
    """Some JAX versions box jaxpr literals in wrapper types (e.g.
    ``jax._src.literals.TypedNdArray``) that are array-like but not
    ``jax.Array`` subclasses. These must still emit valid code instead of
    falling through to the 'unknown value' path (which produced syntactically
    invalid output like ``b + ``)."""

    class _FakeTypedNdArray:
        # Mimics the public surface of jax's literal wrapper.
        def __init__(self, val):
            self.val = np.asarray(val)

        @property
        def dtype(self):
            return self.val.dtype

        @property
        def shape(self):
            return self.val.shape

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self.val, dtype=dtype)

    def test_is_array_recognizes_wrapper(self):
        from brainstate.transform import _ir_tocode
        w = self._FakeTypedNdArray(np.float32(2.0))
        self.assertTrue(_ir_tocode.is_array(w))
        # A dtype object must NOT be misclassified as an array.
        self.assertFalse(_ir_tocode.is_array(np.dtype('float32')))

    def test_scalar_wrapper_emits_valid_constant(self):
        import ast
        from brainstate.transform import _ir_tocode
        w = self._FakeTypedNdArray(np.float32(2.5))
        node = _ir_tocode._astify_value(w)
        # Must unparse to a non-empty, parseable expression equal to 2.5.
        src = ast.unparse(ast.fix_missing_locations(node))
        self.assertTrue(src.strip())
        self.assertEqual(eval(src), 2.5)

    def test_array_wrapper_emits_valid_array(self):
        import ast
        from brainstate.transform import _ir_tocode
        w = self._FakeTypedNdArray(np.float32([1.0, 2.0, 3.0]))
        node = _ir_tocode._astify_value(w)
        src = ast.unparse(ast.fix_missing_locations(node))
        self.assertIn('array', src)
        out = eval(src, {'jax': jax, 'jnp': jnp, 'np': np})
        self.assertTrue(np.allclose(np.asarray(out), [1.0, 2.0, 3.0]))


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


def _unparse(node):
    """Unparse an AST node (or single statement) to source for assertions."""
    return ast.unparse(ast.fix_missing_locations(node))


class _Color(enum.Enum):
    """Module-level enum so its ``__qualname__`` is just the class name."""
    RED = 1
    GREEN = 2


class TestSlicingHandlers(unittest.TestCase):
    """Round-trip the slice family of primitive handlers."""

    def test_slice_with_strides(self):
        """``jax.lax.slice`` -> ``x[start:stop:step]`` subscript."""
        def f(x):
            return jax.lax.slice(x, (1,), (5,), (2,))
        _roundtrip_check(self, f, jnp.float32([0., 1., 2., 3., 4., 5.]))

    def test_slice_no_strides(self):
        """``slice`` handler fills strides with None when params['strides'] is None."""
        def f(x):
            return jax.lax.slice(x, (1,), (4,))
        _roundtrip_check(self, f, jnp.float32([0., 1., 2., 3., 4.]))

    def test_slice_2d(self):
        """Multi-dimensional slicing emits a tuple of slice objects."""
        def f(x):
            return jax.lax.slice(x, (0, 1), (2, 3))
        _roundtrip_check(self, f, jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4))

    def test_python_basic_slice(self):
        """Plain Python slicing also lowers to the ``slice`` primitive."""
        def f(x):
            return x[1:4]
        _roundtrip_check(self, f, jnp.float32([10., 11., 12., 13., 14.]))


class TestDotGeneralHandlers(unittest.TestCase):
    """Round-trip the dot_general handler across its branches."""

    def test_simple_matmul(self):
        """Simple matmul branch -> ``jax.numpy.matmul``."""
        def f(a, b):
            return jnp.matmul(a, b)
        _roundtrip_check(self, f,
                         jnp.float32([[1., 2.], [3., 4.]]),
                         jnp.float32([[5., 6.], [7., 8.]]))

    def test_matmul_with_astype(self):
        """Matmul whose preferred_element_type differs appends ``.astype``."""
        def f(a, b):
            return jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())),
                                       preferred_element_type=jnp.float16)
        _roundtrip_check(self, f,
                         jnp.float32([[1., 2.], [3., 4.]]),
                         jnp.float32([[5.], [6.]]))

    def test_general_batched_dot(self):
        """Non-simple dimension numbers take the general ``jax.lax.dot_general`` path."""
        def f(a, b):
            return jnp.einsum('bij,bjk->bik', a, b)
        _roundtrip_check(self, f,
                         jnp.float32(np.arange(24).reshape(2, 3, 4)),
                         jnp.float32(np.arange(40).reshape(2, 4, 5)))


class TestConvertElementType(unittest.TestCase):
    """Round-trip convert_element_type -> ``.astype``."""

    def test_float_to_int(self):
        """Casting float to int emits ``x.astype(jax.numpy.int32)``."""
        def f(x):
            return x.astype(jnp.int32)
        _roundtrip_check(self, f, jnp.float32([1.4, 2.6, 3.9]))

    def test_int_to_float(self):
        """Casting int to float."""
        def f(x):
            return x.astype(jnp.float32)
        _roundtrip_check(self, f, jnp.int32([1, 2, 3]))

    def test_to_bool(self):
        """Casting to bool exercises the ``bool_`` dtype branch of _astify_value."""
        def f(x):
            return x.astype(jnp.bool_)
        _roundtrip_check(self, f, jnp.float32([0., 1., 2.]))


class TestReshapeHandler(unittest.TestCase):
    """Round-trip the reshape handler (with and without a transpose)."""

    def test_plain_reshape(self):
        """``jnp.reshape`` with no dimensions param."""
        def f(x):
            return jnp.reshape(x, (3, 2))
        _roundtrip_check(self, f, jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3))

    def test_reshape_with_dimensions(self):
        """``jax.lax.reshape`` with a transpose folded in (dimensions != None)."""
        def f(x):
            return jax.lax.reshape(x, (6,), dimensions=(1, 0))
        _roundtrip_check(self, f, jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3))


class TestReductionAndCumulative(unittest.TestCase):
    """Round-trip reduction, cumulative and arg-reduction handlers."""

    def test_reduce_max_axis(self):
        """reduce_max with an axis -> ``jax.numpy.max(x, axis=...)``."""
        def f(x):
            return jnp.max(x, axis=1)
        _roundtrip_check(self, f, jnp.float32([[1., 5.], [4., 3.]]))

    def test_reduce_min_axis(self):
        """reduce_min with an axis."""
        def f(x):
            return jnp.min(x, axis=0)
        _roundtrip_check(self, f, jnp.float32([[1., 5.], [4., 3.]]))

    def test_reduce_prod(self):
        """reduce_prod over all axes."""
        def f(x):
            return jnp.prod(x)
        _roundtrip_check(self, f, jnp.float32([1., 2., 3., 4.]))

    def test_reduce_all(self):
        """reduce_and -> ``jax.numpy.all``."""
        def f(x):
            return jnp.all(x > 0)
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))

    def test_reduce_any(self):
        """reduce_or -> ``jax.numpy.any``."""
        def f(x):
            return jnp.any(x > 5)
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))

    def test_cumprod(self):
        """cumprod cumulative handler."""
        def f(x):
            return jnp.cumprod(x)
        _roundtrip_check(self, f, jnp.float32([1., 2., 3., 4.]))

    def test_cumsum_axis(self):
        """cumsum with an explicit axis keyword."""
        def f(x):
            return jnp.cumsum(x, axis=1)
        _roundtrip_check(self, f, jnp.float32([[1., 2.], [3., 4.]]))

    def test_cummax(self):
        """cummax cumulative handler."""
        def f(x):
            return jax.lax.cummax(x)
        _roundtrip_check(self, f, jnp.float32([1., 3., 2., 5., 4.]))

    def test_cummin_reverse(self):
        """cummin with reverse=True exercises the ``reverse`` keyword branch."""
        def f(x):
            return jax.lax.cummin(x, reverse=True)
        _roundtrip_check(self, f, jnp.float32([5., 3., 4., 1., 2.]))

    def test_argmin(self):
        """argmin arg-reduction handler."""
        def f(x):
            return jnp.argmin(x)
        _roundtrip_check(self, f, jnp.float32([3., 1., 2.]))

    def test_argmax_axis(self):
        """argmax with an axis."""
        def f(x):
            return jnp.argmax(x, axis=1)
        _roundtrip_check(self, f, jnp.float32([[1., 5.], [7., 2.]]))


class TestElementwiseHandlers(unittest.TestCase):
    """Round-trip the elementwise unary/binary primitives."""

    def test_more_unary_math(self):
        """A spread of transcendental/elementwise unary ops via _call_noparams."""
        for op in (jnp.exp2, jnp.log1p, jnp.expm1, jax.lax.rsqrt, jnp.cbrt,
                   jnp.sign, jnp.floor, jnp.ceil, jnp.round, jnp.square):
            with self.subTest(op=op.__name__):
                _roundtrip_check(self, (lambda x, _op=op: _op(x)),
                                 jnp.float32([0.3, 1.7, 2.2]), fn_name='f')

    def test_binary_elementwise(self):
        """Binary elementwise ops (atan2, nextafter) via _call_noparams."""
        for op in (jax.lax.atan2, jax.lax.nextafter):
            with self.subTest(op=op.__name__):
                _roundtrip_check(self, (lambda x, y, _op=op: _op(x, y)),
                                 jnp.float32([1., 2., 3.]),
                                 jnp.float32([0.5, 1.5, 2.5]), fn_name='f')

    def test_neg(self):
        """neg -> ``jax.lax.neg``."""
        def f(x):
            return -x
        _roundtrip_check(self, f, jnp.float32([1., -2., 3.]))

    def test_min_max_two_args(self):
        """lax.min / lax.max as ``normal_fn`` handlers (both args are params)."""
        def f(x, y):
            return jax.lax.min(x, y), jax.lax.max(x, y)
        _roundtrip_check(self, f, jnp.float32([1., 5., 3.]), jnp.float32([4., 2., 6.]))

    def test_all_comparisons(self):
        """All six comparison ops via _cmpop_fn."""
        def f(x, y):
            return (x < y, x > y, x <= y, x >= y, x == y, x != y)
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]), jnp.float32([2., 2., 1.]))

    def test_integer_pow_array(self):
        """integer_pow on an array argument."""
        def f(x):
            return x ** 4
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))

    def test_shift_ops(self):
        """Bit-shift primitives via _call_noparams on integer arrays."""
        def f(x, y):
            return (jax.lax.shift_left(x, y),
                    jax.lax.shift_right_logical(x, y))
        _roundtrip_check(self, f, jnp.int32([1, 2, 6]), jnp.int32([1, 1, 2]))


class TestShapeHandlers(unittest.TestCase):
    """Round-trip squeeze/expand_dims/concatenate/transpose/broadcast handlers."""

    def test_squeeze_multi(self):
        """squeeze of several axes."""
        def f(x):
            return jax.lax.squeeze(x, (0, 2))
        _roundtrip_check(self, f, jnp.ones((1, 3, 1), dtype=jnp.float32))

    def test_expand_dims_multi(self):
        """expand_dims of several axes."""
        def f(x):
            return jnp.expand_dims(x, (0, 2))
        _roundtrip_check(self, f, jnp.float32([1., 2.]))

    def test_concatenate_axis(self):
        """concatenate along a non-default axis."""
        def f(a, b):
            return jnp.concatenate([a, b], axis=1)
        _roundtrip_check(self, f,
                         jnp.float32([[1., 2.], [3., 4.]]),
                         jnp.float32([[5.], [6.]]))

    def test_transpose(self):
        """transpose -> ``jax.lax.transpose``."""
        def f(x):
            return jax.lax.transpose(x, (1, 0))
        _roundtrip_check(self, f, jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3))

    def test_broadcast(self):
        """broadcast -> ``jax.lax.broadcast``."""
        def f(x):
            return jax.lax.broadcast(x, (2,))
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))

    def test_broadcast_in_dim_general(self):
        """broadcast_in_dim with non-trivial broadcast_dimensions (general path)."""
        def f(x):
            return jax.lax.broadcast_in_dim(x, (3, 2), (1,))
        _roundtrip_check(self, f, jnp.float32([1., 2.]))

    def test_select_n(self):
        """select_n via ``jnp.where``."""
        def f(x):
            return jnp.where(x > 0, x, -x)
        _roundtrip_check(self, f, jnp.float32([-1., 2., -3.]))


class TestBroadcastZerosOnes(unittest.TestCase):
    """broadcast_in_dim recognises zeros/ones and emits jnp.zeros/jnp.ones."""

    def test_zeros(self):
        """A zero scalar broadcast emits ``jax.numpy.zeros``."""
        def f():
            return jnp.zeros((3,), dtype=jnp.float32)
        src = fn_to_python_code(f)
        gen = _exec_generated(src, 'f')
        self.assertTrue(np.allclose(np.asarray(gen()), np.zeros(3)))
        # jax >= 0.10 folds the scalar broadcast into a recognisable jnp.zeros;
        # older jax emits an equivalent broadcast_in_dim, which round-trips equally.
        if 'broadcast_in_dim' not in src:
            self.assertIn('zeros', src)

    def test_ones(self):
        """A one scalar broadcast emits ``jax.numpy.ones``."""
        def f():
            return jnp.ones((2, 2), dtype=jnp.float32)
        src = fn_to_python_code(f)
        gen = _exec_generated(src, 'f')
        self.assertTrue(np.allclose(np.asarray(gen()), np.ones((2, 2))))
        # jax >= 0.10 folds the scalar broadcast into a recognisable jnp.ones;
        # older jax emits an equivalent broadcast_in_dim, which round-trips equally.
        if 'broadcast_in_dim' not in src:
            self.assertIn('ones', src)


class TestDynamicSliceGeneration(unittest.TestCase):
    """Code-generation coverage for dynamic_slice / dynamic_update_slice.

    The *generated* code for these primitives cannot be executed in this JAX
    version: the negative-index normalisation that JAX inserts compares the
    index against a scalar literal that JAX boxes as ``TypedInt``, and
    ``_astify_value`` emits that object's (non-evaluable) ``repr`` -- a genuine
    source bug documented in the audit. These tests therefore only assert that
    the handlers run and emit the expected call, exercising their lines without
    executing the (broken) output.
    """

    def test_dynamic_slice_emits_call(self):
        """dynamic_slice handler emits ``jax.lax.dynamic_slice(...)``."""
        def f(x, i):
            return jax.lax.dynamic_slice(x, (i,), (2,))
        src = fn_to_python_code(f, jnp.arange(5.0, dtype=jnp.float32), 1)
        self.assertIn('jax.lax.dynamic_slice', src)
        self.assertIn('slice_sizes', src)

    def test_dynamic_update_slice_emits_call(self):
        """dynamic_update_slice handler emits ``jax.lax.dynamic_update_slice(...)``."""
        def f(x, u, i):
            return jax.lax.dynamic_update_slice(x, u, (i,))
        src = fn_to_python_code(f, jnp.arange(5.0, dtype=jnp.float32),
                                jnp.float32([9., 9.]), 1)
        self.assertIn('jax.lax.dynamic_update_slice', src)

    def test_dynamic_update_slice_multi_index_tuple(self):
        """A 2-D dynamic_update_slice packs its start indices into a tuple."""
        def f(x, u, i, j):
            return jax.lax.dynamic_update_slice(x, u, (i, j))
        src = fn_to_python_code(f, jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4),
                                jnp.ones((1, 2), dtype=jnp.float32), 0, 0)
        self.assertIn('jax.lax.dynamic_update_slice', src)


class TestBroadcastFull(unittest.TestCase):
    """broadcast_in_dim recognises an integer ``full`` constant."""

    def test_full_int(self):
        """A non-0/1 integer fill emits ``jax.numpy.full`` and round-trips."""
        def f():
            return jnp.full((2, 3), 7, dtype=jnp.int32)
        src = fn_to_python_code(f)
        gen = _exec_generated(src, 'f')
        self.assertTrue(np.allclose(np.asarray(gen()),
                                    np.full((2, 3), 7, dtype=np.int32)))
        # jax >= 0.10 folds the scalar broadcast into a recognisable jnp.full;
        # older jax emits an equivalent broadcast_in_dim, which round-trips equally.
        if 'broadcast_in_dim' not in src:
            self.assertIn('full', src)


class TestControlFlowHandlers(unittest.TestCase):
    """Round-trip scan/pjit/remat control-flow handlers."""

    def test_scan_simple_single_carry(self):
        """scan with a single carry and a single xs (the simple branch)."""
        def f(init, xs):
            def body(c, e):
                return c + e, c
            final, ys = jax.lax.scan(body, init, xs)
            return final, ys
        _roundtrip_check(self, f, jnp.float32(0.0), jnp.float32([1., 2., 3.]))

    def test_scan_multi_carry(self):
        """scan with multiple carries takes the tuple-repacking branch."""
        def f(a, b, xs):
            def body(carry, e):
                ca, cb = carry
                return (ca + e, cb * e), ca
            (fa, fb), ys = jax.lax.scan(body, (a, b), xs)
            return fa, fb, ys
        _roundtrip_check(self, f, jnp.float32(0.0), jnp.float32(1.0),
                         jnp.float32([1., 2., 3.]))

    def test_scan_zero_carry(self):
        """scan with zero carry (map-like) exercises the num_carry==0 branch."""
        def f(xs):
            def body(_, e):
                return (), e * 2.0
            _, ys = jax.lax.scan(body, (), xs)
            return ys
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))

    def test_scan_multi_carry_no_ys(self):
        """Multi-carry scan with no stacked outputs (num_carry == len(outvars))."""
        def f(a, b, xs):
            def body(carry, e):
                ca, cb = carry
                return (ca + e, cb * e), None
            (fa, fb), _ = jax.lax.scan(body, (a, b), xs)
            return fa, fb
        _roundtrip_check(self, f, jnp.float32(0.0), jnp.float32(1.0),
                         jnp.float32([1., 2., 3.]))

    def test_pjit_nested(self):
        """A nested ``jax.jit`` lowers to the pjit handler."""
        def f(x):
            return jax.jit(lambda y: y * y)(x) + 1.0
        _roundtrip_check(self, f, jnp.float32(3.0))

    def test_remat(self):
        """``jax.checkpoint`` lowers to the remat2 handler."""
        def f(x):
            return jax.checkpoint(lambda y: y * y * y)(x) + 1.0
        _roundtrip_check(self, f, jnp.float32(2.0))

    def test_add_any_via_grad(self):
        """Gradient accumulation introduces the add_any primitive."""
        def f(x):
            return jax.grad(lambda y: jnp.sum(y * y) + jnp.sum(y * y))(x)
        _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))

    def test_scan_with_consts_generation(self):
        """A scan closing over a constant takes the num_consts != 0 branch.

        The emitted code is not executed here: the carry's initial value is a
        boxed scalar literal (``TypedFloat``) that the documented source bug
        renders as a non-evaluable ``repr``. The test asserts the scan handler
        ran and partial-evaluated the constant into the loop body.
        """
        def f(w, xs):
            def body(c, e):
                return c + e * w, c
            final, ys = jax.lax.scan(body, 0.0, xs)
            return final, ys
        src = fn_to_python_code(f, jnp.float32(2.0), jnp.float32([1., 2., 3.]))
        self.assertIn('jax.lax.scan', src)
        self.assertIn('def fn_', src)


class TestScanMapAndClosedCall(unittest.TestCase):
    """Directly exercise the _astify_map and _astify_closed_call handlers.

    ``_astify_map`` is not wired to any primitive in the current code (the scan
    handler inlines map-like loops itself) and ``closed_call`` is eagerly
    inlined by modern JAX, so both are reached here by constructing the
    relevant equation and invoking the handler directly, then executing the
    generated source to confirm it is correct.
    """

    @staticmethod
    def _scan_eqn(fn, *args):
        closed = jax.make_jaxpr(fn)(*args)
        folded = _ir_tocode.constant_fold_jaxpr(closed.jaxpr)
        return folded, [e for e in folded.eqns if e.primitive.name == 'scan'][0]

    def test_astify_map_roundtrip(self):
        """_astify_map emits a ``jax.lax.map`` call that reproduces the loop."""
        def f(xs):
            def body(_, e):
                return (), e * 2.0
            _, ys = jax.lax.scan(body, (), xs)
            return ys

        xs = jnp.float32([1., 2., 3.])
        folded, scan_eqn = self._scan_eqn(f, xs)
        self.assertEqual(scan_eqn.params['num_carry'], 0)

        state = _ir_tocode.SourcerorState()
        for v in folded.invars:
            state.str_name(v)
        stmts = _ir_tocode._astify_map(state, scan_eqn)
        arg_names = [state.str_name(v) for v in folded.invars]
        fn_def = ast.FunctionDef(
            name='mapped',
            args=ast.arguments(
                args=[ast.arg(arg=n) for n in arg_names],
                vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
                defaults=[], posonlyargs=[]),
            body=list(stmts) + [ast.Return(value=ast.Name(
                id=state.str_name(scan_eqn.outvars[0]), ctx=ast.Load()))],
            decorator_list=[])
        src = ast.unparse(ast.fix_missing_locations(ast.Module(body=[fn_def],
                                                               type_ignores=[])))
        ns = {'jax': jax, 'jnp': jnp, 'np': np}
        exec(src, ns)
        got = ns['mapped'](xs)
        self.assertTrue(np.allclose(np.asarray(got), np.asarray(f(xs))))

    def test_astify_closed_call_roundtrip(self):
        """_astify_closed_call emits a nested function and calls it."""
        from types import SimpleNamespace

        def inner(x, y):
            return x * y + 1.0
        inner_cj = jax.make_jaxpr(inner)(jnp.float32(2.0), jnp.float32(3.0))

        def outer(x, y):
            return inner(x, y)
        outer_cj = jax.make_jaxpr(outer)(jnp.float32(2.0), jnp.float32(3.0))

        eqn = SimpleNamespace(
            primitive=SimpleNamespace(name='closed_call'),
            invars=list(outer_cj.jaxpr.invars),
            outvars=list(outer_cj.jaxpr.outvars),
            params={'call_jaxpr': inner_cj},
        )
        state = _ir_tocode.SourcerorState()
        arg_names = [state.str_name(v) for v in eqn.invars]
        stmts = _ir_tocode._astify_closed_call(state, eqn)
        fn_def = ast.FunctionDef(
            name='caller',
            args=ast.arguments(
                args=[ast.arg(arg=n) for n in arg_names],
                vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
                defaults=[], posonlyargs=[]),
            body=list(stmts) + [ast.Return(value=ast.Name(
                id=state.str_name(eqn.outvars[0]), ctx=ast.Load()))],
            decorator_list=[])
        src = ast.unparse(ast.fix_missing_locations(ast.Module(body=[fn_def],
                                                               type_ignores=[])))
        ns = {'jax': jax, 'jnp': jnp, 'np': np}
        exec(src, ns)
        got = ns['caller'](jnp.float32(2.0), jnp.float32(3.0))
        self.assertTrue(np.allclose(np.asarray(got), 7.0))


class TestConstantFolding(unittest.TestCase):
    """Exercise constant_fold_jaxpr / partial_eval_jaxpr."""

    def test_literal_integer_pow_folds(self):
        """A wholly-constant integer_pow is folded away before code generation."""
        def f(x):
            return x + jax.lax.integer_pow(jnp.float32(2.0), 3)
        src = fn_to_python_code(f, jnp.float32(1.0))
        # The constant 2**3 == 8 should be inlined; no integer_pow call remains.
        self.assertNotIn('integer_pow', src)
        gen = _exec_generated(src, 'f')
        self.assertTrue(np.allclose(np.asarray(gen(jnp.float32(5.0))), 13.0))

    def test_constant_fold_chain(self):
        """A chain of constant ops folds to a single inlined literal."""
        def f(x):
            c = jax.lax.integer_pow(jnp.float32(3.0), 2) + jnp.float32(1.0)
            return x * c
        src = fn_to_python_code(f, jnp.float32(2.0))
        gen = _exec_generated(src, 'f')
        self.assertTrue(np.allclose(np.asarray(gen(jnp.float32(2.0))), 20.0))

    def test_partial_eval_jaxpr_direct(self):
        """partial_eval_jaxpr with a supplied env evaluates the matching var."""
        def f(x, y):
            return x * y + 1.0
        closed = jax.make_jaxpr(f)(jnp.float32(2.0), jnp.float32(3.0))
        jaxpr = closed.jaxpr
        # Bind the first invar to a concrete value; the result jaxpr should keep
        # the second invar as the only remaining argument.
        env = {jaxpr.invars[0]: np.float32(4.0)}
        new_jaxpr = _ir_tocode.partial_eval_jaxpr(jaxpr, env)
        self.assertLessEqual(len(new_jaxpr.invars), len(jaxpr.invars))

    def test_partial_eval_jaxpr_literal_env(self):
        """partial_eval with a Literal in the env exercises read_or_self's Literal branch."""
        from brainstate._compatible_import import Literal
        def f(x, y):
            return x * y + 1.0
        jaxpr = jax.make_jaxpr(f)(jnp.float32(2.0), jnp.float32(3.0)).jaxpr
        lit = Literal(np.float32(4.0), jaxpr.invars[0].aval)
        new_jaxpr = _ir_tocode.partial_eval_jaxpr(jaxpr, {jaxpr.invars[0]: lit})
        # Only the unbound second argument should remain.
        self.assertEqual(len(new_jaxpr.invars), 1)

    def test_eval_eqn_closed_call_returns_jaxpr(self):
        """_eval_eqn dispatches closed_call to partial_eval_jaxpr (returns a Jaxpr)."""
        from types import SimpleNamespace
        def inner(x, y):
            return x * y + 1.0
        inner_cj = jax.make_jaxpr(inner)(jnp.float32(2.0), jnp.float32(3.0))
        eqn = SimpleNamespace(primitive=SimpleNamespace(name='closed_call'),
                              params={'call_jaxpr': inner_cj})
        out = _ir_tocode._eval_eqn(eqn, [np.float32(2.0), np.float32(3.0)])
        from brainstate._compatible_import import Jaxpr
        self.assertIsInstance(out, Jaxpr)

    def test_partial_eval_inlines_closed_call_jaxpr(self):
        """A foldable closed_call equation is inlined during partial evaluation.

        Modern JAX inlines ``closed_call`` before a jaxpr is ever produced, so
        this synthesises the equation to drive the Jaxpr-inlining branch of
        ``partial_eval_jaxpr`` (where ``_eval_eqn`` returns a nested jaxpr).
        """
        # ``new_jaxpr_eqn`` lives in jax.core (<0.10) or jax.extend.core (>=0.10);
        # use the project's version-aware shim rather than a fixed import path.
        from brainstate._compatible_import import new_jaxpr_eqn
        from jax._src.core import closed_call_p

        def inner(x, y):
            return x * y + 1.0
        inner_cj = jax.make_jaxpr(inner)(jnp.float32(2.0), jnp.float32(3.0))

        def outer(x, y):
            return inner(x, y)
        oj = jax.make_jaxpr(outer)(jnp.float32(2.0), jnp.float32(3.0)).jaxpr

        eqn = new_jaxpr_eqn(list(oj.invars), list(oj.outvars), closed_call_p,
                            {'call_jaxpr': inner_cj}, set())
        synthetic = oj.replace(eqns=[eqn])
        env = {oj.invars[0]: np.float32(2.0), oj.invars[1]: np.float32(3.0)}
        folded = _ir_tocode.partial_eval_jaxpr(synthetic, env)
        # Everything folds to the constant 2*3 + 1 == 7.
        self.assertEqual(len(folded.eqns), 0)
        self.assertAlmostEqual(float(np.asarray(folded.outvars[0].val)), 7.0)

    def test_identity_passthrough_keeps_invar(self):
        """An identity function keeps its (otherwise unused) invar as a parameter."""
        def f(x):
            return x
        src = fn_to_python_code(f, jnp.float32(1.0))
        gen = _exec_generated(src, 'f')
        self.assertTrue(np.allclose(np.asarray(gen(jnp.float32(7.0))), 7.0))


class TestJaxprToPythonCodeEntry(unittest.TestCase):
    """Exercise the jaxpr_to_python_code public entry point directly."""

    def test_jaxpr_entry_basic(self):
        """jaxpr_to_python_code generates a runnable function from a raw jaxpr."""
        def f(x):
            return jnp.sum(x * 2.0)
        closed = jax.make_jaxpr(f)(jnp.float32([1., 2., 3.]))
        src = jaxpr_to_python_code(closed.jaxpr, fn_name='myfunc')
        self.assertIn('def myfunc', src)
        ns = {'jax': jax, 'jnp': jnp, 'np': np}
        exec(src, ns)
        got = ns['myfunc'](jnp.float32([1., 2., 3.]))
        self.assertTrue(np.allclose(np.asarray(got), 12.0))

    def test_jaxpr_entry_default_name(self):
        """The default fn_name is used when none is supplied."""
        def f(x):
            return x + 1.0
        closed = jax.make_jaxpr(f)(jnp.float32(1.0))
        src = jaxpr_to_python_code(closed.jaxpr)
        self.assertIn('def generated_function', src)


class TestRegisterPrimHandler(unittest.TestCase):
    """Exercise register_prim_handler / register_prim_as registration paths."""

    def test_register_custom_handler_roundtrip(self):
        """A freshly registered handler is used when generating code."""
        from brainstate.transform._ir_tocode import (
            register_prim_handler, prim_to_python, normal_fn,
        )

        # 'rev' has no built-in handler; register one (forwarding the
        # ``dimensions`` param as a kwarg) and round-trip it, then restore.
        prim = 'rev'
        had = prim in prim_to_python
        old = prim_to_python.get(prim)
        try:
            register_prim_handler(prim, normal_fn('jax.lax.rev'))

            def f(x):
                return jax.lax.rev(x, (0,))
            _roundtrip_check(self, f, jnp.float32([1., 2., 3.]))
        finally:
            if had:
                prim_to_python[prim] = old
            else:
                prim_to_python.pop(prim, None)

    def test_register_overwrite_warns(self):
        """Overwriting an existing handler emits a warning."""
        from brainstate.transform._ir_tocode import (
            register_prim_handler, prim_to_python,
        )
        old = prim_to_python['add']
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                register_prim_handler('add', old)
            self.assertTrue(any('Overwriting' in str(w.message) for w in caught))
        finally:
            prim_to_python['add'] = old

    def test_register_prim_as_decorator(self):
        """register_prim_as registers the decorated function and returns it."""
        from brainstate.transform._ir_tocode import (
            register_prim_as, prim_to_python,
        )

        sentinel_name = '__brainstate_test_prim__'
        try:
            @register_prim_as(sentinel_name)
            def handler(state, eqn):
                return None

            self.assertIs(prim_to_python[sentinel_name], handler)
        finally:
            prim_to_python.pop(sentinel_name, None)


class TestAstifyValueBranches(unittest.TestCase):
    """Directly unit-test the _astify_value / _astify_array branches."""

    def test_dtype_named_branches(self):
        """Named dtypes map to ``jax.numpy.<name>`` attribute access."""
        self.assertEqual(_unparse(_ir_tocode._astify_value(jnp.dtype('float32'))),
                         'jax.numpy.float32')
        self.assertEqual(_unparse(_ir_tocode._astify_value(jnp.dtype('int64'))),
                         'jax.numpy.int64')
        self.assertEqual(_unparse(_ir_tocode._astify_value(jnp.dtype('bfloat16'))),
                         'jax.numpy.bfloat16')

    def test_dtype_bool_branch(self):
        """The bool dtype maps to ``jax.numpy.bool_``."""
        self.assertEqual(_unparse(_ir_tocode._astify_value(jnp.dtype('bool'))),
                         'jax.numpy.bool_')

    def test_dtype_generic_branch(self):
        """An unusual dtype falls back to ``jax.numpy.dtype('...')``."""
        src = _unparse(_ir_tocode._astify_value(jnp.dtype('uint8')))
        self.assertIn("dtype('uint8')", src)

    def test_unspecified_value(self):
        """UNSPECIFIED emits a name and records the import."""
        from jax._src.sharding_impls import UNSPECIFIED
        _ir_tocode.prefix_imports.clear()
        try:
            node = _ir_tocode._astify_value(UNSPECIFIED)
            self.assertEqual(_unparse(node), 'UNSPECIFIED')
            self.assertTrue(any('UNSPECIFIED' in s for s in _ir_tocode.prefix_imports))
        finally:
            _ir_tocode.prefix_imports.clear()

    def test_enum_value(self):
        """An enum value emits ``ClassName.MEMBER`` attribute access."""
        node = _ir_tocode._astify_value(_Color.GREEN)
        self.assertEqual(_unparse(node), '_Color.GREEN')

    def test_nested_tuple_value(self):
        """Nested tuples/lists/None/str recurse to a tuple literal."""
        node = _ir_tocode._astify_value((1, (2.0, None), 'x'))
        self.assertEqual(_unparse(node), "(1, (2.0, None), 'x')")

    def test_unknown_value_warns(self):
        """An unrecognised value type warns and falls back to repr parsing."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            node = _ir_tocode._astify_value(slice(1, 5, 2))
        self.assertTrue(any('Unknown value type' in str(w.message) for w in caught))
        self.assertIn('slice', _unparse(node))

    def test_astify_array_int64(self):
        """A numpy int64 scalar emits a plain integer constant."""
        node = _ir_tocode._astify_array(np.int64(7))
        self.assertEqual(_unparse(node), '7')

    def test_astify_array_scalar_standard_dtype(self):
        """A 0-d float32 array emits a bare constant (no dtype wrapper)."""
        node = _ir_tocode._astify_array(np.float32(2.5))
        self.assertEqual(eval(_unparse(node)), 2.5)

    def test_astify_array_scalar_nonstandard_dtype(self):
        """A 0-d uint8 array wraps the constant in its dtype call."""
        node = _ir_tocode._astify_array(np.uint8(5))
        src = _unparse(node)
        self.assertIn("dtype('uint8')", src)

    def test_astify_array_multidim(self):
        """A multi-dimensional array emits ``jax.numpy.array([...], dtype=...)``."""
        node = _ir_tocode._astify_array(np.float32([[1., 2.], [3., 4.]]))
        src = _unparse(node)
        self.assertIn('jax.numpy.array', src)
        out = eval(src, {'jax': jax, 'jnp': jnp, 'np': np})
        self.assertTrue(np.allclose(np.asarray(out), [[1., 2.], [3., 4.]]))


class TestCoerceToNumpy(unittest.TestCase):
    """Exercise the _coerce_to_numpy fallbacks."""

    def test_native_numpy_passthrough(self):
        """Native numpy scalars/arrays are returned unchanged."""
        arr = np.float32([1., 2.])
        self.assertIs(_ir_tocode._coerce_to_numpy(arr), arr)

    def test_val_attribute_preferred(self):
        """An object exposing ``.val`` as ndarray uses it directly."""
        class Boxed:
            def __init__(self, v):
                self.val = np.asarray(v)
        out = _ir_tocode._coerce_to_numpy(Boxed(np.float32([3., 4.])))
        self.assertTrue(np.allclose(out, [3., 4.]))

    def test_array_protocol_fallback(self):
        """An object implementing ``__array__`` is converted via numpy."""
        class ArrLike:
            def __array__(self, dtype=None, copy=None):
                return np.float32([5., 6.])
        out = _ir_tocode._coerce_to_numpy(ArrLike())
        self.assertTrue(np.allclose(out, [5., 6.]))

    def test_attribute_fallback_when_asarray_fails(self):
        """When ``np.asarray`` fails, the ``_value`` attribute fallback is used."""
        class Weird:
            _value = np.float32([7., 8.])

            def __array__(self, *a, **k):
                raise ValueError('cannot convert')
        out = _ir_tocode._coerce_to_numpy(Weird())
        self.assertTrue(np.allclose(out, [7., 8.]))

    def test_attribute_fallback_skips_unconvertible_attrs(self):
        """A first fallback attr that also fails is skipped for the next one."""
        class _Bad:
            def __array__(self, *a, **k):
                raise ValueError('bad inner')

        class Weird:
            # ``_value`` cannot be coerced (forces the inner except/continue),
            # ``array`` succeeds.
            _value = _Bad()
            array = np.float32([1., 2.])

            def __array__(self, *a, **k):
                raise ValueError('cannot convert')
        out = _ir_tocode._coerce_to_numpy(Weird())
        self.assertTrue(np.allclose(out, [1., 2.]))

    def test_coerce_reraises_when_no_fallback(self):
        """If nothing is convertible, the original error is re-raised."""
        class Hopeless:
            def __array__(self, *a, **k):
                raise ValueError('cannot convert')
        with self.assertRaises(Exception):
            _ir_tocode._coerce_to_numpy(Hopeless())


class TestHelperContainers(unittest.TestCase):
    """Exercise IdentitySet / IdentityMap / SourcerorState helpers."""

    def test_identity_set_compares_by_identity(self):
        """Equal-but-distinct objects are kept as separate set members."""
        s = _ir_tocode.IdentitySet()
        a, b = [1], [1]
        s.add(a)
        s.add(b)
        self.assertEqual(len(s), 2)
        self.assertIn(a, s)
        s.discard(a)
        self.assertNotIn(a, s)
        self.assertEqual(len(s), 1)
        self.assertIn('IdentitySet', repr(s))
        self.assertIn('IdentitySet', str(s))
        self.assertEqual(list(s), [b])

    def test_identity_set_init_empty(self):
        """The constructor accepts no iterable (empty set)."""
        s = _ir_tocode.IdentitySet()
        self.assertEqual(len(s), 0)

    def test_identity_map_compares_by_identity(self):
        """Keys are compared by identity, allowing equal-but-distinct keys."""
        m = _ir_tocode.IdentityMap()
        a, b = [1], [1]
        m[a] = 'x'
        m[b] = 'y'
        self.assertEqual(len(m), 2)
        self.assertEqual(m[a], 'x')
        self.assertIn(a, m)
        del m[a]
        self.assertNotIn(a, m)
        self.assertEqual(len(m), 1)
        self.assertIn('IdentityMap', repr(m))
        self.assertIn('IdentityMap', str(m))
        self.assertIn('y', list(m))

    def test_identity_map_init_mapping(self):
        """The constructor accepts an initial mapping (exercises update)."""
        m = _ir_tocode.IdentityMap({'k': 5})
        self.assertEqual(len(m), 1)
        a = [1]
        m[a] = 9
        self.assertEqual(m[a], 9)

    def test_sourceror_skolem_increments(self):
        """skolem produces unique incrementing names."""
        st = _ir_tocode.SourcerorState()
        self.assertEqual(st.skolem('fn'), 'fn_1')
        self.assertEqual(st.skolem('fn'), 'fn_2')
        self.assertEqual(st.skolem('g'), 'g_3')

    def test_sourceror_name_many_vars(self):
        """Naming >26 variables exercises the multi-letter naming loop."""
        st = _ir_tocode.SourcerorState()

        class V:
            pass
        # Keep references so the distinct objects are not garbage-collected and
        # their ids reused (IdentityMap keys on object identity).
        vs = [V() for _ in range(30)]
        names = [st.str_name(v) for v in vs]
        # First 26 are single letters; the 27th onward are multi-letter and all
        # 30 names are unique.
        self.assertEqual(names[0], 'a')
        self.assertEqual(names[25], 'z')
        self.assertEqual(len(set(names)), 30)
        self.assertTrue(any(len(n) > 1 for n in names))

    def test_sourceror_name_caches(self):
        """The same Var returns the same generated name on repeated calls."""
        st = _ir_tocode.SourcerorState()

        class V:
            pass
        v = V()
        self.assertEqual(st.str_name(v), st.str_name(v))
        node = st.name(v)
        self.assertIsInstance(node, ast.Name)


class TestLeafWrapping(unittest.TestCase):
    """Exercise _maybe_wrap_fn_for_leaves via pytree-structured arguments."""

    def test_pytree_argument_wrapped(self):
        """A function taking a container argument is wrapped to flatten leaves."""
        def f(d):
            return d['a'] + d['b']
        arg = {'a': jnp.float32(2.0), 'b': jnp.float32(3.0)}
        src = fn_to_python_code(f, arg)
        # The wrapper accepts *args/**kwargs and flattens via tree_leaves.
        self.assertIn('tree_leaves', src)
        ns = {'jax': jax, 'jnp': jnp, 'np': np}
        exec(src, ns)
        got = ns['f'](arg)
        self.assertTrue(np.allclose(np.asarray(got), 5.0))


class TestMiscEdgeCases(unittest.TestCase):
    """Small edge cases: missing __name__, invalid atoms."""

    def test_callable_without_name_falls_back(self):
        """A callable without ``__name__`` (functools.partial) is named generically."""
        import functools
        p = functools.partial(lambda x, y: x + y, jnp.float32(1.0))
        self.assertFalse(hasattr(p, '__name__'))
        src = fn_to_python_code(p, jnp.float32(2.0))
        # The AttributeError branch picks the generic fallback name.
        self.assertIn('def generated_function', src)

    def test_astify_atom_literal(self):
        """_astify_atom on a Literal forwards to _astify_value."""
        from brainstate._compatible_import import Literal
        jaxpr = jax.make_jaxpr(lambda x: x + 1.0)(jnp.float32(1.0)).jaxpr
        aval = jaxpr.invars[0].aval
        lit = Literal(np.float32(2.0), aval)
        state = _ir_tocode.SourcerorState()
        node = _ir_tocode._astify_atom(state, lit)
        self.assertEqual(eval(_unparse(node)), 2.0)

    def test_astify_atom_invalid_type_raises(self):
        """_astify_atom on neither a Literal nor a Var raises NotImplementedError."""
        state = _ir_tocode.SourcerorState()
        with self.assertRaises(NotImplementedError):
            _ir_tocode._astify_atom(state, object())

    def test_maybe_tuple_vars_single_and_multi(self):
        """maybe_tuple_vars returns the lone element, or a Tuple for many."""
        single = _ir_tocode.maybe_tuple_vars([ast.Name(id='a', ctx=ast.Load())])
        self.assertIsInstance(single, ast.Name)
        multi = _ir_tocode.maybe_tuple_vars([ast.Name(id='a', ctx=ast.Load()),
                                             ast.Name(id='b', ctx=ast.Load())])
        self.assertIsInstance(multi, ast.Tuple)

    def test_maybe_untuple_vars(self):
        """maybe_untuple_vars stars the value only when is_tuple is True."""
        name = ast.Name(id='a', ctx=ast.Load())
        self.assertIsInstance(_ir_tocode.maybe_untuple_vars(name, True), ast.Starred)
        self.assertIs(_ir_tocode.maybe_untuple_vars(name, False), name)

    def _single_var_eqn(self, params):
        """Build a one-in/one-out synthetic equation sharing the same Var."""
        from types import SimpleNamespace
        jaxpr = jax.make_jaxpr(lambda x: x + 1.0)(jnp.float32([1., 2.])).jaxpr
        invar = jaxpr.invars[0]
        outvar = jaxpr.eqns[0].outvars[0]
        state = _ir_tocode.SourcerorState()
        state.str_name(invar)
        eqn = SimpleNamespace(invars=[invar], outvars=[outvar], params=params)
        return state, eqn

    def test_expand_dims_handler_direct(self):
        """_expand_dims_handler emits ``jax.lax.expand_dims``.

        ``jnp.expand_dims`` lowers to ``broadcast_in_dim`` in this JAX version,
        so the handler is exercised directly with a synthetic equation.
        """
        state, eqn = self._single_var_eqn({'dimensions': (0, 2)})
        node = _ir_tocode._expand_dims_handler(state, eqn)
        self.assertEqual(_unparse(node), 'b = jax.lax.expand_dims(a, (0, 2))')

    def test_random_wrap_handler_direct(self):
        """_astify_random_wrap is a no-op assignment of its input."""
        state, eqn = self._single_var_eqn({})
        node = _ir_tocode._astify_random_wrap(state, eqn)
        self.assertEqual(_unparse(node), 'b = a')

    def test_reduce_fn_axes_none(self):
        """_reduce_fn with axes=None omits the ``axis`` keyword."""
        state, eqn = self._single_var_eqn({'axes': None})
        node = _ir_tocode._reduce_fn('jax.numpy.sum')(state, eqn)
        self.assertEqual(_unparse(node), 'b = jax.numpy.sum(a)')

    def test_pjit_handler_call_jaxpr_fallback(self):
        """_astify_pjit falls back to the ``call_jaxpr`` param when ``jaxpr`` is absent.

        Newer JAX uses a ``jaxpr`` param; the ``call_jaxpr`` fallback covers
        older releases and is driven here with a synthetic equation.
        """
        from types import SimpleNamespace
        jaxpr = jax.make_jaxpr(lambda x: x + 1.0)(jnp.float32(1.0)).jaxpr
        invar = jaxpr.invars[0]
        outvar = jaxpr.eqns[0].outvars[0]
        inner_cj = jax.make_jaxpr(lambda x: x * 2.0)(jnp.float32(1.0))
        state = _ir_tocode.SourcerorState()
        state.str_name(invar)
        eqn = SimpleNamespace(invars=[invar], outvars=[outvar],
                              params={'call_jaxpr': inner_cj})
        stmts = _ir_tocode._astify_pjit(state, eqn)
        src = _unparse(ast.Module(body=stmts, type_ignores=[]))
        self.assertIn('jax.jit', src)


class TestBoxedScalarLiteralRegression(unittest.TestCase):
    """Regression tests for JAX-boxed scalar literals in generated code.

    Current JAX boxes constant scalars as ``jax._src.literals.TypedInt`` /
    ``TypedFloat`` (subclasses of ``int`` / ``float``). ``_astify_value`` must
    coerce these to plain Python scalars; otherwise ``ast.unparse`` renders the
    object repr (e.g. ``TypedFloat(7.0, dtype=float32)``) and the generated code
    raises ``NameError`` when executed.
    """

    def test_float_full_roundtrips_and_executes(self):
        """A boxed float literal from ``jnp.full`` round-trips and runs."""

        def f(x):
            return x + jnp.full((2,), 7.0)

        _roundtrip_check(self, f, jnp.ones((2,), dtype=jnp.float32))

    def test_generated_code_has_no_boxed_repr(self):
        """The generated source contains a bare ``7.0`` literal, not the repr."""

        def f(x):
            return x + jnp.full((3,), 7.0)

        src = fn_to_python_code(f, jnp.ones((3,), dtype=jnp.float32))
        self.assertNotIn('TypedFloat', src)
        self.assertNotIn('TypedInt', src)
        self.assertIn('7.0', src)

    def test_dynamic_slice_roundtrips_and_executes(self):
        """``dynamic_slice`` (whose index normalization boxes ``0``) executes."""

        def f(x):
            return jax.lax.dynamic_slice(x, (1,), (2,))

        _roundtrip_check(self, f, jnp.arange(5.0, dtype=jnp.float32))

    def test_dynamic_update_slice_roundtrips_and_executes(self):
        """``dynamic_update_slice`` round-trips and runs without ``NameError``."""

        def f(x, y):
            return jax.lax.dynamic_update_slice(x, y, (1,))

        _roundtrip_check(
            self,
            f,
            jnp.zeros((4,), dtype=jnp.float32),
            jnp.ones((2,), dtype=jnp.float32),
        )

    def test_boxed_int_literal_coerced(self):
        """A boxed integer literal is coerced to a plain ``int`` constant."""
        # Boxed literals (jax._src.literals.TypedInt) were introduced in jax 0.10;
        # earlier versions have no such object to coerce, so skip there.
        try:
            from jax._src import literals
            typed_int = literals.TypedInt(5, dtype=jnp.int32.dtype)
        except (ImportError, AttributeError):
            self.skipTest('jax._src.literals.TypedInt requires jax>=0.10')

        node = _ir_tocode._astify_value(typed_int)
        self.assertIsInstance(node, ast.Constant)
        self.assertEqual(node.value, 5)
        self.assertIs(type(node.value), int)


if __name__ == '__main__':
    unittest.main()
