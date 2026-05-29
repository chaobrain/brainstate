# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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
import brainstate as bst
from brainstate.transform._debug import (
    debug_nan,
    debug_nan_if,
    DebugNan,
    _has_nan_flag,
    _extract_user_source,
)


# ---------------------------------------------------------------------------
# _has_nan_flag (output-gate helper: True iff any inexact value is NaN or Inf)
# ---------------------------------------------------------------------------

class TestHasNanFlag(unittest.TestCase):

    def test_no_nan(self):
        self.assertFalse(bool(_has_nan_flag([jnp.array([1.0, 2.0, 3.0])])))

    def test_nan(self):
        self.assertTrue(bool(_has_nan_flag([jnp.array([1.0, jnp.nan, 3.0])])))

    def test_inf(self):
        self.assertTrue(bool(_has_nan_flag([jnp.array([1.0, jnp.inf])])))

    def test_neg_inf(self):
        self.assertTrue(bool(_has_nan_flag([jnp.array([-jnp.inf])])))

    def test_integer_array_ignored(self):
        self.assertFalse(bool(_has_nan_flag([jnp.array([1, 2, 3])])))

    def test_empty_list(self):
        self.assertFalse(bool(_has_nan_flag([])))

    def test_mixed_clean_and_nan(self):
        self.assertTrue(bool(_has_nan_flag([jnp.array([1.0]), jnp.array([jnp.nan])])))

    def test_complex_nan_detected(self):
        # complex dtypes must be scanned too (jnp.floating would exclude them)
        self.assertTrue(bool(_has_nan_flag([jnp.array([jnp.nan + 0.0j], dtype=jnp.complex64)])))

    def test_complex_clean(self):
        self.assertFalse(bool(_has_nan_flag([jnp.array([1.0 + 2.0j], dtype=jnp.complex64)])))


# ---------------------------------------------------------------------------
# _extract_user_source
# ---------------------------------------------------------------------------

class TestExtractUserSource(unittest.TestCase):

    def test_none_returns_unknown(self):
        self.assertIn("unknown", _extract_user_source(None))

    def test_excludes_brainstate_internals(self):
        from brainstate.transform._make_jaxpr import StatefulFunction

        class Model(bst.graph.Node):
            def __init__(self):
                self.w = bst.State(jnp.array([1.0]))

            def __call__(self, x):
                return jnp.log(self.w.value) + x

        sf = StatefulFunction(Model())
        cache_key = sf.get_arg_cache_key(jnp.array([1.0]), compile_if_miss=True)
        closed = sf.get_jaxpr_by_cache(cache_key)
        for eqn in closed.jaxpr.eqns:
            src = _extract_user_source(getattr(eqn, 'source_info', None))
            self.assertNotIn('_make_jaxpr.py', src)


# ---------------------------------------------------------------------------
# Basic detection (NaN, Inf, clean) — output-reaching contamination
# ---------------------------------------------------------------------------

class TestBasic(unittest.TestCase):

    def test_nan_raises_with_primitive_and_source(self):
        def fn(x):
            return jnp.log(x)   # log(-1) -> NaN reaches output

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([-1.0, 1.0]))
        msg = str(ctx.exception)
        self.assertIn("NaN", msg)
        self.assertIn("log", msg)
        self.assertIn("Source location", msg)
        self.assertTrue('File' in msg or '.py' in msg)
        self.assertNotIn('_make_jaxpr.py', msg)

    def test_div_by_zero_inf_raises(self):
        def fn(x):
            return 1.0 / x      # 1/0 -> inf reaches output

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([0.0, 1.0]))
        self.assertIn("NaN", str(ctx.exception))

    def test_clean_no_raise(self):
        debug_nan(lambda x: x * 2.0 + 1.0, jnp.array([1.0, 2.0, 3.0]))

    def test_phase_in_message(self):
        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(lambda x: jnp.log(x), jnp.array([-1.0]), phase='forward')
        self.assertIn("forward", str(ctx.exception))

    def test_propagated_nan_attributed_to_source(self):
        def fn(x):
            y = jnp.log(x)   # source
            z = y * 2.0
            return z + 1.0

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([-1.0]))
        self.assertIn("log", str(ctx.exception))


# ---------------------------------------------------------------------------
# Reductions / common primitives must not crash (bug: get_bind_params)
# ---------------------------------------------------------------------------

class TestReductions(unittest.TestCase):

    def test_sum_clean_no_crash(self):
        debug_nan(lambda x: jnp.sum(x), jnp.array([1.0, 2.0, 3.0]))

    def test_mean_clean_no_crash(self):
        debug_nan(lambda x: jnp.mean(x), jnp.array([1.0, 2.0, 3.0]))

    def test_sum_axis_clean_no_crash(self):
        debug_nan(lambda x: jnp.sum(x, axis=0), jnp.ones((3, 2)))

    def test_cumsum_clean_no_crash(self):
        debug_nan(lambda x: jnp.cumsum(x), jnp.array([1.0, 2.0, 3.0]))

    def test_exp_overflow_inf_detected(self):
        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(lambda x: jnp.exp(x), jnp.array([1e30]))
        self.assertIn("Inf", str(ctx.exception))


# ---------------------------------------------------------------------------
# Autodiff / gradients (bug: crashed on reduce_sum in grad jaxpr)
# ---------------------------------------------------------------------------

class TestGrad(unittest.TestCase):

    def test_grad_clean_no_crash(self):
        debug_nan(jax.grad(lambda x: (x * 2.0).sum()), jnp.array([1.0, 2.0]))

    def test_grad_backward_nan_detected(self):
        # backward of sqrt at 0 -> inf/division
        with self.assertRaises(RuntimeError):
            debug_nan(jax.grad(lambda x: jnp.sqrt(x).sum()), jnp.array([0.0, 4.0]))

    def test_value_and_grad_clean_no_crash(self):
        debug_nan(jax.value_and_grad(lambda x: (x ** 2).sum()), jnp.array([1.0, 2.0]))

    def test_vmap_of_grad_clean_no_crash(self):
        g = jax.grad(lambda x: jnp.sum(jnp.log(x)))
        debug_nan(jax.vmap(g), jnp.array([[1.0, 2.0], [3.0, 4.0]]))


# ---------------------------------------------------------------------------
# vmap / batching
# ---------------------------------------------------------------------------

class TestVmap(unittest.TestCase):

    def test_vmap_some_lane_nan_detected(self):
        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(jax.vmap(lambda x: jnp.log(x)), jnp.array([-1.0, 1.0, 2.0]))
        self.assertIn("log", str(ctx.exception))

    def test_vmap_of_cond_masked_nan_is_silent(self):
        # vmapped cond lowers to select_n; the un-taken branch computes log(-1)=NaN
        # but it is masked away and the real output is clean -> must NOT raise.
        def f(x):
            return jax.lax.cond(x > 0.0, lambda y: jnp.log(y), lambda y: y * 2.0, x)

        xs = jnp.array([1.0, -1.0, 2.0])
        # sanity: real result is finite
        assert bool(jnp.all(jnp.isfinite(jax.vmap(f)(xs))))
        debug_nan(jax.vmap(f), xs)   # must be silent


# ---------------------------------------------------------------------------
# Masked-NaN semantics: only output-reaching NaN is reported
# ---------------------------------------------------------------------------

class TestMaskedNanSilent(unittest.TestCase):

    def test_where_guarded_log_is_silent(self):
        # jnp.where computes log on all lanes but discards the bad ones.
        def f(x):
            return jnp.where(x > 0, jnp.log(x), 0.0)

        xs = jnp.array([0.0, -1.0, 1.0])
        assert bool(jnp.all(jnp.isfinite(f(xs))))
        debug_nan(f, xs)   # must be silent

    def test_output_reaching_nan_still_raises(self):
        def f(x):
            return jnp.where(x > 0, 0.0, jnp.log(x))  # selects the NaN lane

        with self.assertRaises(RuntimeError):
            debug_nan(f, jnp.array([-1.0, 1.0]))


# ---------------------------------------------------------------------------
# custom_jvp / custom_vjp primitives (relu/softmax/logsumexp)
# ---------------------------------------------------------------------------

class TestCustomJVP(unittest.TestCase):

    def test_relu_clean_no_crash(self):
        debug_nan(lambda x: jax.nn.relu(x), jnp.array([-1.0, 2.0, 3.0]))

    def test_softmax_clean_no_crash(self):
        debug_nan(lambda x: jax.nn.softmax(x), jnp.array([1.0, 2.0, 3.0]))

    def test_logsumexp_clean_no_crash(self):
        debug_nan(lambda x: jax.scipy.special.logsumexp(x), jnp.array([1.0, 2.0, 3.0]))

    def test_custom_jvp_inner_nan_detected_and_attributed(self):
        @jax.custom_jvp
        def f(x):
            return jnp.log(x)   # inner source

        @f.defjvp
        def _f_jvp(primals, tangents):
            (x,), (xd,) = primals, tangents
            return f(x), xd / x

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(lambda x: f(x), jnp.array([-1.0, 1.0]))
        self.assertIn("log", str(ctx.exception))


# ---------------------------------------------------------------------------
# remat / checkpoint (bug: misattributed to remat2 + inner NaN hidden)
# ---------------------------------------------------------------------------

class TestRemat(unittest.TestCase):

    def test_remat_clean_no_raise(self):
        @jax.checkpoint
        def block(x):
            return x * 2.0

        debug_nan(lambda x: block(x), jnp.array([1.0, 2.0]))

    def test_remat_nan_attributed_to_inner_primitive(self):
        @jax.checkpoint
        def block(x):
            return jnp.log(x)

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(lambda x: block(x), jnp.array([-1.0, 1.0]))
        msg = str(ctx.exception)
        self.assertIn("log", msg)
        self.assertNotIn("remat", msg)


# ---------------------------------------------------------------------------
# Neural-network model integration (relu + matmul + reductions)
# ---------------------------------------------------------------------------

class TestNNModel(unittest.TestCase):

    def test_mlp_forward_clean_no_crash(self):
        class MLP(bst.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = bst.nn.Linear(4, 8)
                self.l2 = bst.nn.Linear(8, 2)

            def __call__(self, x):
                return jax.nn.softmax(self.l2(jax.nn.relu(self.l1(x))))

        model = MLP()
        debug_nan(model, brainstate.random.rand(3, 4))


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

class TestControlFlow(unittest.TestCase):

    def test_cond_true_branch_nan(self):
        def fn(x):
            return jax.lax.cond(x[0] > 0.0, lambda y: jnp.log(y), lambda y: y * 2.0, x)

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([1.0, -1.0]))   # true branch -> log(-1)=NaN reaches output
        self.assertIn("log", str(ctx.exception))

    def test_cond_false_branch_clean(self):
        def fn(x):
            return jax.lax.cond(x[0] > 0.0, lambda y: jnp.log(y), lambda y: y * 2.0, x)

        debug_nan(fn, jnp.array([-1.0, 1.0, 2.0]))   # false branch taken, clean

    def test_switch_more_than_two_branches(self):
        def fn(i, x):
            return jax.lax.switch(i, [lambda y: y * 2.0, lambda y: jnp.log(y), lambda y: y + 1.0], x)

        with self.assertRaises(RuntimeError):
            debug_nan(fn, jnp.array(1), jnp.array([-1.0, 1.0]))

    def test_while_with_closure_clean(self):
        # body closes over `step` -> while primitive has body_nconsts=1 (old code crashed)
        def fn(x, step):
            return jax.lax.while_loop(lambda v: v < 5.0, lambda v: v + step, x)

        debug_nan(fn, jnp.array(0.0), jnp.array(1.0))

    def test_while_mixed_carry_clean(self):
        def fn(x):
            def cond_fn(carry):
                i, acc = carry
                return i < 5

            def body_fn(carry):
                i, acc = carry
                return i + 1, acc + x

            return jax.lax.while_loop(cond_fn, body_fn, (jnp.array(0, jnp.int32), jnp.array(0.0)))

        debug_nan(fn, jnp.array(2.0))

    def test_scan_body_nan(self):
        def fn(xs):
            return jax.lax.scan(lambda c, x: (c + jnp.log(x), c), 0.0, xs)[0]

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(fn, jnp.array([-1.0, 1.0, 2.0]))
        self.assertIn("log", str(ctx.exception))

    def test_scan_clean(self):
        def fn(xs):
            return jax.lax.scan(lambda c, x: (c + x, x * 2.0), 0.0, xs)[0]

        debug_nan(fn, jnp.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Inf / complex detection
# ---------------------------------------------------------------------------

class TestInfComplex(unittest.TestCase):

    def test_inf_times_zero_nan_detected(self):
        # inf*0 -> nan; old code masked this (input already had inf)
        with self.assertRaises(RuntimeError):
            debug_nan(lambda x: x * 0.0, jnp.array([jnp.inf]))

    def test_complex_nan_detected(self):
        with self.assertRaises(RuntimeError):
            debug_nan(lambda x: x / x, jnp.array([0.0 + 0.0j], dtype=jnp.complex64))

    def test_nan_input_reaches_output_flagged(self):
        # If the input already carries NaN and it reaches the output, flag it
        # (output is contaminated).
        with self.assertRaises(RuntimeError):
            debug_nan(lambda x: x * 2.0, jnp.array([1.0, jnp.nan, 3.0]))


# ---------------------------------------------------------------------------
# debug_nan_if
# ---------------------------------------------------------------------------

class TestDebugNanIf(unittest.TestCase):

    def test_false_predicate_no_raise(self):
        debug_nan_if(False, lambda x: jnp.log(x), jnp.array([-1.0, 1.0]))

    def test_true_predicate_raises(self):
        with self.assertRaises(RuntimeError):
            debug_nan_if(True, lambda x: jnp.log(x), jnp.array([-1.0, 1.0]))

    def test_jax_array_false_no_raise(self):
        debug_nan_if(jnp.array(False), lambda x: jnp.log(x), jnp.array([-1.0, 1.0]))

    def test_jax_array_true_raises(self):
        with self.assertRaises(RuntimeError):
            debug_nan_if(jnp.array(True), lambda x: jnp.log(x), jnp.array([-1.0, 1.0]))

    def test_jit_clean_no_raise(self):
        @jax.jit
        def f(x, trigger):
            debug_nan_if(trigger, lambda y: jnp.log(y), x)
            return x

        out = f(jnp.array([1.0, 2.0]), jnp.array(False))
        self.assertEqual(out.shape, (2,))

    def test_jit_with_nan_raises(self):
        @jax.jit
        def f(x, trigger):
            debug_nan_if(trigger, lambda y: jnp.log(y), x)
            return x

        with self.assertRaises(Exception):
            jax.block_until_ready(f(jnp.array([-1.0, 1.0]), jnp.array(True)))

    def test_jit_nan_present_but_trigger_false_is_silent(self):
        # NaN-prone data, but the trigger is False -> must stay silent.
        @jax.jit
        def f(x, trigger):
            debug_nan_if(trigger, lambda y: jnp.log(y), x)
            return x

        out = jax.block_until_ready(f(jnp.array([-1.0, 1.0]), jnp.array(False)))
        self.assertEqual(out.shape, (2,))

    def test_vmapped_predicate(self):
        # unvmap(..., 'any') collapses lanes; any True lane triggers.
        def f(x, trig):
            debug_nan_if(trig, lambda y: jnp.log(y), x)
            return x

        with self.assertRaises(Exception):
            jax.block_until_ready(
                jax.vmap(f)(jnp.array([-1.0, 1.0]), jnp.array([True, False]))
            )


# ---------------------------------------------------------------------------
# Stateful models
# ---------------------------------------------------------------------------

class TestStateIntegration(unittest.TestCase):

    def test_state_value_nan_detected_and_attributed(self):
        class Model(bst.graph.Node):
            def __init__(self):
                self.w = bst.State(jnp.array([1.0, -1.0]))

            def __call__(self, x):
                return jnp.log(self.w.value) + x   # log(-1)=NaN from the state

        with self.assertRaises(RuntimeError) as ctx:
            debug_nan(Model(), jnp.array([1.0, 2.0]))
        msg = str(ctx.exception)
        self.assertIn("log", msg)
        self.assertNotIn('_make_jaxpr.py', msg)

    def test_clean_stateful_no_raise(self):
        class Model(bst.graph.Node):
            def __init__(self):
                self.w = bst.State(jnp.array([1.0, 2.0]))

            def __call__(self, x):
                return self.w.value * x

        debug_nan(Model(), jnp.array([1.0, 2.0]))

    def test_state_write_nan_detected(self):
        class Model(bst.graph.Node):
            def __init__(self):
                self.s = bst.State(jnp.array([1.0]))

            def __call__(self, x):
                self.s.value = jnp.log(x)   # writes NaN to state
                return self.s.value.sum()

        with self.assertRaises(RuntimeError):
            debug_nan(Model(), jnp.array([-1.0]))

    def test_multiple_states_clean(self):
        class Model(bst.graph.Node):
            def __init__(self):
                self.a = bst.State(jnp.array([1.0]))
                self.b = bst.State(jnp.array([2.0]))

            def __call__(self, x):
                return self.a.value * self.b.value + x

        debug_nan(Model(), jnp.array([1.0]))


# ---------------------------------------------------------------------------
# DebugNan class interface
# ---------------------------------------------------------------------------

class TestDebugNanClass(unittest.TestCase):

    def test_check_raises(self):
        d = DebugNan(lambda x: jnp.log(x), jnp.array([-1.0]), phase='test')
        with self.assertRaises(RuntimeError) as ctx:
            d.check()
        self.assertIn("test", str(ctx.exception))

    def test_check_clean_no_raise(self):
        DebugNan(lambda x: x * 2.0, jnp.array([1.0])).check()

    def test_check_if_false_no_raise(self):
        DebugNan(lambda x: jnp.log(x), jnp.array([-1.0])).check_if(jnp.array(False))

    def test_check_if_true_raises(self):
        with self.assertRaises(RuntimeError):
            DebugNan(lambda x: jnp.log(x), jnp.array([-1.0])).check_if(jnp.array(True))


# ---------------------------------------------------------------------------
# GradientTransform integration (consumer of debug_nan_if via debug_nan=True)
# ---------------------------------------------------------------------------

class TestGradTransformIntegration(unittest.TestCase):

    def test_grad_debug_nan_clean_no_raise(self):
        w = bst.State(jnp.array([1.0, 2.0]))

        def loss(x):
            return jnp.sum((x * w.value) ** 2)

        g = bst.transform.grad(loss, grad_states=w, debug_nan=True)
        out = g(jnp.array([1.0, 2.0]))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

    def test_grad_debug_nan_backward_raises(self):
        w = bst.State(jnp.array([0.0, 4.0]))

        def loss(x):
            return jnp.sum(jnp.sqrt(w.value) * x)   # d/dw sqrt(0) -> inf

        g = bst.transform.grad(loss, grad_states=w, debug_nan=True)
        with self.assertRaises(Exception) as ctx:
            jax.block_until_ready(g(jnp.array([1.0, 1.0])))
        self.assertIn("NaN/Inf", str(ctx.exception))

    def test_grad_debug_nan_under_jit_clean_no_raise(self):
        w = bst.State(jnp.array([1.0, 2.0]))

        def loss(x):
            return jnp.sum((x * w.value) ** 2)

        @bst.transform.jit
        def step(x):
            return bst.transform.grad(loss, grad_states=w, debug_nan=True)(x)

        out = jax.block_until_ready(step(jnp.array([1.0, 2.0])))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

    def test_grad_debug_nan_under_jit_raises(self):
        w = bst.State(jnp.array([0.0, 4.0]))

        def loss(x):
            return jnp.sum(jnp.sqrt(w.value) * x)

        @bst.transform.jit
        def step(x):
            return bst.transform.grad(loss, grad_states=w, debug_nan=True)(x)

        with self.assertRaises(Exception):
            jax.block_until_ready(step(jnp.array([1.0, 1.0])))


if __name__ == '__main__':
    unittest.main()
