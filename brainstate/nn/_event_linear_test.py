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


import jax
import jax.numpy as jnp
import pytest

import braintools
import brainstate


class TestEventLinear:
    @pytest.mark.parametrize('bool_x', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test1(self, homo_w, bool_x):
        x = brainstate.random.rand(20) < 0.1
        if not bool_x:
            x = jnp.asarray(x, dtype=float)
        m = brainstate.nn.EventLinear(
            20, 40,
            1.5 if homo_w else braintools.init.KaimingUniform(),
            float_as_event=bool_x
        )
        y = m(x)
        print(y)

        assert (jnp.allclose(y, (x.sum() * m.weight.value) if homo_w else (x @ m.weight.value)))

    def test_grad_bool(self):
        n_in = 20
        n_out = 30
        x = brainstate.random.rand(n_in) < 0.3
        fn = brainstate.nn.EventLinear(n_in, n_out, braintools.init.KaimingUniform())

        with pytest.raises(TypeError):
            print(jax.grad(lambda x: fn(x).sum())(x))

    @pytest.mark.parametrize('bool_x', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vjp(self, bool_x, homo_w):
        n_in = 20
        n_out = 30
        if bool_x:
            x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)
        else:
            x = brainstate.random.rand(n_in)

        fn = brainstate.nn.EventLinear(
            n_in,
            n_out,
            1.5 if homo_w else braintools.init.KaimingUniform(),
            float_as_event=bool_x
        )
        w = fn.weight.value

        def f(x, w):
            fn.weight.value = w
            return fn(x).sum()

        r1 = brainstate.transform.grad(f, argnums=(0, 1))(x, w)

        # -------------------
        # TRUE gradients

        def f2(x, w):
            y = (x @ (jnp.ones([n_in, n_out]) * w)) if homo_w else (x @ w)
            return y.sum()

        r2 = jax.grad(f2, argnums=(0, 1))(x, w)
        assert (jnp.allclose(r1[0], r2[0]))

        if not jnp.allclose(r1[1], r2[1]):
            print(r1[1] - r2[1])

        assert (jnp.allclose(r1[1], r2[1]))

    @pytest.mark.parametrize('bool_x', [True, False])
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_jvp(self, bool_x, homo_w):
        n_in = 20
        n_out = 30
        if bool_x:
            x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)
        else:
            x = brainstate.random.rand(n_in)

        fn = brainstate.nn.EventLinear(
            n_in, n_out, 1.5 if homo_w else braintools.init.KaimingUniform(),
            float_as_event=bool_x
        )
        w = fn.weight.value

        def f(x, w):
            # restore_value: untracked functional injection — this runs under
            # raw jax.jvp (no brainstate equivalent) to verify the custom JVP
            # rule numerically
            fn.weight.restore_value(w)
            return fn(x)

        o1, r1 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        # -------------------
        # TRUE gradients

        def f2(x, w):
            y = (x @ (jnp.ones([n_in, n_out]) * w)) if homo_w else (x @ w)
            return y

        o2, r2 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert (jnp.allclose(o1, o2))
        assert (jnp.allclose(r1, r2))

    @pytest.mark.parametrize('float_as_event', [True, False])
    def test_homogeneous_matches_dense_nonbinary(self, float_as_event):
        """Regression for C4: the scalar-weight (homogeneous) path must agree with the
        dense/heterogeneous path for non-binary float input, for both float_as_event modes.

        With ``float_as_event=True`` the dense ``brainevent.EventArray`` path counts each
        nonzero entry as a unit event, so the homogeneous path must reduce by event count
        (``sum(spk != 0)``) rather than by spike value. With ``float_as_event=False`` both
        paths sum spike values.
        """
        n_in = 10
        n_out = 8
        weight = 1.5
        spk = jnp.asarray([0., 2., 0., 3., 0., 0., 1., 0., 0., 5.])

        # Homogeneous (scalar weight) path.
        homo = brainstate.nn.EventLinear(n_in, n_out, weight, float_as_event=float_as_event)
        y_homo = homo(spk)

        # Equivalent dense / heterogeneous path: a full weight matrix filled with the scalar.
        dense = brainstate.nn.EventLinear(
            n_in, n_out,
            braintools.init.Constant(weight),
            float_as_event=float_as_event,
        )
        y_dense = dense(spk)

        assert jnp.allclose(y_homo, y_dense), (
            f"float_as_event={float_as_event}: homogeneous {y_homo} != dense {y_dense}"
        )

        # Sanity-check the expected reduction explicitly.
        if float_as_event:
            expected_scalar = jnp.sum(spk != 0) * weight  # event count = 4 nonzeros
        else:
            expected_scalar = jnp.sum(spk) * weight        # value sum = 11
        assert jnp.allclose(y_homo, jnp.ones((n_out,)) * expected_scalar)
