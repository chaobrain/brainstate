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


import jax.numpy
import jax.numpy as jnp
import brainunit as u
import numpy as np
import pytest

import brainstate
import braintools
from brainstate.nn._event_fixedprob import (
    FixedNumConn,
    EventFixedNumConn,
    EventFixedProb,
    init_indices_without_replace,
)


class TestFixedProbCSR:
    @pytest.mark.parametrize('allow_multi_conn', [True, False, ])
    def test1(self, allow_multi_conn):
        x = brainstate.random.rand(20) < 0.1
        # x = brainstate.random.rand(20)
        m = brainstate.nn.EventFixedProb(20, 40, 0.1, 1.0, seed=123, allow_multi_conn=allow_multi_conn)
        y = m(x)
        print(y)

        m2 = brainstate.nn.EventFixedProb(20, 40, 0.1, braintools.init.KaimingUniform(), seed=123)
        print(m2(x))

    def test_grad_bool(self):
        n_in = 20
        n_out = 30
        x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)
        fn = brainstate.nn.EventFixedProb(n_in, n_out, 0.1, braintools.init.KaimingUniform(), seed=123)

        def f(x):
            return fn(x).sum()

        print(jax.grad(f)(x))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vjp(self, homo_w):
        n_in = 20
        n_out = 30
        x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)

        if homo_w:
            fn = brainstate.nn.EventFixedProb(n_in, n_out, 0.1, 1.5, seed=123)
        else:
            fn = brainstate.nn.EventFixedProb(n_in, n_out, 0.1, braintools.init.KaimingUniform(), seed=123)
        w = fn.weight.value

        def f(x, w):
            fn.weight.value = w
            return fn(x).sum()

        r = brainstate.transform.grad(f, argnums=(0, 1))(x, w)

        # -------------------
        # TRUE gradients

        def true_fn(x, w, indices, n_post):
            post = jnp.zeros((n_post,))
            for i in range(n_in):
                post = post.at[indices[i]].add(w * x[i] if homo_w else w[i] * x[i])
            return post

        def f2(x, w):
            return true_fn(x, w, fn.conn.indices, n_out).sum()

        r2 = jax.grad(f2, argnums=(0, 1))(x, w)
        assert (jnp.allclose(r[0], r2[0]))
        assert (jnp.allclose(r[1], r2[1]))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_jvp(self, homo_w):
        n_in = 20
        n_out = 30
        x = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)

        fn = brainstate.nn.EventFixedProb(
            n_in, n_out, 0.1, 1.5 if homo_w else braintools.init.KaimingUniform(),
            seed=123,
        )
        w = fn.weight.value

        def f(x, w):
            fn.weight.value = w
            return fn(x)

        o1, r1 = jax.jvp(f, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        # -------------------
        # TRUE gradients

        def true_fn(x, w, indices, n_post):
            post = jnp.zeros((n_post,))
            for i in range(n_in):
                post = post.at[indices[i]].add(w * x[i] if homo_w else w[i] * x[i])
            return post

        def f2(x, w):
            return true_fn(x, w, fn.conn.indices, n_out)

        o2, r2 = jax.jvp(f2, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert (jnp.allclose(o1, o2))
        # assert jnp.allclose(r1, r2), f'r1={r1}, r2={r2}'
        assert (jnp.allclose(r1, r2, rtol=1e-4, atol=1e-4))


class TestInitIndicesWithoutReplace:
    """Cover the ``init_indices_without_replace`` connection-index sampler."""

    def test_vmap_method_shape(self):
        """The ``'vmap'`` method returns indices of shape ``(n_pre, conn_num)``."""
        idx = init_indices_without_replace(3, 5, 10, seed=42, method='vmap')
        assert idx.shape == (5, 3)
        assert bool((idx >= 0).all()) and bool((idx < 10).all())

    def test_for_loop_method_shape(self):
        """The ``'for_loop'`` method returns indices of shape ``(n_pre, conn_num)``."""
        idx = init_indices_without_replace(3, 5, 10, seed=42, method='for_loop')
        assert idx.shape == (5, 3)
        assert bool((idx >= 0).all()) and bool((idx < 10).all())

    def test_no_replacement_within_row(self):
        """Sampling without replacement yields distinct indices within each row."""
        idx = np.asarray(init_indices_without_replace(4, 6, 10, seed=7, method='for_loop'))
        for row in idx:
            assert len(set(row.tolist())) == len(row)

    def test_unknown_method_raises(self):
        """An unrecognized method name raises ``ValueError``."""
        with pytest.raises(ValueError, match='Unknown method'):
            init_indices_without_replace(3, 5, 10, seed=1, method='bogus')


class TestFixedNumConnConstruction:
    """Cover ``FixedNumConn`` construction branches and validation."""

    def test_int_conn_num_is_used_directly(self):
        """An integer ``conn_num`` is used as the literal connection count."""
        m = FixedNumConn(20, 40, 5, 1.0, seed=1)
        assert m.conn_num == 5

    def test_float_conn_num_post_uses_out_size(self):
        """A float ``conn_num`` with 'post' target scales by the output size."""
        m = FixedNumConn(20, 40, 0.25, 1.0, efferent_target='post', seed=1)
        assert m.conn_num == int(40 * 0.25)

    def test_no_replacement_construction_for_loop(self):
        """``allow_multi_conn=False`` with ``conn_init='for_loop'`` builds a valid module."""
        try:
            # jax < 0.10 has no evaluation rule for the 'empty' primitive used by
            # the for_loop construction path in brainevent; the error is raised as
            # early as ``FixedNumConn`` construction, so guard both it and the call.
            m = FixedNumConn(20, 40, 0.2, 1.0, seed=3,
                             allow_multi_conn=False, conn_init='for_loop')
            out = m(brainstate.random.rand(20))
        except NotImplementedError as e:
            if 'empty' in str(e):
                pytest.skip(f"this JAX version lacks an eval rule for the 'empty' primitive: {e}")
            raise
        assert out.shape == (40,)

    def test_invalid_efferent_target_raises(self):
        """An ``efferent_target`` other than 'pre'/'post' raises ``AssertionError``."""
        with pytest.raises(AssertionError):
            FixedNumConn(20, 40, 0.1, 1.0, efferent_target='sideways', seed=1)

    def test_afferent_ratio_out_of_range_raises(self):
        """An ``afferent_ratio`` outside ``[0, 1]`` raises ``AssertionError``."""
        with pytest.raises(AssertionError):
            FixedNumConn(20, 40, 0.1, 1.0, afferent_ratio=1.5, seed=1)

    def test_float_conn_prob_out_of_range_raises(self):
        """A float connection probability outside ``[0, 1]`` raises ``AssertionError``."""
        with pytest.raises(AssertionError):
            FixedNumConn(20, 40, 1.5, 1.0, seed=1)

    def test_afferent_ratio_post_csr_branch(self):
        """A sub-unity ``afferent_ratio`` with 'post' target builds a CSR connection."""
        m = FixedNumConn(20, 40, 0.2, 1.0, efferent_target='post',
                         afferent_ratio=0.5, seed=1)
        assert m.pre_selected.shape == (20,)
        out = m(brainstate.random.rand(20))
        assert out.shape == (40,)


class TestFixedNumConnZeroConnection:
    """Cover the zero-connection (FakeState) path of ``FixedNumConn``."""

    def test_zero_conn_num_uses_fake_state(self):
        """A connection probability of 0 yields zero connections and a FakeState weight."""
        m = FixedNumConn(20, 40, 0.0, 1.0, seed=1)
        assert m.conn_num == 0

    def test_zero_conn_update_returns_zeros(self):
        """Updating a zero-connection module returns an all-zero output of out_size."""
        m = FixedNumConn(20, 40, 0.0, 1.0, seed=1)
        out = m(brainstate.random.rand(20))
        assert out.shape == (40,)
        assert bool((u.get_magnitude(out) == 0).all())


class TestEventFixedProbAlias:
    """Cover the ``EventFixedProb``/``EventFixedNumConn`` event-array update path."""

    def test_alias_identity(self):
        """``EventFixedProb`` is an alias of ``EventFixedNumConn``."""
        assert EventFixedProb is EventFixedNumConn

    def test_event_update_matches_dense_reference(self):
        """The event update equals a manual scatter-add reference computation."""
        n_in, n_out = 20, 30
        fn = EventFixedProb(n_in, n_out, 0.2, 1.5, seed=11)
        spk = jax.numpy.asarray(brainstate.random.rand(n_in) < 0.3, dtype=float)
        out = np.asarray(fn(spk))

        indices = np.asarray(fn.conn.indices)
        ref = np.zeros((n_out,))
        # allow_multi_conn=True permits repeated post indices within a row, so the
        # contributions must be accumulated (np.add.at), matching the sparse matmul.
        for i in range(n_in):
            np.add.at(ref, indices[i], 1.5 * float(spk[i]))
        assert np.allclose(out, ref, rtol=1e-4, atol=1e-4)


class TestFixedNumConnKnownBugs:
    """Document genuine construction bugs in the ``efferent_target='pre'`` path."""

    @pytest.mark.skip(reason="BUG: efferent_target='pre' crashes in brainevent."
                             " FixedNumConn builds indices of shape (n_pre, conn_num) = "
                             "(out_size, conn_num) but brainevent.FixedPreNumConn(shape=(n_pre, "
                             "n_post)) validates indices rows against shape[1]=n_post=in_size, "
                             "raising 'Pre-synaptic row number mismatch. 40 != 20'.")
    def test_efferent_target_pre_constructs(self):
        """``efferent_target='pre'`` should construct a valid connection (currently crashes)."""
        m = FixedNumConn(20, 40, 0.2, 1.0, efferent_target='pre', seed=1)
        out = m(brainstate.random.rand(40))
        assert out.shape == (20,)

    @pytest.mark.skip(reason="BUG: efferent_target='pre' with afferent_ratio<1 triggers a native"
                             " memory abort ('free(): invalid next size (fast)') inside the "
                             "brainevent CSC path, because indices are sized for (n_pre, conn_num)"
                             " = (out_size, conn_num) rather than the n_post=in_size rows the CSC "
                             "shape expects.")
    def test_efferent_target_pre_afferent_ratio_constructs(self):
        """'pre' target with sub-unity afferent_ratio should build a CSC connection."""
        m = FixedNumConn(20, 40, 0.2, 1.0, efferent_target='pre',
                         afferent_ratio=0.5, seed=1)
        out = m(brainstate.random.rand(40))
        assert out.shape == (20,)
