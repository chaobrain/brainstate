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


from typing import Union, Callable, Optional

import brainevent
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainstate import random, transform, environ
from brainstate._state import ParamState, FakeState
from brainstate.transform import for_loop
from brainstate.typing import Size, ArrayLike
from . import init as init
from ._module import Module

__all__ = [
    'FixedNumConn',
    'EventFixedNumConn',
    'EventFixedProb',
]


def init_indices_without_replace(
    conn_num: int,
    n_pre: int,
    n_post: int,
    seed: int | None,
    method: str
):
    rng = random.default_rng(seed)

    if method == 'vmap':
        @transform.vmap(axis_size=n_pre)
        def rand_indices():
            return rng.choice(n_post, size=(conn_num,), replace=False)

        return rand_indices()

    elif method == 'for_loop':
        return for_loop(
            lambda *args: rng.choice(n_post, size=(conn_num,), replace=False),
            length=n_pre
        )

    else:
        raise ValueError(f"Unknown method: {method}")


class FixedNumConn(Module):
    """
    The ``FixedNumConn`` module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    in_size : Size
        Number of pre-synaptic neurons, i.e., input size.
    out_size : Size
        Number of post-synaptic neurons, i.e., output size.
    conn_num : float, int
        If it is a float, representing the probability of connection, i.e., connection probability.

        If it is an integer, representing the number of connections.
    conn_weight : float or callable or jax.Array or brainunit.Quantity
        Maximum synaptic conductance, i.e., synaptic weight.
    efferent_target : str, optional
        The target of the connection. Default is 'post', meaning that each pre-synaptic neuron connects to
        a fixed number of post-synaptic neurons. The connection number is determined by the value of ``n_conn``.

        If 'pre', each post-synaptic neuron connects to a fixed number of pre-synaptic neurons.
    conn_init : str, optional
        The initialization method of the connection weight. Default is 'vmap', meaning that the connection weight
        is initialized by parallelized across multiple threads.

        If 'for_loop', the connection weight is initialized by a for loop.
    allow_multi_conn : bool, optional
        Whether multiple connections are allowed from a single pre-synaptic neuron.
        Default is True, meaning that a value of ``a`` can be selected multiple times.
    seed: int, optional
        Random seed. Default is None. If None, the default random seed will be used.
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        conn_num: Union[int, float],
        conn_weight: Union[Callable, ArrayLike],
        efferent_target: str = 'post',  # 'pre' or 'post'
        afferent_ratio: Union[int, float] = 1.,
        allow_multi_conn: bool = True,
        seed: Optional[int] = None,
        name: Optional[str] = None,
        conn_init: str = 'vmap',  # 'vmap' or 'for_loop'
        param_type: type = ParamState,
    ):
        super().__init__(name=name)

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.efferent_target = efferent_target
        assert efferent_target in ('pre', 'post'), 'The target of the connection must be either "pre" or "post".'
        assert 0. <= afferent_ratio <= 1., 'Afferent ratio must be in [0, 1].'
        if isinstance(conn_num, float):
            assert 0. <= conn_num <= 1., 'Connection probability must be in [0, 1].'
            conn_num = (int(self.out_size[-1] * conn_num)
                        if efferent_target == 'post' else
                        int(self.in_size[-1] * conn_num))
        assert isinstance(conn_num, int), 'Connection number must be an integer.'
        self.conn_num = conn_num
        self.seed = seed
        self.allow_multi_conn = allow_multi_conn

        # connections
        if self.conn_num >= 1:
            if self.efferent_target == 'post':
                n_post = self.out_size[-1]
                n_pre = self.in_size[-1]
            else:
                n_post = self.in_size[-1]
                n_pre = self.out_size[-1]

            with jax.ensure_compile_time_eval():
                if allow_multi_conn:
                    rng = np.random if seed is None else np.random.RandomState(seed)
                    indices = rng.randint(0, n_post, size=(n_pre, self.conn_num))
                else:
                    indices = init_indices_without_replace(self.conn_num, n_pre, n_post, seed, conn_init)
                indices = u.math.asarray(indices, dtype=environ.ditype())

            if afferent_ratio == 1.:
                conn_weight = u.math.asarray(init.param(conn_weight, (n_pre, self.conn_num), allow_none=False))
                self.weight = param_type(conn_weight)
                csr = (
                    brainevent.FixedPostNumConn((conn_weight, indices), shape=(n_pre, n_post))
                    if self.efferent_target == 'post' else
                    brainevent.FixedPreNumConn((conn_weight, indices), shape=(n_pre, n_post))
                )
                self.conn = csr

            else:
                self.pre_selected = np.random.random(n_pre) < afferent_ratio
                indices = indices[self.pre_selected].flatten()
                conn_weight = u.math.asarray(init.param(conn_weight, (indices.size,), allow_none=False))
                self.weight = param_type(conn_weight)
                indptr = (jnp.arange(1, n_pre + 1) * self.conn_num -
                          jnp.cumsum(~self.pre_selected) * self.conn_num)
                indptr = jnp.insert(indptr, 0, 0)  # insert 0 at the beginning
                csr = (
                    brainevent.CSR((conn_weight, indices, indptr), shape=(n_pre, n_post))
                    if self.efferent_target == 'post' else
                    brainevent.CSC((conn_weight, indices, indptr), shape=(n_pre, n_post))
                )
                self.conn = csr

        else:
            conn_weight = u.math.asarray(init.param(conn_weight, (), allow_none=False))
            self.weight = FakeState(conn_weight)

    def update(self, x) -> Union[jax.Array, u.Quantity]:
        if self.conn_num >= 1:
            csr = self.conn.with_data(self.weight.value)
            return x @ csr
        else:
            weight = self.weight.value
            r = u.math.zeros(x.shape[:-1] + (self.out_size[-1],), dtype=weight.dtype)
            return u.maybe_decimal(u.Quantity(r, unit=u.get_unit(weight), dtype=environ.dftype()))


class EventFixedNumConn(FixedNumConn):
    """
    The FixedProb module implements a fixed probability connection with CSR sparse data structure.

    Parameters
    ----------
    in_size : Size
        Number of pre-synaptic neurons, i.e., input size.
    out_size : Size
        Number of post-synaptic neurons, i.e., output size.
    conn_num : float, int
        If it is a float, representing the probability of connection, i.e., connection probability.

        If it is an integer, representing the number of connections.
    conn_weight : float or callable or jax.Array or brainunit.Quantity
        Maximum synaptic conductance, i.e., synaptic weight.
    conn_target : str, optional
        The target of the connection. Default is 'post', meaning that each pre-synaptic neuron connects to
        a fixed number of post-synaptic neurons. The connection number is determined by the value of ``n_conn``.

        If 'pre', each post-synaptic neuron connects to a fixed number of pre-synaptic neurons.
    conn_init : str, optional
        The initialization method of the connection weight. Default is 'vmap', meaning that the connection weight
        is initialized by parallelized across multiple threads.

        If 'for_loop', the connection weight is initialized by a for loop.
    allow_multi_conn : bool, optional
        Whether multiple connections are allowed from a single pre-synaptic neuron.
        Default is True, meaning that a value of ``a`` can be selected multiple times.
    seed: int, optional
        Random seed. Default is None. If None, the default random seed will be used.
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainstate.nn'

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        return super().update(
            brainevent.EventArray(spk)
        )


EventFixedProb = EventFixedNumConn
