# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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
from __future__ import annotations

import brainunit as u
import jax
import numpy as np
from typing import Union, Optional, Sequence, Callable

from brainstate import environ, init, random
from brainstate._state import ShortTermState
from brainstate._state import State, maybe_state
from brainstate.compile import while_loop
from brainstate.nn._dynamics._dynamics_base import Dynamics, Prefetch
from brainstate.nn._module import Module
from brainstate.typing import ArrayLike, Size, DTypeLike

__all__ = [
    'SpikeTime',
    'PoissonSpike',
    'PoissonEncoder',
    'PoissonInput',
    'poisson_input',
]


class SpikeTime(Dynamics):
    """The input neuron group characterized by spikes emitting at given times.

    >>> # Get 2 neurons, firing spikes at 10 ms and 20 ms.
    >>> SpikeTime(2, times=[10, 20])
    >>> # or
    >>> # Get 2 neurons, the neuron 0 fires spikes at 10 ms and 20 ms.
    >>> SpikeTime(2, times=[10, 20], indices=[0, 0])
    >>> # or
    >>> # Get 2 neurons, neuron 0 fires at 10 ms and 30 ms, neuron 1 fires at 20 ms.
    >>> SpikeTime(2, times=[10, 20, 30], indices=[0, 1, 0])
    >>> # or
    >>> # Get 2 neurons; at 10 ms, neuron 0 fires; at 20 ms, neuron 0 and 1 fire;
    >>> # at 30 ms, neuron 1 fires.
    >>> SpikeTime(2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])

    Parameters
    ----------
    in_size : int, tuple, list
        The neuron group geometry.
    indices : list, tuple, ArrayType
        The neuron indices at each time point to emit spikes.
    times : list, tuple, ArrayType
        The time points which generate the spikes.
    name : str, optional
        The name of the dynamic system.
    """

    def __init__(
        self,
        in_size: Size,
        indices: Union[Sequence, ArrayLike],
        times: Union[Sequence, ArrayLike],
        spk_type: DTypeLike = bool,
        name: Optional[str] = None,
        need_sort: bool = True,
    ):
        super().__init__(in_size=in_size, name=name)

        # parameters
        if len(indices) != len(times):
            raise ValueError(f'The length of "indices" and "times" must be the same. '
                             f'However, we got {len(indices)} != {len(times)}.')
        self.num_times = len(times)
        self.spk_type = spk_type

        # data about times and indices
        self.times = u.math.asarray(times)
        self.indices = u.math.asarray(indices, dtype=environ.ditype())
        if need_sort:
            sort_idx = u.math.argsort(self.times)
            self.indices = self.indices[sort_idx]
            self.times = self.times[sort_idx]

    def init_state(self, *args, **kwargs):
        self.i = ShortTermState(-1)

    def reset_state(self, batch_size=None, **kwargs):
        self.i.value = -1

    def update(self):
        t = environ.get('t')

        def _cond_fun(spikes):
            i = self.i.value
            return u.math.logical_and(i < self.num_times, t >= self.times[i])

        def _body_fun(spikes):
            i = self.i.value
            spikes = spikes.at[..., self.indices[i]].set(True)
            self.i.value += 1
            return spikes

        spike = u.math.zeros(self.varshape, dtype=self.spk_type)
        spike = while_loop(_cond_fun, _body_fun, spike)
        return spike


class PoissonSpike(Dynamics):
    """
    Poisson Neuron Group.
    """

    def __init__(
        self,
        in_size: Size,
        freqs: Union[ArrayLike, Callable],
        spk_type: DTypeLike = bool,
        name: Optional[str] = None,
    ):
        super().__init__(in_size=in_size, name=name)

        self.spk_type = spk_type

        # parameters
        self.freqs = init.param(freqs, self.varshape, allow_none=False)

    def update(self):
        spikes = random.rand(self.varshape) <= (self.freqs * environ.get_dt())
        spikes = u.math.asarray(spikes, dtype=self.spk_type)
        return spikes


class PoissonEncoder(Dynamics):
    """
    Poisson Neuron Group.
    """

    def __init__(
        self,
        in_size: Size,
        spk_type: DTypeLike = bool,
        name: Optional[str] = None,
    ):
        super().__init__(in_size=in_size, name=name)
        self.spk_type = spk_type

    def update(self, freqs: ArrayLike):
        spikes = random.rand(*self.varshape) <= (freqs * environ.get_dt())
        spikes = u.math.asarray(spikes, dtype=self.spk_type)
        return spikes


class PoissonInput(Module):
    """
    Poisson Input to the given :py:class:`brainstate.State`.

    Adds independent Poisson input to a target variable. For large
    numbers of inputs, this is much more efficient than creating a
    `PoissonGroup`. The synaptic events are generated randomly during the
    simulation and are not preloaded and stored in memory. All the inputs must
    target the same variable, have the same frequency and same synaptic weight.
    All neurons in the target variable receive independent realizations of
    Poisson spike trains.

    Args:
      target: The variable that is targeted by this input. Should be an instance of :py:class:`~.Variable`.
      num_input: The number of inputs.
      freq: The frequency of each of the inputs. Must be a scalar.
      weight: The synaptic weight. Must be a scalar.
      name: The target name.
    """

    def __init__(
        self,
        target: Prefetch,
        indices: Union[np.ndarray, jax.Array],
        num_input: int,
        freq: Union[int, float],
        weight: Union[int, float],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.target = target
        self.indices = indices
        self.num_input = num_input
        self.freq = freq
        self.weight = weight

    def update(self):
        target_state = getattr(self.target.module, self.target.item)

        # generate Poisson input
        poisson_input(
            self.freq,
            self.num_input,
            self.weight,
            target_state,
            self.indices,
        )


def poisson_input(
    freq: u.Quantity[u.Hz],
    num_input: int,
    weight: u.Quantity,
    target: State,
    indices: Optional[Union[np.ndarray, jax.Array]] = None,
    refractory: Optional[Union[jax.Array]] = None,
):
    """
    Generates Poisson-distributed input spikes to a target state variable.

    This function simulates Poisson input to a given state, updating the target
    variable with generated spikes based on the specified frequency, number of inputs,
    and synaptic weight. The input can be applied to specific indices of the target
    or to the entire target if indices are not provided.

    Parameters
    ----------
    freq : u.Quantity[u.Hz]
        The frequency of the Poisson input in Hertz.
    num_input : int
        The number of input channels or neurons generating the Poisson spikes.
    weight : u.Quantity
        The synaptic weight applied to each spike.
    target : State
        The target state variable to which the Poisson input is applied.
    indices : Optional[Union[np.ndarray, jax.Array]], optional
        Specific indices of the target to apply the input. If None, the input is applied
        to the entire target.
    refractory : Optional[Union[jax.Array]], optional
        A boolean array indicating which parts of the target are in a refractory state
        and should not be updated. Should be the same length as the target.

    Returns
    -------
    None
        The function updates the target state in place with the generated Poisson input.
    """
    freq = maybe_state(freq)
    weight = maybe_state(weight)

    assert isinstance(target, State), 'The target must be a State.'
    p = freq * environ.get_dt()
    p = p.to_decimal() if isinstance(p, u.Quantity) else p
    a = num_input * p
    b = num_input * (1 - p)
    tar_val = target.value
    cond = u.math.logical_and(a > 5, b > 5)

    if indices is None:
        # generate Poisson input
        branch1 = jax.tree.map(
            lambda tar: random.normal(
                a,
                b * p,
                tar.shape,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity
        )
        branch2 = jax.tree.map(
            lambda tar: random.binomial(
                num_input,
                p,
                tar.shape,
                check_valid=False,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity,
        )

        inp = jax.tree.map(
            lambda b1, b2: u.math.where(cond, b1, b2),
            branch1,
            branch2,
            is_leaf=u.math.is_quantity,
        )

        # inp = jax.lax.cond(
        #     cond,
        #     lambda rand_key: jax.tree.map(
        #         lambda tar: random.normal(
        #             a,
        #             b * p,
        #             tar.shape,
        #             key=rand_key,
        #             dtype=tar.dtype
        #         ),
        #         tar_val,
        #         is_leaf=u.math.is_quantity
        #     ),
        #     lambda rand_key: jax.tree.map(
        #         lambda tar: random.binomial(
        #             num_input,
        #             p,
        #             tar.shape,
        #             key=rand_key,
        #             check_valid=False,
        #             dtype=tar.dtype
        #         ),
        #         tar_val,
        #         is_leaf=u.math.is_quantity,
        #     ),
        #     random.split_key()
        # )

        # update target variable
        data = jax.tree.map(
            lambda tar, x: tar + x * weight,
            target.value,
            inp,
            is_leaf=u.math.is_quantity
        )

    else:
        # generate Poisson input
        branch1 = jax.tree.map(
            lambda tar: random.normal(
                a,
                b * p,
                tar[indices].shape,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity
        )
        branch2 = jax.tree.map(
            lambda tar: random.binomial(
                num_input,
                p,
                tar[indices].shape,
                # check_valid=False,
                dtype=tar.dtype
            ),
            tar_val,
            is_leaf=u.math.is_quantity
        )

        inp = jax.tree.map(
            lambda b1, b2: u.math.where(cond, b1, b2),
            branch1,
            branch2,
            is_leaf=u.math.is_quantity,
        )

        # inp = jax.lax.cond(
        #     cond,
        #     lambda rand_key: jax.tree.map(
        #         lambda tar: random.normal(
        #             a,
        #             b * p,
        #             tar[indices].shape,
        #             key=rand_key,
        #             dtype=tar.dtype
        #         ),
        #         tar_val,
        #         is_leaf=u.math.is_quantity
        #     ),
        #     lambda rand_key: jax.tree.map(
        #         lambda tar: random.binomial(
        #             num_input,
        #             p,
        #             tar[indices].shape,
        #             key=rand_key,
        #             check_valid=False,
        #             dtype=tar.dtype
        #         ),
        #         tar_val,
        #         is_leaf=u.math.is_quantity
        #     ),
        #     random.split_key()
        # )

        # update target variable
        data = jax.tree.map(
            lambda x, tar: tar.at[indices].add(x * weight),
            inp,
            tar_val,
            is_leaf=u.math.is_quantity
        )

    if refractory is not None:
        target.value = jax.tree.map(
            lambda x, tar: u.math.where(refractory, tar, x),
            data,
            tar_val,
            is_leaf=u.math.is_quantity
        )
    else:
        target.value = data
