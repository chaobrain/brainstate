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

from brainstate._state import ParamState
from brainstate.typing import Size, ArrayLike
from . import init as init
from ._module import Module

__all__ = [
    'EventLinear',
]


class EventLinear(Module):
    """

    Parameters
    ----------
    in_size : Size
        Number of pre-synaptic neurons, i.e., input size.
    out_size : Size
        Number of post-synaptic neurons, i.e., output size.
    weight : float or callable or jax.Array or brainunit.Quantity
        Maximum synaptic conductance.
    block_size : int, optional
        Block size for parallel computation.
    float_as_event : bool, optional
        Whether to treat float as event.
    name : str, optional
        Name of the module.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        weight: Union[Callable, ArrayLike],
        float_as_event: bool = True,
        block_size: int = 64,
        name: Optional[str] = None,
        param_type: type = ParamState,
    ):
        super().__init__(name=name)

        # network parameters
        self.in_size = in_size
        self.out_size = out_size
        self.float_as_event = float_as_event
        self.block_size = block_size

        # maximum synaptic conductance
        weight = init.param(weight, (self.in_size[-1], self.out_size[-1]), allow_none=False)
        self.weight = param_type(weight)

    def update(self, spk: jax.Array) -> Union[jax.Array, u.Quantity]:
        weight = self.weight.value
        if u.math.size(weight) == 1:
            # Homogeneous (scalar) weight: every post-synaptic neuron receives the
            # same total input. The reduction over ``spk`` must mirror the dense path
            # below so the two stay numerically consistent:
            #
            # - ``float_as_event=True``: the dense ``brainevent.EventArray`` path treats
            #   each nonzero entry as a unit event, so the *forward* value reduces by
            #   the event count (``sum(spk != 0)``). Its custom VJP, however, propagates
            #   the value-sum gradient w.r.t. ``spk`` (as ``spk @ weight`` would). We
            #   reproduce both: the forward equals the event count while the gradient
            #   flows through the value sum (the stop-gradient cancels in the forward
            #   pass but leaves a unit derivative on ``spk``).
            # - ``float_as_event=False``: the dense ``spk @ weight`` path sums spike
            #   *values*, so reduce by the value sum directly.
            if self.float_as_event:
                n_events = u.math.sum(spk != 0)
                value_sum = u.math.sum(spk)
                reduced = n_events + (value_sum - jax.lax.stop_gradient(value_sum))
            else:
                reduced = u.math.sum(spk)
            return u.math.ones(self.out_size) * (reduced * weight)

        if self.float_as_event:
            return brainevent.EventArray(spk) @ weight
        else:
            return spk @ weight
