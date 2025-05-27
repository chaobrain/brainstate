# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
# -*- coding: utf-8 -*-


from typing import Callable, Union

import brainunit as u

from brainstate._compatible_import import brainevent
from brainstate.mixin import ParamDescriber, AlignPost, UpdateReturn
from ._dynamics import Dynamics, Projection
from ._projection import AlignPostProj, RawProj
from ._stp import ShortTermPlasticity
from ._synapse import Synapse
from ._synouts import SynOut

__all__ = [
    'align_pre_projection',
    'align_post_projection',
]


class align_pre_projection(Projection):
    """
    Represents a pre-synaptic alignment projection mechanism.

    This class inherits from the `Projection` base class and is designed to
    manage the pre-synaptic alignment process in neural network simulations.
    It takes into account pre-synaptic dynamics, synaptic properties, delays,
    communication functions, synaptic outputs, post-synaptic dynamics, and
    short-term plasticity.

    Attributes:
        pre (Dynamics): The pre-synaptic dynamics object.
        syn (Synapse): The synaptic object after pre-synaptic alignment.
        delay (u.Quantity[u.second]): The output delay from the synapse.
        projection (RawProj): The raw projection object handling communication,
            output, and post-synaptic dynamics.
        stp (ShortTermPlasticity, optional): The short-term plasticity object,
            defaults to None.
    """

    def __init__(
        self,
        pre: Dynamics,
        syn: Union[Synapse, ParamDescriber[Synapse]],
        delay: u.Quantity[u.second] | None,
        comm: Callable,
        out: SynOut,
        post: Dynamics,
        stp: ShortTermPlasticity = None,
    ):
        super().__init__()
        pre = pre
        syn: Synapse = pre.align_pre(syn)
        assert isinstance(syn, UpdateReturn), "Synapse must implement UpdateReturn interface"
        # require "syn" implement the "update_return()" function
        self.delay = syn.output_delay(delay)
        self.projection = RawProj(comm=comm, out=out, post=post)
        self.stp = stp

    def update(self):
        x = self.delay()
        if self.stp is not None:
            x = self.stp(x)
        return self.projection(x)


class align_post_projection(Projection):
    """
    Represents a post-synaptic alignment projection mechanism.

    This class inherits from the `Projection` base class and is designed to
    manage the post-synaptic alignment process in neural network simulations.
    It takes into account spike generators, communication functions, synaptic
    properties, synaptic outputs, post-synaptic dynamics, and short-term plasticity.

    Args:
        *spike_generator: Callable(s) that generate spike events or transform input spikes.
        comm (Callable): Communication function for the projection.
        syn (Union[AlignPost, ParamDescriber[AlignPost]]): The post-synaptic alignment object or its parameter describer.
        out (Union[SynOut, ParamDescriber[SynOut]]): The synaptic output object or its parameter describer.
        post (Dynamics): The post-synaptic dynamics object.
        stp (ShortTermPlasticity, optional): The short-term plasticity object, defaults to None.

    """
    def __init__(
        self,
        *spike_generator,
        comm: Callable,
        syn: Union[AlignPost, ParamDescriber[AlignPost]],
        out: Union[SynOut, ParamDescriber[SynOut]],
        post: Dynamics,
        stp: ShortTermPlasticity = None,
    ):
        super().__init__()
        self.spike_generator = spike_generator
        self.projection = AlignPostProj(comm=comm, syn=syn, out=out, post=post)
        self.stp = stp

    def update(self, *x):
        for fun in self.spike_generator:
            x = fun(*x)
            if isinstance(x, (tuple, list)):
                x = tuple(x)
            else:
                x = (x,)
        assert len(x) == 1, "Spike generator must return a single value or a tuple/list of values"
        x = brainevent.BinaryArray(x[0])  # Ensure input is a BinaryFloat for spike generation
        if self.stp is not None:
            x = brainevent.MaskedFloat(self.stp(x))  # Ensure STP output is a MaskedFloat
        return self.projection(x)


class align_pre_ltp(Projection):
    pass


class align_post_ltp(Projection):
    pass
