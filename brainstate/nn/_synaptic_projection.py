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


from typing import Callable

import brainunit as u

from brainstate._compatible_import import brainevent
from brainstate.mixin import ParamDescriber, AlignPost, UpdateReturn
from ._dynamics import Dynamics, Projection
from ._projection import AlignPostProj, RawProj
from ._stp import ShortTermPlasticity
from ._synapse import Synapse
from ._synouts import SynOut

__all__ = [
    'align_pre_synapse',
    'align_post_synapse',
]


class align_pre_synapse(Projection):
    def __init__(
        self,
        pre: Dynamics,
        syn: Synapse | ParamDescriber[Synapse],
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


class align_post_synapse(Projection):
    def __init__(
        self,
        *spike_generator,
        comm: Callable,
        syn: AlignPost | ParamDescriber[AlignPost],
        out: SynOut | ParamDescriber[SynOut],
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
