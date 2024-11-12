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

from typing import Union, Callable

from brainstate._state import State
from brainstate.mixin import AlignPost, ParamDescriber, BindCondData, JointTypes
from brainstate.nn._collective_ops import call_order
from brainstate.nn._module import Module
from brainstate.util._others import get_unique_name
from ._dynamics_base import Dynamics, maybe_init_prefetch, Prefetch, PrefetchDelayAt
from ._synouts import SynOut

__all__ = [
    'AlignPostProj',
    'DeltaProj',
    'CurrentProj',
]


class Interaction(Module):
    __module__ = 'brainstate.nn'


def _check_modules(*modules):
    # checking modules
    for module in modules:
        if not callable(module) and not isinstance(module, State):
            raise TypeError(
                f'The module should be a callable function or a brainstate.State, but got {module}.'
            )
    return tuple(modules)


def call_module(module, *args, **kwargs):
    if callable(module):
        return module(*args, **kwargs)
    elif isinstance(module, State):
        return module.value
    else:
        raise TypeError(
            f'The module should be a callable function or a brainstate.State, but got {module}.'
        )


def is_instance(x, cls) -> bool:
    return isinstance(x, cls)


def get_post_repr(syn, out):
    return f'{syn.identifier} // {out.identifier}'


def align_post_add_bef_update(
    syn_desc: ParamDescriber[AlignPost],
    out_desc: ParamDescriber[BindCondData],
    post: Dynamics,
    proj_name: str
):
    # synapse and output initialization
    _post_repr = get_post_repr(syn_desc, out_desc)
    if not post._has_before_update(_post_repr):
        syn_cls = syn_desc()
        out_cls = out_desc()

        # synapse and output initialization
        post.add_current_input(proj_name, out_cls)
        post._add_before_update(_post_repr, _AlignPost(syn_cls, out_cls))
    syn = post._get_before_update(_post_repr).syn
    out = post._get_before_update(_post_repr).out
    return syn, out


class _AlignPost(Module):
    def __init__(
        self,
        syn: Dynamics,
        out: BindCondData
    ):
        super().__init__()
        self.syn = syn
        self.out = out

    def update(self, *args, **kwargs):
        self.out.bind_cond(self.syn(*args, **kwargs))


class AlignPostProj(Interaction):
    """
    Full-chain synaptic projection with the align-post reduction and the automatic synapse merging.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``delay`` -> ``comm`` -> ``syn`` -> ``out`` -> ``post``.

    The ``align-post`` means that the synaptic variables have the same dimension as the post-synaptic neuron group.

    The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
    parameters (such like time constants) will also share the same synaptic variables.

    All align-post projection models prefer to use the event-driven computation mode. This means that the
    ``comm`` model should be the event-driven model.

    Moreover, it's worth noting that ``FullProjAlignPost`` has a different updating order with all align-pre
    projection models. The updating order of align-post projections is ``spikes`` -> ``comm`` -> ``syn`` -> ``out``.
    While, the updating order of all align-pre projection models is usually ``spikes`` -> ``syn`` -> ``comm`` -> ``out``.


    #  brainstate.nn.AlignPostProj(
    #       LIF().prefetch('V').delay.at('I'), bst.surrogate.ReluGrad(), comm, syn, out, post
    #  )

    Args:
      pre: The pre-synaptic neuron group.
      delay: The synaptic delay.
      comm: The synaptic communication.
      syn: The synaptic dynamics.
      out: The synaptic output.
      post: The post-synaptic neuron group.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        *modules,
        comm: Callable,
        syn: Union[ParamDescriber[AlignPost], AlignPost],
        out: Union[ParamDescriber[SynOut], SynOut],
        post: Dynamics,
    ):
        super().__init__(name=get_unique_name(self.__class__.__name__))

        # checking modules
        self.modules = _check_modules(*modules)

        # checking communication model
        if not callable(comm):
            raise TypeError(
                f'The communication should be an instance of callable function, but got {comm}.'
            )

        # checking synapse and output models
        if is_instance(syn, ParamDescriber[AlignPost]):
            if not is_instance(out, ParamDescriber[SynOut]):
                raise TypeError(
                    f'The output should be an instance of describer {ParamDescriber[SynOut]} when '
                    f'the synapse is a describer, but we got {out}.'
                )
            merging = True
        else:
            if not is_instance(out, SynOut):
                raise TypeError(
                    f'The output should be an instance of {SynOut} when the synapse is '
                    f'not a describer, but we got {out}.'
                )
            merging = False
        self.merging = merging

        # checking post model
        if not is_instance(post, Dynamics):
            raise TypeError(
                f'The post should be an instance of {Dynamics}, but got {post}.'
            )

        if merging:
            # synapse and output initialization
            syn, out = align_post_add_bef_update(syn_desc=syn, out_desc=out, post=post, proj_name=self.name)
        else:
            post.add_current_input(self.name, out)

        # references
        self.comm = comm
        self.syn: JointTypes[Dynamics, AlignPost] = syn
        self.out: BindCondData = out
        self.post: Dynamics = post

    @call_order(2)
    def init_state(self, *args, **kwargs):
        for module in self.modules:
            maybe_init_prefetch(module, *args, **kwargs)

    def update(self, *args):
        # call all modules
        for module in self.modules:
            x = call_module(module, *args)
            args = (x,)
        # communication module
        x = self.comm(*args)
        # add synapse input
        self.syn.add_delta_input(self.name, x)
        if not self.merging:
            # synapse and output interaction
            conductance = self.syn()
            self.out.bind_cond(conductance)


class DeltaProj(Interaction):
    r"""Full-chain of the synaptic projection for the Delta synapse model.

    The synaptic projection requires the input is the spiking data, otherwise
    the synapse is not the Delta synapse model.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``delay`` -> ``comm`` -> ``post``.

    **Model Descriptions**

    .. math::

        I_{syn} (t) = \sum_{j\in C} g_{\mathrm{max}} * \delta(t-t_j-D)

    where :math:`g_{\mathrm{max}}` denotes the chemical synaptic strength,
    :math:`t_j` the spiking moment of the presynaptic neuron :math:`j`,
    :math:`C` the set of neurons connected to the post-synaptic neuron,
    and :math:`D` the transmission delay of chemical synapses.
    For simplicity, the rise and decay phases of post-synaptic currents are
    omitted in this model.

    #     brainstate.nn.DeltaInteraction(
    #       LIF().prefetch('V'), bst.surrogate.ReluGrad(), comm, post
    #     )

    Args:
      pre: The pre-synaptic neuron group.
      delay: The synaptic delay.
      comm: DynamicalSystem. The synaptic communication.
      post: DynamicalSystem. The post-synaptic neuron group.
    """

    __module__ = 'brainstate.nn'

    def __init__(self, *modules, comm: Callable, post: Dynamics, label=None):
        super().__init__(name=get_unique_name(self.__class__.__name__))

        self.label = label

        # checking modules
        self.modules = _check_modules(*modules)

        # checking communication model
        if not callable(comm):
            raise TypeError(
                f'The communication should be an instance of callable function, but got {comm}.'
            )
        self.comm = comm

        # post model
        if not isinstance(post, Dynamics):
            raise TypeError(
                f'The post should be an instance of {Dynamics}, but got {post}.'
            )
        self.post = post

    @call_order(2)
    def init_state(self, *args, **kwargs):
        for module in self.modules:
            maybe_init_prefetch(module, *args, **kwargs)

    def update(self, *x):
        for module in self.modules:
            x = (call_module(module, *x),)
        assert len(x) == 1, f'The output of the modules should be a single value, but got {x}.'
        x = self.comm(x[0])
        self.post.add_delta_input(self.name, x, label=self.label)


class CurrentProj(Interaction):
    """
    Full-chain synaptic projection with the align-pre reduction and delay+synapse updating and merging.

    The ``full-chain`` means that the model needs to provide all information needed for a projection,
    including ``pre`` -> ``delay`` -> ``syn`` -> ``comm`` -> ``out`` -> ``post``.
    Note here, compared to ``FullProjAlignPreSD``, the ``delay`` and ``syn`` are exchanged.

    The ``align-pre`` means that the synaptic variables have the same dimension as the pre-synaptic neuron group.

    The ``delay+synapse updating`` means that the projection first delivers the pre neuron output (usually the
    spiking)  to the delay model, then computes the synapse states, and finally computes the synaptic current.

    The ``merging`` means that the same delay model is shared by all synapses, and the synapse model with same
    parameters (such like time constants) will also share the same synaptic variables.

    Neither ``FullProjAlignPreDS`` nor ``FullProjAlignPreSD`` facilitates the event-driven computation.
    This is because the ``comm`` is computed after the synapse state, which is a floating-point number, rather
    than the spiking. To facilitate the event-driven computation, please use align post projections.

    #    bint.CurrentInteraction(
    #       LIF().align_pre(bst.nn.Expon.desc()).prefetch('g'), comm, out, post
    #    )

    Args:
      prefetch: The synaptic dynamics.
      comm: The synaptic communication.
      out: The synaptic output.
      post: The post-synaptic neuron group.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        prefetch: Union[Prefetch, PrefetchDelayAt],
        comm: Callable,
        out: SynOut,
        post: Dynamics,
    ):
        super().__init__(name=get_unique_name(self.__class__.__name__))

        # pre-synaptic neuron group
        if not isinstance(prefetch, (Prefetch, PrefetchDelayAt)):
            raise TypeError(f'The pre should be a Prefetch or PrefetchDelayAt, but got {prefetch}.')
        self.prefetch = prefetch

        # check out
        if not isinstance(out, SynOut):
            raise TypeError(f'The out should be a SynOut, but got {out}.')
        self.out = out

        # check post
        if not isinstance(post, Dynamics):
            raise TypeError(f'The post should be a Dynamics, but got {post}.')
        self.post = post
        post.add_current_input(self.name, out)

        # output initialization
        self.comm = comm

    @call_order(2)
    def init_state(self, *args, **kwargs):
        maybe_init_prefetch(self.prefetch, *args, **kwargs)

    def update(self, *x):
        x = self.prefetch(*x)
        x = self.comm(x)
        self.out.bind_cond(x)
