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

# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Callable

import brainunit as u

from brainstate import init, environ
from brainstate._state import ShortTermState, HiddenState
from brainstate.mixin import AlignPost
from brainstate.nn._dynamics._dynamics_base import Dynamics
from brainstate.nn._exp_euler import exp_euler_step
from brainstate.typing import ArrayLike, Size

__all__ = [
    'Synapse', 'Expon', 'DualExpon', 'Alpha', 'STP', 'STD', 'AMPA', 'GABAa',
]


class Synapse(Dynamics):
    """
    Base class for synapse dynamics.

    This class serves as the foundation for all synapse models in the library,
    providing a common interface for implementing various types of synaptic
    connectivity and transmission mechanisms.

    Synapses are responsible for modeling the transmission of signals between
    neurons, including temporal dynamics, plasticity, and neurotransmitter effects.
    All specific synapse implementations (like Expon, Alpha, AMPA, etc.) should
    inherit from this class.

    See Also
    --------
    Expon : Simple first-order exponential decay synapse model
    Alpha : Alpha function synapse model
    DualExpon : Dual exponential synapse model
    STP : Synapse with short-term plasticity
    STD : Synapse with short-term depression
    AMPA : AMPA receptor synapse model
    GABAa : GABAa receptor synapse model
    """
    __module__ = 'brainstate.nn'


class Expon(Synapse, AlignPost):
    r"""
    Exponential decay synapse model.

    This class implements a simple first-order exponential decay synapse model where
    the synaptic conductance g decays exponentially with time constant tau:

    dg/dt = -g/tau + input

    The model is widely used for basic synaptic transmission modeling.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    tau : ArrayLike, default=8.0*u.ms
        Time constant of decay in milliseconds.
    g_initializer : ArrayLike or Callable, default=init.ZeroInit(unit=u.mS)
        Initial value or initializer for synaptic conductance.

    Attributes
    ----------
    g : HiddenState
        Synaptic conductance state variable.
    tau : Parameter
        Time constant of decay.

    Notes
    -----
    The implementation uses an exponential Euler integration method.
    The output of this synapse is the conductance value.

    This class inherits from :py:class:`AlignPost`, which means it can be used in projection patterns
    where synaptic variables are aligned with post-synaptic neurons, enabling event-driven
    computation and more efficient handling of sparse connectivity patterns.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        tau: ArrayLike = 8.0 * u.ms,
        g_initializer: ArrayLike | Callable = init.ZeroInit(unit=u.mS),
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau = init.param(tau, self.varshape)
        self.g_initializer = g_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.g = HiddenState(init.param(self.g_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.g.value = init.param(self.g_initializer, self.varshape, batch_size)

    def update(self, x=None):
        g = exp_euler_step(lambda g: self.sum_current_inputs(-g) / self.tau, self.g.value)
        self.g.value = self.sum_delta_inputs(g)
        if x is not None: self.g.value += x
        return self.g.value


class DualExpon(Synapse, AlignPost):
    r"""
    Dual exponential synapse model.

    This class implements a synapse model with separate rise and decay time constants,
    which produces a more biologically realistic conductance waveform than a single
    exponential model. The model is characterized by the differential equation system:

    dg_rise/dt = -g_rise/tau_rise
    dg_decay/dt = -g_decay/tau_decay
    g = a * (g_decay - g_rise)

    where a is a normalization factor that ensures the peak conductance reaches
    the desired amplitude.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    tau_decay : ArrayLike, default=10.0*u.ms
        Time constant of decay in milliseconds.
    tau_rise : ArrayLike, default=1.0*u.ms
        Time constant of rise in milliseconds.
    A : ArrayLike, optional
        Amplitude scaling factor. If None, a scaling factor is automatically
        calculated to normalize the peak amplitude.
    g_initializer : ArrayLike or Callable, default=init.ZeroInit(unit=u.mS)
        Initial value or initializer for synaptic conductance.

    Attributes
    ----------
    g_rise : HiddenState
        Rise component of synaptic conductance.
    g_decay : HiddenState
        Decay component of synaptic conductance.
    tau_rise : Parameter
        Time constant of rise phase.
    tau_decay : Parameter
        Time constant of decay phase.
    a : Parameter
        Normalization factor calculated from tau_rise, tau_decay, and A.

    Notes
    -----
    The dual exponential model produces a conductance waveform that is more
    physiologically realistic than a simple exponential decay, with a finite
    rise time followed by a slower decay.

    The implementation uses an exponential Euler integration method.
    The output of this synapse is the normalized difference between decay and rise components.

    This class inherits from :py:class:`AlignPost`, which means it can be used in projection patterns
    where synaptic variables are aligned with post-synaptic neurons, enabling event-driven
    computation and more efficient handling of sparse connectivity patterns.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        tau_decay: ArrayLike = 10.0 * u.ms,
        tau_rise: ArrayLike = 1.0 * u.ms,
        A: Optional[ArrayLike] = None,
        g_initializer: ArrayLike | Callable = init.ZeroInit(unit=u.mS),
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau_decay = init.param(tau_decay, self.varshape)
        self.tau_rise = init.param(tau_rise, self.varshape)
        A = self._format_dual_exp_A(A)
        self.a = (self.tau_decay - self.tau_rise) / self.tau_rise / self.tau_decay * A
        self.g_initializer = g_initializer

    def _format_dual_exp_A(self, A):
        A = init.param(A, sizes=self.varshape, allow_none=True)
        if A is None:
            A = (
                self.tau_decay / (self.tau_decay - self.tau_rise) *
                u.math.float_power(self.tau_rise / self.tau_decay,
                                   self.tau_rise / (self.tau_rise - self.tau_decay))
            )
        return A

    def init_state(self, batch_size: int = None, **kwargs):
        self.g_rise = HiddenState(init.param(self.g_initializer, self.varshape, batch_size))
        self.g_decay = HiddenState(init.param(self.g_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.g_rise.value = init.param(self.g_initializer, self.varshape, batch_size)
        self.g_decay.value = init.param(self.g_initializer, self.varshape, batch_size)

    def update(self, x=None):
        g_rise = exp_euler_step(lambda h: -h / self.tau_rise, self.g_rise.value)
        g_decay = exp_euler_step(lambda g: -g / self.tau_decay, self.g_decay.value)
        self.g_rise.value = self.sum_delta_inputs(g_rise)
        self.g_decay.value = self.sum_delta_inputs(g_decay)
        if x is not None:
            self.g_rise.value += x
            self.g_decay.value += x
        return self.a * (self.g_decay.value - self.g_rise.value)


class Alpha(Synapse):
    r"""
    Alpha synapse model.

    This class implements the alpha function synapse model, which produces
    a smooth, biologically realistic synaptic conductance waveform.
    The model is characterized by the differential equation system:

    dh/dt = -h/tau
    dg/dt = -g/tau + h/tau

    This produces a response that rises and then falls with a characteristic
    time constant tau, with peak amplitude occurring at time t = tau.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    tau : ArrayLike, default=8.0*u.ms
        Time constant of the alpha function in milliseconds.
    g_initializer : ArrayLike or Callable, default=init.ZeroInit(unit=u.mS)
        Initial value or initializer for synaptic conductance.

    Attributes
    ----------
    g : HiddenState
        Synaptic conductance state variable.
    h : HiddenState
        Auxiliary state variable for implementing the alpha function.
    tau : Parameter
        Time constant of the alpha function.

    Notes
    -----
    The alpha function is defined as g(t) = (t/tau) * exp(1-t/tau) for t ≥ 0.
    This implementation uses an exponential Euler integration method.
    The output of this synapse is the conductance value.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        tau: ArrayLike = 8.0 * u.ms,
        g_initializer: ArrayLike | Callable = init.ZeroInit(unit=u.mS),
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau = init.param(tau, self.varshape)
        self.g_initializer = g_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.g = HiddenState(init.param(self.g_initializer, self.varshape, batch_size))
        self.h = HiddenState(init.param(self.g_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.g.value = init.param(self.g_initializer, self.varshape, batch_size)
        self.h.value = init.param(self.g_initializer, self.varshape, batch_size)

    def update(self, x=None):
        h = exp_euler_step(lambda h: -h / self.tau, self.h.value)
        self.g.value = exp_euler_step(lambda g, h: -g / self.tau + h / self.tau, self.g.value, self.h.value)
        self.h.value = self.sum_delta_inputs(h)
        if x is not None:
            self.h.value += x
        return self.g.value


class STP(Synapse):
    r"""Synaptic output with short-term plasticity.

    %s

    Args:
      tau_f: float, ArrayType, Callable. The time constant of short-term facilitation.
      tau_d: float, ArrayType, Callable. The time constant of short-term depression.
      U: float, ArrayType, Callable. The fraction of resources used per action potential.
      %s
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        U: ArrayLike = 0.15,
        tau_f: ArrayLike = 1500. * u.ms,
        tau_d: ArrayLike = 200. * u.ms,
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau_f = init.param(tau_f, self.varshape)
        self.tau_d = init.param(tau_d, self.varshape)
        self.U = init.param(U, self.varshape)

    def init_state(self, batch_size: int = None, **kwargs):
        self.x = HiddenState(init.param(init.Constant(1.), self.varshape, batch_size))
        self.u = HiddenState(init.param(init.Constant(self.U), self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.x.value = init.param(init.Constant(1.), self.varshape, batch_size)
        self.u.value = init.param(init.Constant(self.U), self.varshape, batch_size)

    def update(self, pre_spike):
        u = exp_euler_step(lambda u: - u / self.tau_f, self.u.value)
        x = exp_euler_step(lambda x: (1 - x) / self.tau_d, self.x.value)

        # --- original code:
        #   if pre_spike.dtype == jax.numpy.bool_:
        #     u = bm.where(pre_spike, u + self.U * (1 - self.u), u)
        #     x = bm.where(pre_spike, x - u * self.x, x)
        #   else:
        #     u = pre_spike * (u + self.U * (1 - self.u)) + (1 - pre_spike) * u
        #     x = pre_spike * (x - u * self.x) + (1 - pre_spike) * x

        # --- simplified code:
        u = u + pre_spike * self.U * (1 - self.u.value)
        x = x - pre_spike * u * self.x.value

        self.u.value = u
        self.x.value = x
        return u * x * pre_spike


class STD(Synapse):
    r"""Synaptic output with short-term depression.

    %s

    Args:
      tau: float, ArrayType, Callable. The time constant of recovery of the synaptic vesicles.
      U: float, ArrayType, Callable. The fraction of resources used per action potential.
      %s
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        # synapse parameters
        tau: ArrayLike = 200. * u.ms,
        U: ArrayLike = 0.07,
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.tau = init.param(tau, self.varshape)
        self.U = init.param(U, self.varshape)

    def init_state(self, batch_size: int = None, **kwargs):
        self.x = HiddenState(init.param(init.Constant(1.), self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.x.value = init.param(init.Constant(1.), self.varshape, batch_size)

    def update(self, pre_spike):
        x = exp_euler_step(lambda x: (1 - x) / self.tau, self.x.value)

        # --- original code:
        # self.x.value = bm.where(pre_spike, x - self.U * self.x, x)

        # --- simplified code:
        self.x.value = x - pre_spike * self.U * self.x.value

        return self.x.value * pre_spike


class AMPA(Synapse):
    r"""AMPA synapse model.

    **Model Descriptions**

    AMPA receptor is an ionotropic receptor, which is an ion channel.
    When it is bound by neurotransmitters, it will immediately open the
    ion channel, causing the change of membrane potential of postsynaptic neurons.

    A classical model is to use the Markov process to model ion channel switch.
    Here :math:`g` represents the probability of channel opening, :math:`1-g`
    represents the probability of ion channel closing, and :math:`\alpha` and
    :math:`\beta` are the transition probability. Because neurotransmitters can
    open ion channels, the transfer probability from :math:`1-g` to :math:`g`
    is affected by the concentration of neurotransmitters. We denote the concentration
    of neurotransmitters as :math:`[T]` and get the following Markov process.

    .. image:: ../../_static/synapse_markov.png
        :align: center

    We obtained the following formula when describing the process by a differential equation.

    .. math::

        \frac{ds}{dt} =\alpha[T](1-g)-\beta g

    where :math:`\alpha [T]` denotes the transition probability from state :math:`(1-g)`
    to state :math:`(g)`; and :math:`\beta` represents the transition probability of
    the other direction. :math:`\alpha` is the binding constant. :math:`\beta` is the
    unbinding constant. :math:`[T]` is the neurotransmitter concentration, and
    has the duration of 0.5 ms.

    Moreover, the post-synaptic current on the post-synaptic neuron is formulated as

    .. math::

        I_{syn} = g_{max} g (V-E)

    where :math:`g_{max}` is the maximum conductance, and `E` is the reverse potential.

    This module can be used with interface ``brainpy.dyn.ProjAlignPreMg2``, as shown in the following example:

    .. [1] Vijayan S, Kopell N J. Thalamic model of awake alpha oscillations
           and implications for stimulus processing[J]. Proceedings of the
           National Academy of Sciences, 2012, 109(45): 18553-18558.

    Args:
      alpha: float, ArrayType, Callable. Binding constant. [ms^-1 mM^-1]
      beta: float, ArrayType, Callable. Unbinding constant. [ms^-1 mM^-1]
      T: float, ArrayType, Callable. Transmitter concentration when synapse is triggered by
        a pre-synaptic spike. Default 0.5 [mM].
      T_dur: float, ArrayType, Callable. Transmitter concentration duration time after being triggered. Default 1 [ms]
    """

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        alpha: ArrayLike = 0.98 / (u.ms * u.mM),
        beta: ArrayLike = 0.18 / u.ms,
        T: ArrayLike = 0.5 * u.mM,
        T_dur: ArrayLike = 0.5 * u.ms,
        g_initializer: ArrayLike | Callable = init.ZeroInit(),
    ):
        super().__init__(name=name, in_size=in_size)

        # parameters
        self.alpha = init.param(alpha, self.varshape)
        self.beta = init.param(beta, self.varshape)
        self.T = init.param(T, self.varshape)
        self.T_duration = init.param(T_dur, self.varshape)
        self.g_initializer = g_initializer

    def init_state(self, batch_size=None):
        self.g = HiddenState(init.param(self.g_initializer, self.varshape, batch_size))
        self.spike_arrival_time = ShortTermState(init.param(init.Constant(-1e7 * u.ms), self.varshape, batch_size))

    def reset_state(self, batch_or_mode=None, **kwargs):
        self.g.value = init.param(self.g_initializer, self.varshape, batch_or_mode)
        self.spike_arrival_time.value = init.param(init.Constant(-1e7 * u.ms), self.varshape, batch_or_mode)

    def dg(self, g, t, TT):
        return self.alpha * TT * (1 - g) - self.beta * g

    def update(self, pre_spike):
        t = environ.get('t')
        self.spike_arrival_time.value = u.math.where(pre_spike, t, self.spike_arrival_time.value)
        TT = ((t - self.spike_arrival_time.value) < self.T_duration) * self.T
        self.g.value = exp_euler_step(self.dg, self.g.value, t, TT)
        return self.g.value


class GABAa(AMPA):
    r"""GABAa synapse model.

    **Model Descriptions**

    GABAa synapse model has the same equation with the `AMPA synapse <./brainmodels.synapses.AMPA.rst>`_,

    .. math::

        \frac{d g}{d t}&=\alpha[T](1-g) - \beta g \\
        I_{syn}&= - g_{max} g (V - E)

    but with the difference of:

    - Reversal potential of synapse :math:`E` is usually low, typically -80. mV
    - Activating rate constant :math:`\alpha=0.53`
    - De-activating rate constant :math:`\beta=0.18`
    - Transmitter concentration :math:`[T]=1\,\mu ho(\mu S)` when synapse is
      triggered by a pre-synaptic spike, with the duration of 1. ms.

    This module can be used with interface ``brainpy.dyn.ProjAlignPreMg2``, as shown in the following example:

    .. [1] Destexhe, Alain, and Denis Paré. "Impact of network activity
           on the integrative properties of neocortical pyramidal neurons
           in vivo." Journal of neurophysiology 81.4 (1999): 1531-1547.

    Args:
      alpha: float, ArrayType, Callable. Binding constant. Default 0.062
      beta: float, ArrayType, Callable. Unbinding constant. Default 3.57
      T: float, ArrayType, Callable. Transmitter concentration when synapse is triggered by
        a pre-synaptic spike.. Default 1 [mM].
      T_dur: float, ArrayType, Callable. Transmitter concentration duration time
        after being triggered. Default 1 [ms]
    """

    def __init__(
        self,
        in_size: Size,
        name: Optional[str] = None,
        alpha: ArrayLike = 0.53 / (u.ms * u.mM),
        beta: ArrayLike = 0.18 / u.ms,
        T: ArrayLike = 1.0 * u.mM,
        T_dur: ArrayLike = 1.0 * u.ms,
        g_initializer: ArrayLike | Callable = init.ZeroInit(),
    ):
        super().__init__(
            alpha=alpha,
            beta=beta,
            T=T,
            T_dur=T_dur,
            name=name,
            in_size=in_size,
            g_initializer=g_initializer
        )
