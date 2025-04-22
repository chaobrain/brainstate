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

    $$
    dg/dt = -g/\tau + \text{input}
    $$

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

    where $a$ is a normalization factor that ensures the peak conductance reaches
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
    time constant $\tau$, with peak amplitude occurring at time $t = \tau$.

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
    r"""
    Synapse with short-term plasticity.

    This class implements a synapse model with short-term plasticity (STP), which captures
    activity-dependent changes in synaptic efficacy that occur over milliseconds to seconds.
    The model simultaneously accounts for both short-term facilitation and depression
    based on the formulation by Tsodyks & Markram (1998).

    The model is characterized by the following equations:

    $$
    \frac{du}{dt} = -\frac{u}{\tau_f} + U \cdot (1 - u) \cdot \delta(t - t_{spike})
    $$

    $$
    \frac{dx}{dt} = \frac{1 - x}{\tau_d} - u \cdot x \cdot \delta(t - t_{spike})
    $$

    $$
    g_{syn} = u \cdot x
    $$

    where:
    - $u$ represents the utilization of synaptic efficacy (facilitation variable)
    - $x$ represents the available synaptic resources (depression variable)
    - $\tau_f$ is the facilitation time constant
    - $\tau_d$ is the depression time constant
    - $U$ is the baseline utilization parameter
    - $\delta(t - t_{spike})$ is the Dirac delta function representing presynaptic spikes
    - $g_{syn}$ is the effective synaptic conductance

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    U : ArrayLike, default=0.15
        Baseline utilization parameter (fraction of resources used per action potential).
    tau_f : ArrayLike, default=1500.*u.ms
        Time constant of short-term facilitation in milliseconds.
    tau_d : ArrayLike, default=200.*u.ms
        Time constant of short-term depression (recovery of synaptic resources) in milliseconds.

    Attributes
    ----------
    u : HiddenState
        Utilization of synaptic efficacy (facilitation variable).
    x : HiddenState
        Available synaptic resources (depression variable).

    Notes
    -----
    - Larger values of tau_f produce stronger facilitation effects.
    - Larger values of tau_d lead to slower recovery from depression.
    - The parameter U controls the initial release probability.
    - The effective synaptic strength is the product of u and x.

    References
    ----------
    .. [1] Tsodyks, M. V., & Markram, H. (1997). The neural code between neocortical
           pyramidal neurons depends on neurotransmitter release probability.
           Proceedings of the National Academy of Sciences, 94(2), 719-723.
    .. [2] Tsodyks, M., Pawelzik, K., & Markram, H. (1998). Neural networks with dynamic
           synapses. Neural computation, 10(4), 821-835.
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
    r"""
    Synapse with short-term depression.

    This class implements a synapse model with short-term depression (STD), which captures
    activity-dependent reduction in synaptic efficacy, typically caused by depletion of
    neurotransmitter vesicles following repeated stimulation.

    The model is characterized by the following equation:

    $$
    \frac{dx}{dt} = \frac{1 - x}{\tau} - U \cdot x \cdot \delta(t - t_{spike})
    $$

    $$
    g_{syn} = x
    $$

    where:
    - $x$ represents the available synaptic resources (depression variable)
    - $\tau$ is the depression recovery time constant
    - $U$ is the utilization parameter (fraction of resources depleted per spike)
    - $\delta(t - t_{spike})$ is the Dirac delta function representing presynaptic spikes
    - $g_{syn}$ is the effective synaptic conductance

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    tau : ArrayLike, default=200.*u.ms
        Time constant governing recovery of synaptic resources in milliseconds.
    U : ArrayLike, default=0.07
        Utilization parameter (fraction of resources used per action potential).

    Attributes
    ----------
    x : HiddenState
        Available synaptic resources (depression variable).

    Notes
    -----
    - Larger values of tau lead to slower recovery from depression.
    - Larger values of U cause stronger depression with each spike.
    - This model is a simplified version of the STP model that only includes depression.

    References
    ----------
    .. [1] Abbott, L. F., Varela, J. A., Sen, K., & Nelson, S. B. (1997). Synaptic
           depression and cortical gain control. Science, 275(5297), 220-224.
    .. [2] Tsodyks, M. V., & Markram, H. (1997). The neural code between neocortical
           pyramidal neurons depends on neurotransmitter release probability.
           Proceedings of the National Academy of Sciences, 94(2), 719-723.
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
    r"""AMPA receptor synapse model.

    This class implements a kinetic model of AMPA (α-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid)
    receptor-mediated synaptic transmission. AMPA receptors are ionotropic glutamate receptors that mediate
    fast excitatory synaptic transmission in the central nervous system.

    The model uses a Markov process approach to describe the state transitions of AMPA receptors
    between closed and open states, governed by neurotransmitter binding:

    $$
    \frac{dg}{dt} = \alpha [T] (1-g) - \beta g
    $$

    $$
    I_{syn} = g_{max} \cdot g \cdot (V - E)
    $$

    where:
    - $g$ represents the fraction of receptors in the open state
    - $\alpha$ is the binding rate constant [ms^-1 mM^-1]
    - $\beta$ is the unbinding rate constant [ms^-1]
    - $[T]$ is the neurotransmitter concentration [mM]
    - $I_{syn}$ is the resulting synaptic current
    - $g_{max}$ is the maximum conductance
    - $V$ is the membrane potential
    - $E$ is the reversal potential

    The neurotransmitter concentration $[T]$ follows a square pulse of amplitude T and
    duration T_dur after each presynaptic spike.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    alpha : ArrayLike, default=0.98/(u.ms*u.mM)
        Binding rate constant [ms^-1 mM^-1].
    beta : ArrayLike, default=0.18/u.ms
        Unbinding rate constant [ms^-1].
    T : ArrayLike, default=0.5*u.mM
        Peak neurotransmitter concentration when released [mM].
    T_dur : ArrayLike, default=0.5*u.ms
        Duration of neurotransmitter presence in the synaptic cleft [ms].
    g_initializer : ArrayLike or Callable, default=init.ZeroInit()
        Initial value or initializer for the synaptic conductance.

    Attributes
    ----------
    g : HiddenState
        Fraction of receptors in the open state.
    spike_arrival_time : ShortTermState
        Time of the most recent presynaptic spike.

    Notes
    -----
    - The model captures the fast-rising and relatively fast-decaying excitatory currents
      characteristic of AMPA receptor-mediated transmission.
    - The time course of the synaptic conductance is determined by both the binding and
      unbinding rate constants and the duration of transmitter presence.
    - This implementation uses an exponential Euler integration method.

    References
    ----------
    .. [1] Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1994). Synthesis of models for
           excitable membranes, synaptic transmission and neuromodulation using a common
           kinetic formalism. Journal of computational neuroscience, 1(3), 195-230.
    .. [2] Vijayan, S., & Kopell, N. J. (2012). Thalamic model of awake alpha oscillations
           and implications for stimulus processing. Proceedings of the National Academy
           of Sciences, 109(45), 18553-18558.
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
    r"""GABAa receptor synapse model.

    This class implements a kinetic model of GABAa (gamma-aminobutyric acid type A)
    receptor-mediated synaptic transmission. GABAa receptors are ionotropic chloride channels
    that mediate fast inhibitory synaptic transmission in the central nervous system.

    The model uses the same Markov process approach as the AMPA model but with different
    kinetic parameters appropriate for GABAa receptors:

    $$
    \frac{dg}{dt} = \alpha [T] (1-g) - \beta g
    $$

    $$
    I_{syn} = - g_{max} \cdot g \cdot (V - E)
    $$

    where:
    - $g$ represents the fraction of receptors in the open state
    - $\alpha$ is the binding rate constant [ms^-1 mM^-1], typically slower than AMPA
    - $\beta$ is the unbinding rate constant [ms^-1]
    - $[T]$ is the neurotransmitter (GABA) concentration [mM]
    - $I_{syn}$ is the resulting synaptic current (note the negative sign indicating inhibition)
    - $g_{max}$ is the maximum conductance
    - $V$ is the membrane potential
    - $E$ is the reversal potential (typically around -80 mV for chloride)

    The neurotransmitter concentration $[T]$ follows a square pulse of amplitude T and
    duration T_dur after each presynaptic spike.

    Parameters
    ----------
    in_size : Size
        Size of the input.
    name : str, optional
        Name of the synapse instance.
    alpha : ArrayLike, default=0.53/(u.ms*u.mM)
        Binding rate constant [ms^-1 mM^-1]. Typically slower than AMPA receptors.
    beta : ArrayLike, default=0.18/u.ms
        Unbinding rate constant [ms^-1].
    T : ArrayLike, default=1.0*u.mM
        Peak neurotransmitter concentration when released [mM]. Higher than AMPA.
    T_dur : ArrayLike, default=1.0*u.ms
        Duration of neurotransmitter presence in the synaptic cleft [ms]. Longer than AMPA.
    g_initializer : ArrayLike or Callable, default=init.ZeroInit()
        Initial value or initializer for the synaptic conductance.

    Attributes
    ----------
    Inherits all attributes from AMPA class.

    Notes
    -----
    - GABAa receptors typically produce slower-rising and longer-lasting currents compared to AMPA receptors.
    - The inhibitory nature of GABAa receptors is reflected in the convention of using a negative sign in the
      synaptic current equation.
    - The reversal potential for GABAa receptors is typically around -80 mV (due to chloride), making them
      inhibitory for neurons with resting potentials more positive than this value.
    - This model does not include desensitization, which can be significant for prolonged GABA exposure.

    References
    ----------
    .. [1] Destexhe, A., Mainen, Z. F., & Sejnowski, T. J. (1994). Synthesis of models for
           excitable membranes, synaptic transmission and neuromodulation using a common
           kinetic formalism. Journal of computational neuroscience, 1(3), 195-230.
    .. [2] Destexhe, A., & Paré, D. (1999). Impact of network activity on the integrative
           properties of neocortical pyramidal neurons in vivo. Journal of neurophysiology,
           81(4), 1531-1547.
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
