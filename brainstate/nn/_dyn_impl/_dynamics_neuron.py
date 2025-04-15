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

import brainunit as u
import jax
from typing import Callable, Optional

from brainstate import init, surrogate, environ
from brainstate._state import HiddenState, ShortTermState
from brainstate.nn._dynamics._dynamics_base import Dynamics
from brainstate.nn._exp_euler import exp_euler_step
from brainstate.typing import ArrayLike, Size

__all__ = [
    'Neuron', 'IF', 'LIF', 'LIFRef', 'ALIF',
]


class Neuron(Dynamics):
    """
    Base class for all spiking neuron models.

    This abstract class serves as the foundation for implementing various spiking neuron
    models. It extends the Dynamics class and provides common functionality for spike
    generation and membrane potential dynamics.

    All neuron models should inherit from this class and implement the required methods,
    particularly the `get_spike()` method which defines the spike generation mechanism.

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    spk_fun : Callable, default=surrogate.InvSquareGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation:
        - 'soft': subtract threshold from membrane potential
        - 'hard': use stop_gradient for reset
    name : str, optional
        Name of the neuron layer.

    Methods
    -------
    get_spike(*args, **kwargs)
        Abstract method that generates spikes based on neuron state variables.
        Must be implemented by subclasses.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        spk_fun: Callable = surrogate.InvSquareGrad(),
        spk_reset: str = 'soft',
        name: Optional[str] = None,
    ):
        super().__init__(in_size, name=name)
        self.spk_reset = spk_reset
        self.spk_fun = spk_fun

    def get_spike(self, *args, **kwargs):
        raise NotImplementedError


class IF(Neuron):
    r"""
    Integrate-and-Fire (IF) neuron model.

    The IF neuron is one of the simplest spiking neuron models that captures
    the essential dynamics of biological neurons. It accumulates input current
    until a threshold is reached, at which point it fires a spike and resets.

    Mathematical model:

        $$
        \tau dV/dt = -V + R I(t)
        $$

    Spike condition:
        If $V ≥ V_{th}$: emit spike and reset $V = V - V_{th}$

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1.0 * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=5.0 * u.ms
        Membrane time constant.
    V_th : ArrayLike, default=1.0 * u.mV
        Firing threshold voltage (should be positive).
    V_initializer : Callable, default=init.Constant(0. * u.mV)
        Initializer for the membrane potential state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation:
        - 'soft': subtract threshold V = V - V_th
        - 'hard': strict reset using stop_gradient
    name : str, optional
        Name of the neuron layer.

    State Variables
    --------------
    V : HiddenState
        Membrane potential.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the neuron state variables.
    reset_state(batch_size=None, **kwargs)
        Reset the neuron state variables.
    get_spike(V=None)
        Generate spikes based on the membrane potential.
    update(x=0. * u.mA)
        Update the neuron state for one time step and return spikes.
    """

    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 5. * u.ms,
        V_th: ArrayLike = 1. * u.mV,  # should be positive
        V_initializer: Callable = init.Constant(0. * u.mV),
        spk_fun: Callable = surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = init.param(R, self.varshape)
        self.tau = init.param(tau, self.varshape)
        self.V_th = init.param(V_th, self.varshape)
        self.V_initializer = V_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = HiddenState(init.param(self.V_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = init.param(self.V_initializer, self.varshape, batch_size)

    def get_spike(self, V=None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / self.V_th
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        # reset
        last_V = self.V.value
        last_spike = self.get_spike(self.V.value)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_V)
        V = last_V - V_th * last_spike
        # membrane potential
        dv = lambda v: (-v + self.R * self.sum_current_inputs(x, v)) / self.tau
        V = exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        return self.get_spike(V)


class LIF(Neuron):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    The LIF neuron extends the IF model by adding a leak term, making the membrane
    potential decay towards a resting value when no input is present. This makes the
    model more biologically realistic.

    Mathematical model:
        τ·dV/dt = -(V - V_rest) + R·I(t)

    Spike condition:
        If V ≥ V_th: emit spike and reset V = V_reset

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1.0 * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=5.0 * u.ms
        Membrane time constant.
    V_th : ArrayLike, default=1.0 * u.mV
        Firing threshold voltage.
    V_reset : ArrayLike, default=0.0 * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=0.0 * u.mV
        Resting membrane potential.
    V_initializer : Callable, default=init.Constant(0. * u.mV)
        Initializer for the membrane potential state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation:
        - 'soft': subtract threshold V = V - V_th
        - 'hard': strict reset using stop_gradient
    name : str, optional
        Name of the neuron layer.

    State Variables
    --------------
    V : HiddenState
        Membrane potential.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the neuron state variables.
    reset_state(batch_size=None, **kwargs)
        Reset the neuron state variables.
    get_spike(V=None)
        Generate spikes based on the membrane potential.
    update(x=0. * u.mA)
        Update the neuron state for one time step and return spikes.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 5. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        V_reset: ArrayLike = 0. * u.mV,
        V_rest: ArrayLike = 0. * u.mV,
        V_initializer: Callable = init.Constant(0. * u.mV),
        spk_fun: Callable = surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = init.param(R, self.varshape)
        self.tau = init.param(tau, self.varshape)
        self.V_th = init.param(V_th, self.varshape)
        self.V_rest = init.param(V_rest, self.varshape)
        self.V_reset = init.param(V_reset, self.varshape)
        self.V_initializer = V_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = HiddenState(init.param(self.V_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = init.param(self.V_initializer, self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        last_v = self.V.value
        lst_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * lst_spk
        # membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
        V = exp_euler_step(dv, V)
        V = self.sum_delta_inputs(V)
        self.V.value = V
        return self.get_spike(V)


class LIFRef(Neuron):
    """
    Leaky Integrate-and-Fire with Refractory Period (LIFRef) neuron model.

    This model extends the LIF neuron by adding a refractory period during which
    the neuron cannot fire again after a spike, regardless of input strength.
    This better mimics the behavior of biological neurons.

    Mathematical model:
        τ·dV/dt = -(V - V_rest) + R·I(t)  (when not in refractory period)
        V = V_reset                        (during refractory period)

    Spike condition:
        If V ≥ V_th: emit spike, reset V = V_reset, and enter refractory period
        for duration τ_ref

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1.0 * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=5.0 * u.ms
        Membrane time constant.
    tau_ref : ArrayLike, default=5.0 * u.ms
        Refractory period duration.
    V_th : ArrayLike, default=1.0 * u.mV
        Firing threshold voltage.
    V_reset : ArrayLike, default=0.0 * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=0.0 * u.mV
        Resting membrane potential.
    V_initializer : Callable, default=init.Constant(0. * u.mV)
        Initializer for the membrane potential state.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation:
        - 'soft': subtract threshold V = V - V_th
        - 'hard': strict reset using stop_gradient
    name : str, optional
        Name of the neuron layer.

    State Variables
    --------------
    V : HiddenState
        Membrane potential.
    last_spike_time : ShortTermState
        Time of the last spike, used to implement refractory period.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the neuron state variables.
    reset_state(batch_size=None, **kwargs)
        Reset the neuron state variables.
    get_spike(V=None)
        Generate spikes based on the membrane potential.
    update(x=0. * u.mA)
        Update the neuron state for one time step and return spikes.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 5. * u.ms,
        tau_ref: ArrayLike = 5. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        V_reset: ArrayLike = 0. * u.mV,
        V_rest: ArrayLike = 0. * u.mV,
        V_initializer: Callable = init.Constant(0. * u.mV),
        spk_fun: Callable = surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = init.param(R, self.varshape)
        self.tau = init.param(tau, self.varshape)
        self.tau_ref = init.param(tau_ref, self.varshape)
        self.V_th = init.param(V_th, self.varshape)
        self.V_rest = init.param(V_rest, self.varshape)
        self.V_reset = init.param(V_reset, self.varshape)
        self.V_initializer = V_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = HiddenState(init.param(self.V_initializer, self.varshape, batch_size))
        self.last_spike_time = ShortTermState(init.param(init.Constant(-1e7 * u.ms), self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = init.param(self.V_initializer, self.varshape, batch_size)
        self.last_spike_time.value = init.param(init.Constant(-1e7 * u.ms), self.varshape, batch_size)

    def get_spike(self, V: ArrayLike = None):
        V = self.V.value if V is None else V
        v_scaled = (V - self.V_th) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        t = environ.get('t')
        last_v = self.V.value
        lst_spk = self.get_spike(last_v)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        last_v = last_v - (V_th - self.V_reset) * lst_spk
        # membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
        V = exp_euler_step(dv, last_v)
        V = self.sum_delta_inputs(V)
        self.V.value = u.math.where(t - self.last_spike_time.value < self.tau_ref, last_v, V)
        # spike time evaluation
        lst_spk_time = u.math.where(self.V.value >= self.V_th, environ.get('t'), self.last_spike_time.value)
        self.last_spike_time.value = jax.lax.stop_gradient(lst_spk_time)
        return self.get_spike()


class ALIF(Neuron):
    """
    Adaptive Leaky Integrate-and-Fire (ALIF) neuron model.

    The ALIF model extends the LIF neuron by adding an adaptation variable that
    increases the effective firing threshold after each spike, allowing the neuron
    to adapt its firing rate. This creates spike-frequency adaptation, a common
    feature in biological neurons.

    Mathematical model:
        τ·dV/dt = -(V - V_rest) + R·I(t)
        τ_a·da/dt = -a

    Spike condition:
        If V ≥ V_th + β·a: emit spike, reset V = V_reset, and increment a = a + 1

    Parameters
    ----------
    in_size : Size
        Size of the input to the neuron.
    R : ArrayLike, default=1.0 * u.ohm
        Membrane resistance.
    tau : ArrayLike, default=5.0 * u.ms
        Membrane time constant.
    tau_a : ArrayLike, default=100.0 * u.ms
        Adaptation time constant (typically longer than tau).
    V_th : ArrayLike, default=1.0 * u.mV
        Base firing threshold voltage.
    V_reset : ArrayLike, default=0.0 * u.mV
        Reset voltage after spike.
    V_rest : ArrayLike, default=0.0 * u.mV
        Resting membrane potential.
    beta : ArrayLike, default=0.1 * u.mV
        Adaptation coupling parameter that scales the effect of the adaptation variable.
    spk_fun : Callable, default=surrogate.ReluGrad()
        Surrogate gradient function for the non-differentiable spike generation.
    spk_reset : str, default='soft'
        Reset mechanism after spike generation:
        - 'soft': subtract threshold V = V - V_th
        - 'hard': strict reset using stop_gradient
    V_initializer : Callable, default=init.Constant(0. * u.mV)
        Initializer for the membrane potential state.
    a_initializer : Callable, default=init.Constant(0.)
        Initializer for the adaptation variable.
    name : str, optional
        Name of the neuron layer.

    State Variables
    --------------
    V : HiddenState
        Membrane potential.
    a : HiddenState
        Adaptation variable that increases after each spike and decays exponentially.

    Methods
    -------
    init_state(batch_size=None, **kwargs)
        Initialize the neuron state variables.
    reset_state(batch_size=None, **kwargs)
        Reset the neuron state variables.
    get_spike(V=None, a=None)
        Generate spikes based on the membrane potential and adaptation variable.
    update(x=0. * u.mA)
        Update the neuron state for one time step and return spikes.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        R: ArrayLike = 1. * u.ohm,
        tau: ArrayLike = 5. * u.ms,
        tau_a: ArrayLike = 100. * u.ms,
        V_th: ArrayLike = 1. * u.mV,
        V_reset: ArrayLike = 0. * u.mV,
        V_rest: ArrayLike = 0. * u.mV,
        beta: ArrayLike = 0.1 * u.mV,
        spk_fun: Callable = surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        V_initializer: Callable = init.Constant(0. * u.mV),
        a_initializer: Callable = init.Constant(0.),
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spk_fun, spk_reset=spk_reset)

        # parameters
        self.R = init.param(R, self.varshape)
        self.tau = init.param(tau, self.varshape)
        self.tau_a = init.param(tau_a, self.varshape)
        self.V_th = init.param(V_th, self.varshape)
        self.V_reset = init.param(V_reset, self.varshape)
        self.V_rest = init.param(V_rest, self.varshape)
        self.beta = init.param(beta, self.varshape)

        # functions
        self.V_initializer = V_initializer
        self.a_initializer = a_initializer

    def init_state(self, batch_size: int = None, **kwargs):
        self.V = HiddenState(init.param(self.V_initializer, self.varshape, batch_size))
        self.a = HiddenState(init.param(self.a_initializer, self.varshape, batch_size))

    def reset_state(self, batch_size: int = None, **kwargs):
        self.V.value = init.param(self.V_initializer, self.varshape, batch_size)
        self.a.value = init.param(self.a_initializer, self.varshape, batch_size)

    def get_spike(self, V=None, a=None):
        V = self.V.value if V is None else V
        a = self.a.value if a is None else a
        v_scaled = (V - self.V_th - self.beta * a) / (self.V_th - self.V_reset)
        return self.spk_fun(v_scaled)

    def update(self, x=0. * u.mA):
        last_v = self.V.value
        last_a = self.a.value
        lst_spk = self.get_spike(last_v, last_a)
        V_th = self.V_th if self.spk_reset == 'soft' else jax.lax.stop_gradient(last_v)
        V = last_v - (V_th - self.V_reset) * lst_spk
        a = last_a + lst_spk
        # membrane potential
        dv = lambda v: (-v + self.V_rest + self.R * self.sum_current_inputs(x, v)) / self.tau
        da = lambda a: -a / self.tau_a
        V = exp_euler_step(dv, V)
        a = exp_euler_step(da, a)
        self.V.value = self.sum_delta_inputs(V)
        self.a.value = a
        return self.get_spike(self.V.value, self.a.value)
