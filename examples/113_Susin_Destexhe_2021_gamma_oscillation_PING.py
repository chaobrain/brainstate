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


#
# Implementation of the paper:
#
# - Susin, Eduarda, and Alain Destexhe. “Integration, coincidence detection and resonance in networks of
#   spiking neurons expressing gamma oscillations and asynchronous states.” PLoS computational biology 17.9 (2021): e1009416.
#
# PING Network for Generating Gamma Oscillation


import os

os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'

import brainunit as u
import brainstate as bst

from Susin_Destexhe_2021_gamma_oscillation import (
  get_inputs, visualize_simulation_results, RS_par, FS_par, AdEx
)


class PINGNet(bst.nn.DynamicsGroup):
  def __init__(self):
    super().__init__()

    self.num_exc = 20000
    self.num_inh = 5000
    self.exc_syn_tau = 1. * u.ms
    self.inh_syn_tau = 7.5 * u.ms
    self.exc_syn_weight = 5. * u.nS
    self.inh_syn_weight = 3.34 * u.nS
    self.ext_weight = 4. * u.nS
    self.delay = 1.5 * u.ms

    # neuronal populations
    RS_par_ = RS_par.copy()
    FS_par_ = FS_par.copy()
    RS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
    FS_par_.update(Vth=-50 * u.mV, V_sp_th=-40 * u.mV)
    self.rs_pop = AdEx(self.num_exc, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **RS_par_)
    self.fs_pop = AdEx(self.num_inh, tau_e=self.exc_syn_tau, tau_i=self.inh_syn_tau, **FS_par_)
    self.ext_pop = bst.nn.PoissonEncoder(self.num_exc)

    # Poisson inputs
    self.ext_to_FS = bst.nn.DeltaProj(
      comm=bst.event.FixedProb(self.num_exc, self.num_inh, 0.02, self.ext_weight),
      post=self.fs_pop,
      label='ge'
    )
    self.ext_to_RS = bst.nn.DeltaProj(
      comm=bst.event.FixedProb(self.num_exc, self.num_exc, 0.02, self.ext_weight),
      post=self.rs_pop,
      label='ge'
    )

    # synaptic projections
    self.RS_to_FS = bst.nn.DeltaProj(
      self.rs_pop.prefetch('spike').delay.at(self.delay),
      comm=bst.event.FixedProb(self.num_exc, self.num_inh, 0.02, self.exc_syn_weight),
      post=self.fs_pop,
      label='ge'
    )
    self.RS_to_RS = bst.nn.DeltaProj(
      self.rs_pop.prefetch('spike').delay.at(self.delay),
      comm=bst.event.FixedProb(self.num_exc, self.num_exc, 0.02, self.exc_syn_weight),
      post=self.rs_pop,
      label='ge'
    )
    self.FS_to_RS = bst.nn.DeltaProj(
      self.fs_pop.prefetch('spike').delay.at(self.delay),
      comm=bst.event.FixedProb(self.num_inh, self.num_exc, 0.02, self.inh_syn_weight),
      post=self.rs_pop,
      label='gi'
    )
    self.FS_to_FS = bst.nn.DeltaProj(
      self.fs_pop.prefetch('spike').delay.at(self.delay),
      comm=bst.event.FixedProb(self.num_inh, self.num_inh, 0.02, self.inh_syn_weight),
      post=self.fs_pop,
      label='gi'
    )

  def update(self, i, t, freq):
    with bst.environ.context(t=t, i=i):
      ext_spikes = self.ext_pop(freq)
      self.ext_to_FS(ext_spikes)
      self.ext_to_RS(ext_spikes)

      self.RS_to_RS()
      self.RS_to_FS()

      self.FS_to_RS()
      self.FS_to_FS()

      self.rs_pop()
      self.fs_pop()

      return {
        'FS.V0': self.fs_pop.V.value[0],
        'RS.V0': self.rs_pop.V.value[0],
        'FS.spike': self.fs_pop.spike.value,
        'RS.spike': self.rs_pop.spike.value
      }


def simulate_ping_net():
  with bst.environ.context(dt=0.1 * u.ms):
    # inputs
    duration = 6e3 * u.ms
    varied_rates = get_inputs(2. * u.Hz, 3. * u.Hz, 50. * u.ms, 3150 * u.ms, 600 * u.ms, 1e3 * u.ms, duration)

    # network
    net = bst.nn.init_all_states(PINGNet())

    # simulation
    times = u.math.arange(0. * u.ms, duration, bst.environ.get_dt())
    indices = u.math.arange(0, len(times))
    returns = bst.compile.for_loop(net.update, indices, times, varied_rates, pbar=bst.compile.ProgressBar(100))

    # visualization
    visualize_simulation_results(
      times=times,
      spikes={'FS': (returns['FS.spike'], 'inh'),
              'RS': (returns['RS.spike'], 'exc')},
      example_potentials={'FS': returns['FS.V0'],
                          'RS': returns['RS.V0']},
      varied_rates=varied_rates,
      xlim=(2e3 * u.ms, 3.4e3 * u.ms),
      t_lfp_start=1e3 * u.ms,
      t_lfp_end=5e3 * u.ms
    )


simulate_ping_net()
