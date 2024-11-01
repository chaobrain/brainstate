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
# - Diesmann, Markus, Marc-Oliver Gewaltig, and Ad Aertsen. “Stable propagation of synchronous spiking in cortical neural networks.” Nature 402.6761 (1999): 529-533.
#

import braintools as bts
import brainunit as u
import jax
import matplotlib.pyplot as plt

import brainstate as bst

duration = 100. * u.ms

# Neuron model parameters
Vr = -70. * u.mV
Vt = -55. * u.mV
tau_m = 10. * u.ms
tau_ref = 1. * u.ms
tau_psp = 0.325 * u.ms
weight = 4.86 * u.mV
noise = 39.24 * u.mV
spike_sigma = 1. * u.ms

# Neuron groups
n_groups = 10
group_size = 100

# Synapse parameter
delay = 5.0 * u.ms  # ms


# neuron model
# ------------


class Population(bst.nn.Neuron):
  def __init__(self, in_size, **kwargs):
    super().__init__(in_size, **kwargs)

  def init_state(self, *args, **kwargs):
    self.V = bst.HiddenState(Vr + bst.random.random(self.num) * (Vt - Vr))
    self.x = bst.HiddenState(u.math.zeros(self.num) * u.mV)
    self.y = bst.HiddenState(u.math.zeros(self.num) * u.mV)
    self.spike = bst.ShortTermState(u.math.zeros(self.num, dtype=bool))
    self.t_last_spike = bst.ShortTermState(u.math.ones(self.num) * -1e7 * u.ms)

  def update(self):
    dv = lambda V, x: (-(V - Vr) + x) / tau_m
    dx = lambda x, y: (-x + y) / tau_psp
    dy_f = lambda y: -y / tau_psp + 25.27 * u.mV / u.ms
    dy_g = lambda y: noise / u.ms ** 0.5

    t = bst.environ.get('t')
    x = bst.nn.exp_euler_step(dx, self.x.value, self.y.value)
    y = bst.nn.exp_euler_step(dy_f, dy_g, self.y.value)
    V = bst.nn.exp_euler_step(dv, self.V.value, self.x.value)
    in_ref = (t - self.t_last_spike.value) < tau_ref
    V = u.math.where(in_ref, self.V.value, V)
    self.x.value = x
    self.y.value = y
    self.spike.value = V >= Vt
    self.t_last_spike.value = u.math.where(self.spike.value, t, self.t_last_spike.value)
    self.V.value = u.math.where(self.spike.value, Vr, V)
    return self.spike.value


# synaptic  model
# ---------------

class Projection(bst.nn.Synapse):
  def __init__(self, group, **kwargs):
    super().__init__(group.varshape, keep_size=True, **kwargs)

    # neuron group
    self.group = group

    # variables
    self.g = bst.nn.Delay(
      jax.ShapeDtypeStruct([self.group.num], bst.environ.dftype()) * u.mV,
      entries={'I': delay}
    )

  def update(self, ext_spike):
    # synapse model between external and group 1
    g = u.math.zeros(self.group.num, unit=u.mV)
    g[:group_size] = weight * ext_spike.sum()
    # feed-forward connection
    for i in range(1, n_groups):
      s1 = (i - 1) * group_size
      s2 = i * group_size
      s3 = (i + 1) * group_size
      g[s2: s3] = weight * self.group.spike.value[s1: s2].sum()
    # delay push
    self.g.update(g)
    # delay pull
    g = self.g.retrieve_at_step((delay / bst.environ.get_dt()).astype(int))
    # update group
    self.group.y.value += g


# network model
# ---------------

class Net(bst.nn.DynamicsGroup):
  def __init__(self, n_spike):
    super().__init__()
    times = bst.random.randn(n_spike) * spike_sigma + 20 * u.ms
    self.ext = bst.nn.SpikeTime(n_spike, times=times, indices=u.math.arange(n_spike), need_sort=False)
    self.pop = Population(in_size=n_groups * group_size)
    self.syn = Projection(self.pop)

  def update(self, t, i):
    with bst.environ.context(t=t, i=i):
      self.syn(self.ext())
      return self.pop()


# network running
# ---------------

def run_network(spike_num: int, ax):
  bst.random.seed(1)

  with bst.environ.context(dt=0.1 * u.ms):
    # initialization
    net = bst.nn.init_all_states(Net(spike_num))

    # simulation
    times = u.math.arange(0. * u.ms, duration, bst.environ.get_dt())
    indices = u.math.arange(times.size)
    spikes = bst.compile.for_loop(net.update, times, indices, pbar=bst.compile.ProgressBar(10))

  # visualization
  times = times.to_decimal(u.ms)
  t_indices, n_indices = u.math.where(spikes)
  ax.plot(times[t_indices], n_indices, 'k.', markersize=1)
  ax.set_xlabel('Time (ms)')
  ax.set_ylabel('Neuron index')


fig, gs = bts.visualize.get_figure(1, 2, 4, 4)
run_network(spike_num=40, ax=fig.add_subplot(gs[0, 0]))
run_network(spike_num=30, ax=fig.add_subplot(gs[0, 1]))
plt.show()
