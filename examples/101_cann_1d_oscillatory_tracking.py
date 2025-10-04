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

# Implementation of the paper:
#
# - Si Wu, Kosuke Hamaguchi, and Shun-ichi Amari. “Dynamics and computation of continuous attractors.” Neural computation 20.4 (2008): 994-1025.
#
# - Mi, Y., Fung, C. C., Wong, M. K. Y., & Wu, S. (2014). Spike frequency adaptation implements anticipative tracking in continuous attractor neural networks. Advances in neural information processing systems, 1(January), 505.


import brainunit as u
import jax
import matplotlib.pyplot as plt
import brainstate
import numpy as np
from matplotlib import animation
from matplotlib.gridspec import GridSpec



class CANN1D(brainstate.nn.Dynamics):
    def __init__(
        self, num, tau=1., tau_v=50., k=1., a=0.3, A=0.2, J0=1.,
        z_min=-u.math.pi, z_max=u.math.pi, m=0.3
    ):
        super().__init__(num)

        # parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.m = m

        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.x = u.math.linspace(z_min, z_max, num)  # The encoded feature values
        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density

        # The connection matrix
        self.conn_mat = self.make_conn()

    def init_state(self, *args, **kwargs):
        # variables
        self.r = brainstate.HiddenState(u.math.zeros(self.varshape))
        self.u = brainstate.HiddenState(u.math.zeros(self.varshape))
        self.v = brainstate.HiddenState(u.math.zeros(self.varshape))

    def dist(self, d):
        d = u.math.remainder(d, self.z_range)
        d = u.math.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self):
        x_left = u.math.reshape(self.x, (-1, 1))
        x_right = u.math.repeat(self.x.reshape((1, -1)), len(self.x), axis=0)
        d = self.dist(x_left - x_right)
        conn = self.J0 * u.math.exp(-0.5 * u.math.square(d / self.a)) / (u.math.sqrt(2 * u.math.pi) * self.a)
        return conn

    def get_stimulus_by_pos(self, pos):
        return self.A * u.math.exp(-0.25 * u.math.square(self.dist(self.x - pos) / self.a))

    def update(self, inp):
        r1 = u.math.square(self.u.value)
        r2 = 1.0 + self.k * u.math.sum(r1)
        self.r.value = r1 / r2
        Irec = u.math.dot(self.conn_mat, self.r.value)
        self.u.value += (-self.u.value + Irec + inp - self.v.value) / self.tau * brainstate.environ.get_dt()
        self.v.value += (-self.v.value + self.m * self.u.value) / self.tau_v * brainstate.environ.get_dt()


def animate_1d(us, vs, frame_step=1, frame_delay=5,
               xlabel=None, ylabel=None, title_size=12):
    dt = brainstate.environ.get_dt()
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    gs = GridSpec(1, 1, figure=fig)
    fig.add_subplot(gs[0, 0])

    def frame(i):
        t = i * dt
        fig.clf()
        plt.plot(cann.x, np.asarray(get_inp(t)), label='Iext')
        plt.plot(cann.x, us[i], label='u')
        plt.plot(cann.x, vs[i], label='v')
        plt.legend()
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        fig.suptitle(t=f"Time: {t:.2f} ms", fontsize=title_size, fontweight='bold')
        return [fig.gca()]

    anim_result = animation.FuncAnimation(
        fig=fig,
        func=frame,
        frames=np.arange(1, us.shape[0], frame_step),
        init_func=None,
        interval=frame_delay,
        repeat_delay=3000
    )
    plt.show()


dur1, dur2, dur3 = 100., 2000., 500.


@jax.jit
def get_inp(t):
    pos = u.math.where(t < dur1, 0., u.math.where(t < dur1 + dur2, final_pos * (t - dur1) / (dur2 - dur1), final_pos))
    inp = cann.get_stimulus_by_pos(pos)
    return inp


brainstate.environ.set(dt=0.1)
cann = CANN1D(num=512)
cann.init_state()


def run_step(t):
    with brainstate.environ.context(t=t):
        cann(get_inp(t))
        return cann.u.value, cann.v.value


final_pos = cann.a / cann.tau_v * 0.6 * dur2

times = u.math.arange(0, dur1 + dur2 + dur3, brainstate.environ.get_dt())
us, vs = brainstate.compile.for_loop(run_step, times)
animate_1d(us, vs, frame_step=30, frame_delay=5)
