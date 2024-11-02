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

import argparse
import os.path
import pickle
import platform
import time
from functools import reduce
from typing import Any, Callable, Union

import brainscale
import braintools as bts
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from torch.utils.data import DataLoader, IterableDataset

import brainstate as bst

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default='diag', choices=['diag', 'expsm_diag', 'bptt'])
args, _ = parser.parse_known_args()

# training method
if args.method != 'bptt':
    parser.add_argument("--vjp_time", type=str, default='t', choices=['t', 't_minus_1'],
                        help="The VJP time,should be t or t-1.")
    if args.method != 'diag':
        parser.add_argument("--etrace_decay", type=float, default=0.99,
                            help="The time constant of eligibility trace ")

# Learning parameters
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs.")
parser.add_argument("--dt", type=float, default=1., help="The simulation time step.")

# dataset
parser.add_argument("--mode", type=str, default="train", choices=['train', 'sim'], help="The computing mode.")
parser.add_argument("--n_data_worker", type=int, default=1, help="Number of data loading workers (default: 4)")
parser.add_argument("--t_delay", type=float, default=1e3, help="Deta delay length.")
parser.add_argument("--t_fixation", type=float, default=10., help="")

# training parameters
parser.add_argument("--exp_name", type=str, default='', help="")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="The ratio for network simulation.")
parser.add_argument("--acc_th", type=float, default=0.90, help="")
parser.add_argument("--filepath", type=str, default='', help="The name for the current experiment.")

# regularization parameters
parser.add_argument("--spk_reg_factor", type=float, default=0.0, help="Spike regularization factor.")
parser.add_argument("--spk_reg_rate", type=float, default=10., help="Target firing rate.")
parser.add_argument("--v_reg_factor", type=float, default=0.0, help="Voltage regularization factor.")
parser.add_argument("--v_reg_low", type=float, default=-20., help="The lowest voltage for regularization.")
parser.add_argument("--v_reg_high", type=float, default=1.4, help="The highest voltage for regularization.")
parser.add_argument("--weight_L1", type=float, default=0.0, help="The weight L1 regularization.")
parser.add_argument("--weight_L2", type=float, default=0.0, help="The weight L2 regularization.")

# GIF parameters
parser.add_argument("--diff_spike", type=int, default=0, help="0: False, 1: True.")
parser.add_argument("--n_rec", type=int, default=200, help="Number of recurrent neurons.")
parser.add_argument("--A2", type=float, default=1.)
parser.add_argument("--tau_I2", type=float, default=1500.)
parser.add_argument("--tau_neu", type=float, default=100.)
parser.add_argument("--tau_syn", type=float, default=100.)
parser.add_argument("--tau_o", type=float, default=10.)
parser.add_argument("--ff_scale", type=float, default=6.)
parser.add_argument("--rec_scale", type=float, default=2.)

global_args = parser.parse_args()
PyTree = Any


def format_sim_epoch(sim: Union[int, float], length: int):
    if 0. <= sim < 1.:
        return int(length * sim)
    else:
        return int(sim)


def raster_plot(sp_matrix, times):
    """Get spike raster plot which displays the spiking activity
    of a group of neurons over time.

    Parameters
    ----------
    sp_matrix : bnp.ndarray
        The matrix which record spiking activities.
    times : bnp.ndarray
        The time steps.

    Returns
    -------
    raster_plot : tuple
        Include (neuron index, spike time).
    """
    sp_matrix = np.asarray(sp_matrix)
    times = np.asarray(times)
    elements = np.where(sp_matrix > 0.)
    index = elements[1]
    times = times[elements[0]]
    return index, times


class ExponentialSmooth(object):
    def __init__(self, decay: float = 0.8):
        self.decay = decay
        self.value = None

    def update(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * value
        return self.value

    def __call__(self, value, i: int = None):
        return self.update(value)  # / (1. - self.decay ** (i + 1))


class GIF(bst.nn.Neuron):
    def __init__(
        self,
        in_size,
        V_rest=0. * u.mV,
        V_th_inf=1. * u.mV,
        R=1. * u.ohm,
        tau=20. * u.ms,
        tau_I2=50. * u.ms,
        A2=0. * u.mA,
        V_initializer: Callable = bst.init.ZeroInit(unit=u.mV),
        I2_initializer: Callable = bst.init.ZeroInit(unit=u.mA),
        spike_fun: Callable = bst.surrogate.ReluGrad(),
        spk_reset: str = 'soft',
        name: str = None,
    ):
        super().__init__(in_size, name=name, spk_fun=spike_fun, spk_reset=spk_reset)

        # params
        self.V_rest = bst.init.param(V_rest, self.varshape, allow_none=False)
        self.V_th_inf = bst.init.param(V_th_inf, self.varshape, allow_none=False)
        self.R = bst.init.param(R, self.varshape, allow_none=False)
        self.tau = bst.init.param(tau, self.varshape, allow_none=False)
        self.tau_I2 = bst.init.param(tau_I2, self.varshape, allow_none=False)
        self.A2 = bst.init.param(A2, self.varshape, allow_none=False)

        # initializers
        self._V_initializer = V_initializer
        self._I2_initializer = I2_initializer

    def init_state(self, batch_size=None):
        self.V = brainscale.ETraceVar(bst.init.param(self._V_initializer, self.varshape, batch_size))
        self.I2 = brainscale.ETraceVar(bst.init.param(self._I2_initializer, self.varshape, batch_size))

    def update(self, x=0.):
        t = bst.environ.get('t')
        last_spk = self.get_spike()
        if global_args.diff_spike == 0:
            last_spk = jax.lax.stop_gradient(last_spk)
        last_V = self.V.value - self.V_th_inf * last_spk
        last_I2 = self.I2.value - self.A2 * last_spk
        I2 = bst.nn.exp_euler_step(lambda i2: - i2 / self.tau_I2, last_I2)
        V = bst.nn.exp_euler_step(lambda v, Iext: (- v + self.V_rest + self.R * Iext) / self.tau, last_V, (x + I2))
        self.I2.value = I2
        self.V.value = V
        # output
        inp = self.V.value - self.V_th_inf
        inp = jax.nn.standardize(u.get_magnitude(inp))
        return inp

    def get_spike(self, V=None):
        V = self.V.value if V is None else V
        spk = self.spk_fun((V - self.V_th_inf) / self.V_th_inf)
        return spk


class GifNet(bst.nn.Module):
    def __init__(
        self,
        num_in,
        num_rec,
        num_out,
        args,
        filepath: str = None
    ):
        super().__init__()

        self.filepath = filepath

        ff_init = bst.init.KaimingNormal(scale=args.ff_scale, unit=u.mA)
        rec_init = bst.init.KaimingNormal(scale=args.rec_scale, unit=u.mA)
        w = u.math.concatenate([ff_init((num_in, num_rec)), rec_init((num_rec, num_rec))], axis=0)

        # parameters
        self.num_in = num_in
        self.num_rec = num_rec
        self.num_out = num_out
        self.ir2r = brainscale.nn.Linear(num_in + num_rec, num_rec, w_init=w, b_init=bst.init.ZeroInit(unit=u.mA))
        self.exp = brainscale.nn.Expon(num_rec, tau=args.tau_syn * u.ms, g_initializer=bst.init.ZeroInit(unit=u.mA))
        tau_I2 = bst.random.uniform(100., args.tau_I2 * 1.5, num_rec)
        self.r = GIF(
            num_rec,
            V_rest=0. * u.mV,
            V_th_inf=1. * u.mV,
            spike_fun=bst.surrogate.ReluGrad(),
            A2=args.A2 * u.mA,
            tau=args.tau_neu * u.ms,
            tau_I2=tau_I2 * u.ms
        )
        self.out = brainscale.nn.LeakyRateReadout(
            num_rec,
            num_out,
            tau=args.tau_o * u.ms,
            w_init=bst.init.KaimingNormal(scale=args.ff_scale)
        )

    def membrane_reg(self, mem_low: float, mem_high: float, factor: float = 0.):
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(bst.nn.Neuron).unique().values()
            # evaluate the membrane potential
            for l in neurons:
                loss += jnp.square(jnp.mean(jax.nn.relu(l.V.value - mem_high) ** 2 +
                                            jax.nn.relu(mem_low - l.V.value) ** 2))
            loss = loss * factor
        return loss

    def spike_reg(self, target_fr: float, factor: float = 0.):
        # target_fr: Hz
        loss = 0.
        if factor > 0.:
            # extract all Neuron models
            neurons = self.nodes().subset(bst.nn.Neuron).unique().values()
            # evaluate the spiking dynamics
            for l in neurons:
                loss += (jnp.mean(l.get_spike()) - target_fr / 1e3 * bst.environ.get_dt()) ** 2
            loss = loss * factor
        return loss

    def to_state_dict(self):
        res = dict(
            ir2r=self.ir2r.weight_op.value,
            out=self.out.weight_op.value,
            tau_I2=self.r.tau_I2,
        )
        return jax.tree.map(np.asarray, res)

    def from_state_dict(self, state_dict: dict):
        state_dict = jax.tree.map(jnp.asarray, state_dict)
        self.ir2r.weight_op.value = state_dict['ir2r']
        self.out.weight_op.value = state_dict['out']
        self.r.tau_I2 = state_dict['tau_I2']

    def save(self, **kwargs):
        if self.filepath is not None:
            states = self.to_state_dict()
            states.update(kwargs)
            print(f'Saving the model to {self.filepath}/final_states.pkl ...')
            with open(f'{self.filepath}/final_states.pkl', 'wb') as f:
                pickle.dump(states, f)

    def restore(self):
        if self.filepath is not None:
            print(f'Loading the model from {self.filepath}/final_states.pkl ...')
            with open(f'{self.filepath}/final_states.pkl', 'rb') as f:
                states = pickle.load(f)
            self.from_state_dict(states)

    def update(self, spikes):
        cond = self.ir2r(jnp.concatenate([spikes, self.r.get_spike()], axis=-1))
        out = self.r(self.exp(cond))
        return self.out(out)

    def verify(self, dataloader, x_func, num_show=5, sps_inc=10., filepath=None):
        def _step(index, x):
            with bst.environ.context(i=index, t=index * bst.environ.get_dt()):
                out = self.update(x)
            return out, self.r.get_spike(), self.r.V.value

        dataloader = iter(dataloader)
        xs, ys = next(dataloader)  # xs: [n_samples, n_steps, n_in]
        xs = jnp.asarray(x_func(xs))
        print(xs.shape, ys.shape)
        bst.nn.init_all_states(self, xs.shape[1])

        time_indices = np.arange(0, xs.shape[0])
        outs, sps, vs = bst.compile.for_loop(_step, time_indices, xs)
        outs = u.math.as_numpy(outs)
        sps = u.math.as_numpy(sps)
        vs = u.math.as_numpy(vs)
        vs = np.where(sps, vs + sps_inc, vs)

        ts = time_indices * bst.environ.get_dt()
        max_t = xs.shape[0] * bst.environ.get_dt()

        for i in range(num_show):
            fig, gs = bts.visualize.get_figure(4, 1, 2., 10.)

            ax_inp = fig.add_subplot(gs[0, 0])
            indices, times = raster_plot(xs[:, i], ts)
            ax_inp.plot(times, indices, '.')
            ax_inp.set_xlim(0., max_t)
            ax_inp.set_ylabel('Input Activity')

            ax = fig.add_subplot(gs[1, 0])
            plt.plot(ts, vs[:, i])
            # for j in range(0, self.r.num, 10):
            #   pass
            ax.set_xlim(0., max_t)
            ax.set_ylabel('Recurrent Potential')

            # spiking activity
            ax_rec = fig.add_subplot(gs[2, 0])
            indices, times = raster_plot(sps[:, i], ts)
            ax_rec.plot(times, indices, '.')
            ax_rec.set_xlim(0., max_t)
            ax_rec.set_ylabel('Recurrent Spiking')

            # decision activity
            ax_out = fig.add_subplot(gs[3, 0])
            for j in range(outs.shape[-1]):
                ax_out.plot(ts, outs[:, i, j], label=f'Readout {j}', alpha=0.7)
            ax_out.set_ylabel('Output Activity')
            ax_out.set_xlabel('Time [ms]')
            ax_out.set_xlim(0., max_t)
            plt.legend()

            if filepath:
                plt.savefig(f'{filepath}/{i}.png')

        if filepath is None:
            plt.show()
        plt.close()


class Trainer(object):
    """
    The training class with only loss.
    """

    def __init__(
        self,
        target: GifNet,
        opt: bst.optim.Optimizer,
        arguments: argparse.Namespace,
        filepath: str = None
    ):
        super().__init__()

        self.filepath = filepath
        self.file = None
        if filepath:
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath, exist_ok=True)
            self.file = open(f'{self.filepath}/loss.txt', 'w')

        # exponential smoothing
        self.smoother = ExponentialSmooth(0.8)

        # target network
        self.target = target

        # parameters
        self.args = arguments

        # optimizer
        self.opt = opt
        opt.register_trainable_weights(self.target.states().subset(bst.ParamState))

    def print(self, msg):
        print(msg)
        if self.file is not None:
            print(msg, file=self.file)

    def _acc(self, out, target):
        return jnp.mean(jnp.equal(target, jnp.argmax(jnp.mean(out, axis=0), axis=1)))

    def _loss(self, out, target):
        loss = bts.metric.softmax_cross_entropy_with_integer_labels(out, target).mean()

        # L1 regularization loss
        if self.args.weight_L1 != 0.:
            leaves = self.target.states().subset(bst.ParamState).to_dict_values()
            loss += self.args.weight_L1 * reduce(jnp.add, jax.tree.map(lambda x: jnp.sum(jnp.abs(x)), leaves))

        # membrane potential regularization loss
        if self.args.v_reg_factor != 0.:
            mem_low = self.args.v_reg_low
            mem_high = self.args.v_reg_high
            loss += self.target.membrane_reg(mem_low, mem_high, self.args.v_reg_factor)

        # spike regularization loss
        if self.args.spk_reg_factor != 0.:
            fr = self.args.spk_reg_rate
            loss += self.target.spike_reg(fr, self.args.spk_reg_factor)

        return loss

    @bst.compile.jit(static_argnums=(0,))
    def etrace_train(self, inputs, targets):
        # initialize the states
        bst.nn.init_all_states(self.target, inputs.shape[1])
        # weights
        weights = self.target.states().subset(bst.ParamState)

        # the model for a single step
        def _single_step(i, inp, fit: bool = True):
            with bst.environ.context(i=i, t=i * bst.environ.get_dt(), fit=fit):
                out = self.target(inp)
            return out

        # initialize the online learning model
        if self.args.method == 'expsm_diag':
            model = brainscale.DiagIODimAlgorithm(
                _single_step,
                int(self.args.etrace_decay) if self.args.etrace_decay > 1. else self.args.etrace_decay,
                vjp_time=self.args.vjp_time,
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif self.args.method == 'diag':
            model = brainscale.DiagParamDimAlgorithm(
                _single_step,
                vjp_time=self.args.vjp_time,
                mode=bst.mixin.Batching()
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif self.args.method == 'hybrid':
            model = brainscale.DiagHybridDimAlgorithm(
                _single_step,
                int(self.args.etrace_decay) if self.args.etrace_decay > 1. else self.args.etrace_decay,
                vjp_time=self.args.vjp_time,
                mode=bst.mixin.Batching()
            )
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')

        model.show_graph()

        def _etrace_grad(i, inp):
            # call the model
            out = model(i, inp, running_index=i)
            # calculate the loss
            loss = self._loss(out, targets)
            return loss, out

        def _etrace_step(prev_grads, x):
            # no need to return weights and states, since they are generated then no longer needed
            i, inp = x
            f_grad = bst.augment.grad(_etrace_grad, weights, has_aux=True, return_value=True)
            cur_grads, local_loss, out = f_grad(i, inp)
            next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
            return next_grads, (out, local_loss)

        def _etrace_train(indices_, inputs_):
            # forward propagation
            grads = jax.tree.map(jnp.zeros_like, weights.to_dict_values())
            grads, (outs, losses) = bst.compile.scan(_etrace_step, grads, (indices_, inputs_))
            # gradient updates
            grads = bst.functional.clip_grad_norm(grads, 1.)
            self.opt.update(grads)
            # accuracy
            return losses.mean(), outs

        # running indices
        indices = np.arange(inputs.shape[0])
        if self.args.warmup_ratio > 0:
            n_sim = format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
            bst.compile.for_loop(lambda i, inp: model(i, inp, running_index=i), indices[:n_sim], inputs[:n_sim])
            loss, outs = _etrace_train(indices[n_sim:], inputs[n_sim:])
        else:
            loss, outs = _etrace_train(indices, inputs)

        # returns
        return loss, self._acc(outs, targets)

    @bst.compile.jit(static_argnums=(0,))
    def bptt_train(self, inputs, targets):
        # running indices
        indices = np.arange(inputs.shape[0])

        # initialize the states
        bst.nn.init_all_states(self.target, inputs.shape[1])

        # the model for a single step
        def _single_step(i, inp, fit: bool = True):
            with bst.environ.context(i=i, t=i * bst.environ.get_dt(), fit=fit):
                out = self.target(inp)
            return out

        def _run_step_train(i, inp):
            with bst.environ.context(i=i, t=i * bst.environ.get_dt()):
                out = self.target(inp)
                loss = self._loss(out, targets)
            return out, loss

        def _bptt_grad_step():
            if self.args.warmup_ratio > 0:
                n_sim = format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
                _ = bst.compile.for_loop(_single_step, indices[:n_sim], inputs[:n_sim])
                outs, losses = bst.compile.for_loop(_run_step_train, indices[n_sim:], inputs[n_sim:])
            else:
                outs, losses = bst.compile.for_loop(_run_step_train, indices, inputs)
            return losses.mean(), outs

        # gradients
        weights = self.target.states().subset(bst.ParamState)
        grads, loss, outs = bst.augment.grad(_bptt_grad_step, weights, has_aux=True, return_value=True)()

        # optimization
        grads = bst.functional.clip_grad_norm(grads, 1.)
        self.opt.update(grads)

        return loss, self._acc(outs, targets)

    def f_train(self, train_loader, x_func, y_func):
        self.print(self.args)

        max_acc = 0.
        try:
            for i, (x_local, y_local) in enumerate(train_loader):
                if i >= self.args.epochs:
                    break

                t0 = time.time()
                # inputs and targets
                x_local = x_func(x_local)
                y_local = y_func(y_local)

                # training
                loss, acc = (
                    self.bptt_train(x_local, y_local)
                    if self.args.method == 'bptt' else
                    self.etrace_train(x_local, y_local)
                )
                t = time.time() - t0
                self.print(f'Batch {i:4d}, loss = {float(loss):.8f}, acc = {float(acc):.6f}, time = {t:.5f} s')
                if (i + 1) % 100 == 0:
                    self.opt.lr.step_epoch()

                # accuracy
                avg_acc = self.smoother(acc)
                if avg_acc > max_acc:
                    max_acc = avg_acc
                    if platform.platform().startswith('Linux'):
                        self.target.save(loss=loss, acc=acc)
                if max_acc > self.args.acc_th:
                    self.print(f'The training accuracy is greater than {self.args.acc_th * 100}%. Training is stopped.')
                    break
        finally:
            if self.file is not None:
                self.file.close()


class DMS(IterableDataset):
    """
    Delayed match-to-sample task.
    """
    times = ('dead', 'fixation', 'sample', 'delay', 'test')
    output_features = ('non-match', 'match')

    _rotate_choice = {
        '0': 0,
        '45': 1,
        '90': 2,
        '135': 3,
        '180': 4,
        '225': 5,
        '270': 6,
        '315': 7,
        '360': 8,
    }

    def __init__(
        self,
        dt=1. * u.ms,
        t_fixation=500. * u.ms,
        t_sample=500. * u.ms,
        t_delay=1000. * u.ms,
        t_test=500. * u.ms,
        limits=(0., np.pi * 2),
        rotation_match='0',
        kappa=3.,
        bg_fr=1. * u.Hz,
        n_input=100,
        firing_rate=100. * u.Hz,
    ):
        super().__init__()

        self.num_inputs = n_input
        self.num_outputs = 2
        self.firing_rate = firing_rate

        self.dt = dt
        # time
        self.t_fixation = int(t_fixation / dt)
        self.t_sample = int(t_sample / dt)
        self.t_delay = int(t_delay / dt)
        self.t_test = int(t_test / dt)
        self.num_steps = self.t_fixation + self.t_sample + self.t_delay + self.t_test
        self._times = {
            'fixation': self.t_fixation,
            'sample': self.t_sample,
            'delay': self.t_delay,
            'test': self.t_test,
        }
        test_onset = self.t_fixation + self.t_sample + self.t_delay
        self._test_onset = test_onset
        self.test_time = slice(test_onset, test_onset + self.t_test)
        self.fix_time = slice(0, test_onset)
        self.sample_time = slice(self.t_fixation, self.t_fixation + self.t_sample)

        # input shape
        self.rotation_match = rotation_match
        self._rotate = self._rotate_choice[rotation_match]
        self.bg_fr = bg_fr  # background firing rate
        self.v_min = limits[0]
        self.v_max = limits[1]
        self.v_range = limits[1] - limits[0]

        # Tuning function data
        self.n_motion_choice = 8
        self.kappa = kappa  # concentration scaling factor for von Mises

        # Generate list of preferred directions
        # dividing neurons by 2 since two equal
        # groups representing two modalities
        pref_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.num_inputs)

        # Generate list of possible stimulus directions
        stim_dirs = np.arange(self.v_min, self.v_max, self.v_range / self.n_motion_choice)

        d = np.cos(np.expand_dims(stim_dirs, 1) - pref_dirs)
        self.motion_tuning = np.exp(self.kappa * d) / np.exp(self.kappa)

    def sample_a_trial(self, *index):
        fr = np.asarray(self.firing_rate * bst.environ.get_dt())
        bg_fr = np.asarray(self.bg_fr * bst.environ.get_dt())
        return self._dms(self.num_steps,
                         self.num_inputs,
                         self.n_motion_choice,
                         self.motion_tuning,
                         self.sample_time,
                         self.test_time,
                         fr,
                         bg_fr,
                         self._rotate)

    @staticmethod
    @njit
    def _dms(num_steps, num_inputs, n_motion_choice, motion_tuning,
             sample_time, test_time, fr, bg_fr, rotate_dir):
        # data
        X = np.zeros((num_steps, num_inputs))

        # sample
        match = np.random.randint(2)
        sample_dir = np.random.randint(n_motion_choice)

        # Generate the sample and test stimuli based on the rule
        if match == 1:  # match trial
            test_dir = (sample_dir + rotate_dir) % n_motion_choice
        else:
            test_dir = np.random.randint(n_motion_choice)
            while test_dir == ((sample_dir + rotate_dir) % n_motion_choice):
                test_dir = np.random.randint(n_motion_choice)

        # SAMPLE stimulus
        X[sample_time] += motion_tuning[sample_dir] * fr
        # TEST stimulus
        X[test_time] += motion_tuning[test_dir] * fr
        X += bg_fr

        # to spiking
        X = np.random.random(X.shape) < X
        X = X.astype(np.float32)

        # can use a greater weight for test period if needed
        return X, match

    def __iter__(self):
        while True:
            yield self.sample_a_trial()


@njit
def numba_seed(seed):
    np.random.seed(seed)


def load_data():
    # loading the data
    task = DMS(
        dt=bst.environ.get_dt(),
        bg_fr=1. * u.Hz,
        t_fixation=global_args.t_fixation * u.ms,
        t_sample=500. * u.ms,
        t_delay=global_args.t_delay * u.ms,
        t_test=500. * u.ms,
        n_input=100,
        firing_rate=100. * u.Hz
    )
    train_loader = DataLoader(task, batch_size=global_args.batch_size)
    input_process = lambda x: jnp.asarray(x, dtype=bst.environ.dftype()).transpose(1, 0, 2)
    label_process = lambda x: jnp.asarray(x, dtype=bst.environ.ditype())
    global_args.warmup_ratio = task.num_steps - task.t_test
    global_args.n_out = task.num_outputs
    n_in = task.num_inputs
    return train_loader, input_process, label_process, n_in


def network_training():
    # environment setting
    bst.environ.set(dt=global_args.dt * u.ms)

    # loading the data
    train_loader, input_process, label_process, n_in = load_data()

    # creating the network and optimizer
    net = GifNet(n_in, global_args.n_rec, global_args.n_out, global_args, filepath=None)

    if global_args.mode == 'sim':
        if global_args.filepath:
            net.restore()
        net.verify(train_loader, input_process, num_show=5, sps_inc=10.)

    elif global_args.mode == 'train':
        opt = bst.optim.Adam(lr=global_args.lr, weight_decay=global_args.weight_L2)

        # creating the trainer
        trainer = Trainer(net, opt, global_args, filepath=None)
        trainer.f_train(train_loader, x_func=input_process, y_func=label_process)

    else:
        raise ValueError(f'Unknown mode: {global_args.mode}')


if __name__ == '__main__':
    pass
    network_training()
