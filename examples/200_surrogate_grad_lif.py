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


"""
Reproduce the results of the``spytorch`` tutorial 1:

- https://github.com/surrogate-gradient-learning/spytorch/blob/master/notebooks/SpyTorchTutorial1.ipynb

"""

import time

import braintools as bts
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import brainstate as bst


class SNN(bst.nn.DynamicsGroup):
    def __init__(self, num_in, num_rec, num_out):
        super(SNN, self).__init__()

        # parameters
        self.num_in = num_in
        self.num_rec = num_rec
        self.num_out = num_out

        # synapse: i->r
        self.i2r = bst.nn.Sequential(
            bst.nn.Linear(
                num_in, num_rec,
                w_init=bst.init.KaimingNormal(scale=7*(1-(u.math.exp(-bst.environ.get_dt()/(1*u.ms)))), unit=u.mA),
                b_init=bst.init.ZeroInit(unit=u.mA)
            ),
            bst.nn.Expon(num_rec, tau=5. * u.ms, g_initializer=bst.init.Constant(0. * u.mA))
        )
        # recurrent: r
        self.r = bst.nn.LIF(
            num_rec, tau=20 * u.ms, V_reset=0 * u.mV,
            V_rest=0 * u.mV, V_th=1. * u.mV,
            spk_fun=bst.surrogate.ReluGrad()
        )
        # synapse: r->o
        self.r2o = bst.nn.Linear(num_rec, num_out, w_init=bst.init.KaimingNormal())
        # # output: o
        self.o = bst.nn.Expon(num_out, tau=10. * u.ms, g_initializer=bst.init.Constant(0.))

    def update(self, spike):
        return self.o(self.r2o(self.r(self.i2r(spike))))

    def predict(self, spike):
        rec_spikes = self.r(self.i2r(spike))
        out = self.o(self.r2o(rec_spikes))
        return self.r.V.value, rec_spikes, out


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5, show=True):
    fig, gs = bts.visualize.get_figure(*dim, 3, 3)
    if spk is not None:
        mem[spk > 0.0] = spike_height
    if isinstance(mem, u.Quantity):
        mem = mem.to_decimal(u.mV)
    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(mem[:, i])
    if show:
        plt.show()


def print_classification_accuracy(output, target):
    """ Dirty little helper function to compute classification accuracy. """
    m = u.math.max(output, axis=0)  # max over time
    am = u.math.argmax(m, axis=1)  # argmax over output units
    acc = u.math.mean(target == am)  # compare to labels
    print("Accuracy %.3f" % acc)


def predict_and_visualize_net_activity(net):
    bst.nn.init_all_states(net, batch_size=num_sample)
    vs, spikes, outs = bst.compile.for_loop(net.predict, x_data, pbar=bst.compile.ProgressBar(10))
    plot_voltage_traces(vs, spikes, spike_height=5 * u.mV, show=False)
    plot_voltage_traces(outs)
    print_classification_accuracy(outs, y_data)


with bst.environ.context(dt=1.0 * u.ms):
    # network
    net = SNN(100, 4, 2)

    # dataset
    num_step = 200
    num_sample = 256
    freq = 5 * u.Hz
    x_data = bst.random.rand(num_step, num_sample, net.num_in) < freq * bst.environ.get_dt()
    y_data = u.math.asarray(bst.random.rand(num_sample) < 0.5, dtype=int)

    # Before training
    predict_and_visualize_net_activity(net)

    # brainstate optimizer
    optimizer = bst.optim.Adam(lr=3e-3, beta1=0.9, beta2=0.999)
    optimizer.register_trainable_weights(net.states(bst.ParamState))


    # # optax optimizer
    # import optax
    # optimizer = bst.optim.OptaxOptimizer(net.states(bst.ParamState), optax.adam(1e-3))

    def loss_fn():
        predictions = bst.compile.for_loop(net.update, x_data)
        predictions = u.math.mean(predictions, axis=0)  # [T, B, C] -> [B, C]
        return bts.metric.softmax_cross_entropy_with_integer_labels(predictions, y_data).mean()


    @bst.compile.jit
    def train_fn():
        bst.nn.init_all_states(net, batch_size=num_sample)
        grads, l = bst.augment.grad(loss_fn, net.states(bst.ParamState), return_value=True)()
        optimizer.update(grads)
        return l


    # train the network
    train_losses = []
    t0 = time.time()
    for i in range(1, 3001):
        loss = train_fn()
        train_losses.append(loss)
        if i % 100 == 0:
            print(f'Train {i} epoch, loss = {loss:.4f}, used time {time.time() - t0:.4f} s')
            t0 = time.time()

    # visualize the training losses
    plt.plot(np.asarray(jnp.asarray(train_losses)))
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")

    # predict the output according to the input data
    predict_and_visualize_net_activity(net)
