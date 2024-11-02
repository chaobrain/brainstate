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
import time
from functools import partial, reduce
from typing import Union, Iterator, Optional, Any, Dict

import brainscale
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
from braintools import metric
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from tqdm import tqdm

import brainstate as bst

# --------------------------------------------------------------
# The parameters for the training
# --------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, default='diag', help="Training method.")
args, _ = parser.parse_known_args()

# training method
if args.method != 'bptt':
    parser.add_argument("--vjp_time", type=str, default='t', choices=['t', 't_minus_1'],
                        help="The VJP time,should be t or t-1.")
    if args.method != 'diag':
        parser.add_argument("--etrace_decay", type=float, default=0.9,
                            help="The time constant of eligibility trace ")

# Learning parameters
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs.")
parser.add_argument("--dt", type=float, default=0.1, help="The simulation time step.")
parser.add_argument("--loss", type=str, default='cel', choices=['cel', 'mse'], help="Loss function.")
parser.add_argument("--weight_L1", type=float, default=0.0, help="The weight L1 regularization.")
parser.add_argument("--weight_L2", type=float, default=0.0, help="The weight L2 regularization.")

# dataset
parser.add_argument("--dataset", type=str, default="copying", help="Choose between different datasets")
parser.add_argument("--n_data_worker", type=int, default=1, help="Number of data loading workers (default: 4)")
parser.add_argument("--data_length", type=int, default=100, help="Data Length.")

# training parameters
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="The ratio for network simulation.")
parser.add_argument("--exp_name", type=str, default='', help="The name for the current experiment.")

# Network parameters
parser.add_argument("--model", type=str, default='gru', help="The model types.")
parser.add_argument("--n_layer", type=int, default=1, help="Number of recurrent layers.")
parser.add_argument("--n_rec", type=int, default=200, help="Number of recurrent neurons.")

args = parser.parse_args()

# --------------------------------------------------------------
#
# Data
#
# --------------------------------------------------------------

PyTree = Any


def format_sim_epoch(sim: Union[int, float], length: int):
    if 0. <= sim < 1.:
        return int(length * sim)
    else:
        return int(sim)


class Checkpointer(orbax.checkpoint.CheckpointManager):
    def __init__(
        self,
        directory: str,
        max_to_keep: Optional[int] = None,
        save_interval_steps: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
            create=True
        )
        super().__init__(os.path.abspath(directory), options=options, metadata=metadata)

    def save_data(self, args: PyTree, step: int, **kwargs):
        return super().save(step, args=orbax.checkpoint.args.StandardSave(args), **kwargs)

    def load_data(self, args: PyTree, step: int = None, **kwargs):
        self.wait_until_finished()
        step = self.latest_step() if step is None else step
        tree = jax.tree_util.tree_map(orbax.checkpoint.utils.to_shape_dtype_struct, args)
        args = orbax.checkpoint.args.StandardRestore(tree)
        return super().restore(step, args=args, **kwargs)


class _CopyDataset(IterableDataset):
    def __init__(self, time_lag: int):
        super().__init__()
        self.seq_length = time_lag + 20

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ids = np.zeros(self.seq_length, dtype=int)
            ids[:10] = np.random.randint(1, 9, (10,))
            ids[-10:] = np.ones(10) * 9
            x = np.zeros([self.seq_length, 10])
            x[range(self.seq_length), ids] = 1
            yield x, ids[:10]


def _get_copying_task_for_rnn(args):
    dataset = _CopyDataset(args.data_length)
    generator = DataLoader(dataset, batch_size=args.batch_size)
    args.warmup_ratio = int(args.data_length + 10)
    return bst.util.DotDict(
        {'train_loader': generator,
         'in_shape': (10,),
         'out_shape': 10,
         'target_type': 'varied',
         'input_process': lambda x: jnp.asarray(x, dtype=bst.environ.dftype()).transpose(1, 0, 2),
         'label_process': lambda x: jnp.asarray(x).T, }
    )


def get_rnn_data(args, cache_dir=os.path.expanduser("./data")):
    data_to_fun = {
        'copying': _get_copying_task_for_rnn,
    }
    ret = data_to_fun[args.dataset.lower()](args)
    return ret


# --------------------------------------------------------------
#
# Model
#
# --------------------------------------------------------------


class RateRnnNet(bst.nn.Module):
    name2model = {
        'lstm': brainscale.nn.LSTMCell,
        'urlstm': brainscale.nn.URLSTMCell,
        'rnn': brainscale.nn.ValinaRNNCell,
        'gru': brainscale.nn.GRUCell,
    }

    def __init__(self, n_in, n_rec, n_out, n_layer, args, filepath: str = None):
        super().__init__()
        self.checkpointer = None if filepath is None else Checkpointer(filepath, max_to_keep=10)
        layers = []
        for _ in range(n_layer):
            layers.append(self.name2model[args.model](n_in, n_rec))
            n_in = n_rec
        self.layer = bst.nn.Sequential(*layers)
        self.readout = brainscale.nn.Linear(n_rec, n_out, as_etrace_weight=False)

    def update(self, x):
        return self.readout(self.layer(x))

    def save_state(self, step: int, **kwargs) -> bool:
        if self.checkpointer is None:
            return False
        else:
            params = bst.graph.treefy_states(self, bst.ParamState)
            self.checkpointer.save_data(params, step, **kwargs)
            return True

    def load_state(self, step: int = None, **kwargs):
        if self.checkpointer is None:
            return False
        else:
            params = bst.graph.treefy_states(self, bst.ParamState)
            param_values = self.checkpointer.load_data(params, step, **kwargs)
            bst.graph.update_states(self, param_values)
            return True


class Trainer(object):
    """
    The training class with only loss.
    """

    def __init__(self, target, opt: bst.optim.Optimizer, args, target_type, filepath: str | None = None):
        super().__init__()

        self.target_type = target_type
        self.filepath = filepath

        # target network
        self.target = target

        # parameters
        self.args = args

        # loss function
        if args.loss == 'mse':
            self.loss_fn = metric.squared_error
        elif args.loss == 'cel':
            self.loss_fn = metric.softmax_cross_entropy_with_integer_labels
        else:
            raise ValueError

        # gradient functions
        weights = self.target.states().subset(bst.ParamState)

        # optimizer
        self.opt = opt
        opt.register_trainable_weights(weights)

    def _loss(self, out, target):
        loss = self.loss_fn(out, target).mean()

        # L1 regularization loss
        if self.args.weight_L1 != 0.:
            leaves = self.target.states().subset(bst.ParamState).to_dict_values()
            loss += self.args.weight_L1 * reduce(jnp.add, jax.tree.map(metric.l1_loss, leaves))

        return loss

    def _single_step(self, i, inp, fit: bool = True):
        with bst.environ.context(i=i, fit=fit):
            out = self.target(inp)
        return out

    @bst.compile.jit(static_argnums=(0,))
    def etrace_train(self, inputs, target):

        indices = np.arange(inputs.shape[0])

        # initialize the states
        bst.nn.init_all_states(self.target, inputs.shape[1])

        # the model for a single step
        model = partial(self._single_step, fit=True)

        # initialize the online learning model
        if self.args.method == 'expsm_diag':
            model = brainscale.DiagIODimAlgorithm(model, self.args.etrace_decay, vjp_time=self.args.vjp_time)
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif self.args.method == 'diag':
            model = brainscale.DiagParamDimAlgorithm(model, vjp_time=self.args.vjp_time, mode=bst.mixin.Batching())
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        elif self.args.method == 'hybrid':
            model = brainscale.DiagHybridDimAlgorithm(model, self.args.etrace_decay, vjp_time=self.args.vjp_time,
                                                      mode=bst.mixin.Batching())
            model.compile_graph(0, jax.ShapeDtypeStruct(inputs.shape[1:], inputs.dtype))
        else:
            raise ValueError(f'Unknown online learning methods: {self.args.method}.')
        model.graph.show_graph()

        def _etrace_grad_step(i, inp, targets):
            # call the model
            out = model(i, inp, running_index=i)

            # calculate the loss
            loss = self._loss(out, targets)
            return loss, (out, None)

        def _etrace_train(indices, inputs, targets):
            weights = self.target.states(bst.ParamState)
            f_grad = bst.augment.grad(_etrace_grad_step, weights, has_aux=True, return_value=True)

            def f(prev_grads,
                  x):  # no need to return weights and states, since they are generated then no longer needed
                if self.target_type == 'varied':
                    i, inp, tar = x
                    cur_grads, local_loss, (out, _) = f_grad(i, inp, tar)
                    next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)
                else:
                    i, inp = x
                    cur_grads, local_loss, (out, _) = f_grad(i, inp, targets)
                    next_grads = jax.tree.map(lambda a, b: a + b, prev_grads, cur_grads)

                return next_grads, (out, local_loss)

            # forward propagation
            weights = {k: v.value for k, v in weights.items()}
            grads = jax.tree.map(lambda a: jnp.zeros_like(a), weights)
            if self.target_type == 'varied':
                grads, (outs, losses) = bst.compile.scan(f, grads, (indices, inputs, targets))
            else:
                grads, (outs, losses) = bst.compile.scan(f, grads, (indices, inputs))

            self.opt.update(grads)
            return losses.mean()

        def _etrace_predict(indices, inputs):
            bst.compile.for_loop(lambda i, inp: model(i, inp, running_index=i), indices, inputs)

        # running indices
        if self.args.warmup_ratio > 0:
            n_sim = format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
            _etrace_predict(indices[:n_sim], inputs[:n_sim])  # without loss
            r = _etrace_train(indices[n_sim:], inputs[n_sim:], target)
        else:
            r = _etrace_train(indices, inputs, target)

        # returns
        return r

    @bst.compile.jit(static_argnums=(0,))
    def bptt_train(self, inputs, targets):
        # initialize the states
        bst.nn.init_all_states(self.target, inputs.shape[1])

        def _run_step_train(i, inp, targets):
            with bst.environ.context(i=i):
                out = self.target(inp)
                loss = self._loss(out, targets)
                return out, loss

        def _bptt_grad_step(inputs, targets):
            # running indices
            indices = np.arange(inputs.shape[0])

            if self.args.warmup_ratio > 0:
                n_sim = format_sim_epoch(self.args.warmup_ratio, inputs.shape[0])
                _ = bst.compile.for_loop(self._single_step, indices[:n_sim], inputs[:n_sim])
                if self.target_type == 'varied':
                    outs, losses = bst.compile.for_loop(_run_step_train, indices[n_sim:], inputs[n_sim:], targets)
                else:
                    fun = partial(_run_step_train, targets=targets)
                    outs, losses = bst.compile.for_loop(fun, indices[n_sim:], inputs[n_sim:])
            else:
                func = partial(_run_step_train, targets=targets)
                outs, losses = bst.compile.for_loop(func, indices, inputs)
            return losses.mean(), (outs, None)

        # gradients
        f_grad = bst.augment.grad(_bptt_grad_step,
                                  self.target.states(bst.ParamState),
                                  has_aux=True,
                                  return_value=True)
        grads, loss, (outs, _) = f_grad(inputs, targets)

        # optimization
        self.opt.update(grads)

        return loss

    def f_train(self, dataloader, x_func, y_func, total: int):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        with open(f'{self.filepath}/loss.txt', 'w') as f:
            print(self.args, file=f)
            bar = tqdm(enumerate(dataloader), total=total)
            for i, (x_local, y_local) in bar:
                if i == total:
                    break
                # inputs and targets
                x_local = x_func(x_local)
                y_local = y_func(y_local)
                # training
                if self.args.method == 'bptt':
                    r = self.bptt_train(x_local, y_local)
                else:
                    r = self.etrace_train(x_local, y_local)
                print(f'Training {i:5d}, loss = {float(r):.8f}', file=f)
                bar.set_description(f'Training {i:5d}, loss = {float(r):.5f}', refresh=True)

                self.target.save_state(i)


def network_training():
    # environment setting
    # bst.environ.set(dt=args.dt)

    # get file path to output
    now = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(round(time.time() * 1000)) / 1000))
    filepath = f'figs/{args.model} {args.method} {args.dataset}/{now}'

    # loading the data
    dataset = get_rnn_data(args)
    args.n_out = dataset.out_shape

    # creating the network and optimizer
    net = RateRnnNet(np.prod(dataset.in_shape), args.n_rec, args.n_out, args.n_layer, args, filepath)
    opt = bst.optim.Adam(lr=args.lr, weight_decay=args.weight_L2)

    # creating the trainer
    trainer = Trainer(net, opt, args, dataset.target_type, filepath)
    trainer.f_train(
        dataset.train_loader,
        x_func=dataset.input_process,
        y_func=dataset.label_process,
        total=args.epochs
    )


if __name__ == '__main__':
    network_training()
