# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import braintools.optim
import jax.nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import torchvision

import brainstate.nn

# Define transforms for images and targets
root = r"D:\data\MNIST"
train_data = torchvision.datasets.MNIST(root, train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root, train=False, download=True, transform=torchvision.transforms.ToTensor())


class MLP(brainstate.nn.Module):
    def __init__(self, n_in, n_hiddens, n_out):
        super().__init__()
        self.layers = []
        for i, n_hidden in enumerate(n_hiddens):
            self.layers.append(brainstate.nn.Linear(n_in, n_hidden))
            self.layers.append(jax.nn.mish)
            n_in = n_hidden
        self.layers.append(brainstate.nn.Linear(n_in, n_out))

    def update(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train(epochs, batch_size, lr, num_layers, ad_type: str):
    # Initialize model
    mlp = MLP(28 * 28, [256] * num_layers, 10)
    trainable_weights = mlp.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=lr)
    opt.register_trainable_weights(trainable_weights)

    def acc_fn(predicts, targets):
        predicted_class = jnp.argmax(predicts, axis=1)
        return jnp.mean(predicted_class == targets)

    def loss_fn(xs, ys):
        predicts = mlp(xs)
        loss = braintools.metric.softmax_cross_entropy_with_integer_labels(predicts, ys).mean()
        return loss, acc_fn(predicts, ys)

    def asarray(batch):
        xs, ys = batch
        xs = jnp.asarray(xs)
        ys = jnp.asarray(ys)
        return xs.reshape(xs.shape[0], -1), ys

    @brainstate.transform.jit
    def train(batch):
        xs, ys = batch
        if ad_type == 'jvp':
            grads, loss, acc = brainstate.transform.fwd_grad(
                loss_fn, grad_states=trainable_weights, return_value=True, has_aux=True, tangent_size=128,
            )(xs, ys)
        elif ad_type == 'vjp':
            grads, loss, acc = brainstate.transform.grad(
                loss_fn, grad_states=trainable_weights, return_value=True, has_aux=True,
            )(xs, ys)
        elif ad_type == 'sofo':
            loss_f = lambda p: braintools.metric.softmax_cross_entropy_with_integer_labels(p, ys).mean()
            grads, predicts = brainstate.transform.sofo_grad(
                mlp, loss_f, grad_states=trainable_weights, return_value=True, loss='ce')(xs)
            loss = loss_f(predicts)
            acc = acc_fn(predicts, ys)
        else:
            raise ValueError
        opt.update(grads)
        return loss, acc

    @brainstate.transform.jit
    def accuracy(batch):
        inputs, targets = batch
        predicted_class = jnp.argmax(mlp(inputs), axis=1)
        return jnp.mean(predicted_class == targets)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    iter_cnt = 0
    hist_loss = []
    hist_train_acc = []
    for _ in range(epochs):
        for batch_id, batch in enumerate(train_loader):
            loss_value, acc_value = train(asarray(batch))
            hist_loss.append(loss_value)
            hist_train_acc.append(acc_value)
            iter_cnt += 1
            if batch_id % 100 == 0:
                print({'loss': loss_value, 'iter': iter_cnt, 'acc_value': acc_value})

        train_acc = 100 * sum(accuracy(asarray(train_batch)) for train_batch in train_loader) / len(train_loader)
        test_acc = 100 * sum(accuracy(asarray(test_batch)) for test_batch in test_loader) / len(test_loader)
        print(f'Epoch {_}: train_acc={train_acc:.2f}, test_acc={test_acc:.2f}')

    fig, gs = braintools.visualize.get_figure(1, 2, 4, 5)
    fig.add_subplot(gs[0, 0])
    plt.plot(np.asarray(hist_loss))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.add_subplot(gs[0, 1])
    plt.plot(np.asarray(hist_train_acc))
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.savefig(f'{ad_type}.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden layers for MLP')
    parser.add_argument('--ad_type', type=str, default='jvp', choices=['jvp', 'vjp', 'sofo'])
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.lr, args.num_layers, args.ad_type)
