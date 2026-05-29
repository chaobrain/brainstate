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

import braintools
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import brainstate


batch_size = 512
output_size = 10
input_size = 784
iteration = 5000
learning_rate = 1e-3
tangent_size = 512
damping = 1E-7
momentum = 0.9
layers = [100, ]


def main():
    trainloader = DataLoader(
        MNIST('./data', download=True, train=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True,
    )
    testloader = DataLoader(
        MNIST('./data', download=True, train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False,
    )

    class MLP(brainstate.nn.Module):
        def __init__(self, hiddens, latent, activation=brainstate.nn.relu):
            super().__init__()
            self.hiddens = hiddens
            self.latent = latent
            self.activation = activation

            self.layers = []
            n_in = input_size
            for i, hidden in enumerate(self.hiddens):
                self.layers.append(brainstate.nn.Linear(n_in, hidden, name='fc{}'.format(i)))
                self.layers.append(brainstate.nn.ReLU())
                n_in = hidden
            self.layers.append(brainstate.nn.Linear(n_in, latent, name='out'))

        def update(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = MLP(hiddens=layers, latent=output_size)

    # SOFO ("Second-Order Forward-mode Optimization") owns the forward pass: it samples random
    # tangents, takes forward-mode JVPs through ``model`` and ``loss_fn``, builds the Generalised
    # Gauss-Newton matrix in that subspace, and applies the damped natural-gradient direction
    # through an SGD-style update (momentum / weight decay handled internally).
    loss_fn = lambda pred, targets: optax.softmax_cross_entropy_with_integer_labels(pred, targets).mean()
    opt = braintools.optim.SOFO(
        model, loss_fn, lr=learning_rate, loss='ce',
        tangent_size=tangent_size, damping=damping,
        momentum=momentum, nesterov=True, weight_decay=1e-5,
    )
    opt.register_trainable_weights(model.states(brainstate.ParamState))

    def accuracy(pred, targets):
        target_class = jnp.argmax(targets, axis=1) if targets.ndim == 2 else targets
        predicted_class = jnp.argmax(pred, axis=1)
        return jnp.mean(predicted_class == target_class)

    @brainstate.transform.jit
    def train_step(batch_x, batch_y):
        loss = opt.step(batch_x, batch_y)
        acc = accuracy(model(batch_x), batch_y)
        return loss, acc

    for i in range(iteration):
        losses, acces = [], []
        for j, (xs, ys) in enumerate(trainloader):
            xs = jnp.asarray(xs).reshape(xs.shape[0], -1)
            ys = jnp.asarray(ys)
            train_loss, train_acc = train_step(xs, ys)
            losses.append(train_loss)
            acces.append(train_acc)
            print(f'Epoch {i}: Batch {j}: Train: loss:{float(train_loss):.4f}, acc:{float(train_acc):.3f}')
        train_loss = np.mean(np.asarray(losses))
        train_acc = np.mean(np.asarray(acces))
        logging.info(f'Epoch {i}: Train: loss:{float(train_loss):.4f}, acc:{float(train_acc):.3f}')


if __name__ == '__main__':
    main()
