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


# %%
import typing as tp

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from datasets import load_dataset
import brainstate
import braintools


np.random.seed(42)
latent_size = 32
image_shape: tp.Sequence[int] = (28, 28)
steps_per_epoch: int = 200
batch_size: int = 64
epochs: int = 20

dataset = load_dataset('mnist')
X_train = np.array(np.stack(dataset['train']['image']), dtype=np.uint8)
X_test = np.array(np.stack(dataset['test']['image']), dtype=np.uint8)
# Now binarize data
X_train = (X_train > 0).astype(jnp.float32)
X_test = (X_test > 0).astype(jnp.float32)

print('X_train:', X_train.shape, X_train.dtype)
print('X_test:', X_test.shape, X_test.dtype)


class Loss(brainstate.State):
    pass


# %%
class Encoder(brainstate.nn.Module):
    def __init__(self, din: int, dmid: int, dout: int):
        super().__init__()
        self.linear1 = brainstate.nn.Linear(din, dmid)
        self.linear_mean = brainstate.nn.Linear(dmid, dout)
        self.linear_std = brainstate.nn.Linear(dmid, dout)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.linear1(x)
        x = jax.nn.relu(x)

        mean = self.linear_mean(x)
        std = jnp.exp(self.linear_std(x))

        loss = jnp.mean(0.5 * jnp.mean(-jnp.log(std ** 2) - 1.0 + std ** 2 + mean ** 2, axis=-1))
        self.kl_loss = Loss(loss)
        z = mean + std * brainstate.random.normal(size=mean.shape)
        return z


class Decoder(brainstate.nn.Module):
    def __init__(self, din: int, dmid: int, dout: int):
        super().__init__()
        self.linear1 = brainstate.nn.Linear(din, dmid)
        self.linear2 = brainstate.nn.Linear(dmid, dout)

    def __call__(self, z: jax.Array) -> jax.Array:
        z = self.linear1(z)
        z = jax.nn.relu(z)
        logits = self.linear2(z)
        return logits


class VAE(brainstate.nn.Module):
    def __init__(
        self,
        din: int,
        hidden_size: int,
        latent_size: int,
        output_shape: tp.Sequence[int],
    ):
        super().__init__()
        self.output_shape = output_shape
        self.encoder = Encoder(din, hidden_size, latent_size)
        self.decoder = Decoder(
            latent_size, hidden_size, int(np.prod(output_shape))
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        z = self.encoder(x)
        logits = self.decoder(z)
        logits = jnp.reshape(logits, (-1, *self.output_shape))
        return logits

    def generate(self, z):
        logits = self.decoder(z)
        logits = jnp.reshape(logits, (-1, *self.output_shape))
        return jax.nn.sigmoid(logits)


model = VAE(
    din=int(np.prod(image_shape)),
    hidden_size=256,
    latent_size=latent_size,
    output_shape=image_shape,
)

optimizer = braintools.optim.Adam(1e-3)
optimizer.register_trainable_weights(model.states(brainstate.ParamState))


# %%
@brainstate.compile.jit
def train_step(x: jax.Array):
    def loss_fn():
        logits = model(x)
        losses = brainstate.graph.treefy_states(model, Loss)
        kl_loss = sum(jax.tree_util.tree_leaves(losses), 0.0)
        reconstruction_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, x))
        loss = reconstruction_loss + 0.1 * kl_loss
        return loss

    grads, loss = brainstate.augment.grad(loss_fn, optimizer.param_states.to_dict(), return_value=True)()
    optimizer.update(grads)
    return loss


@brainstate.compile.jit
def forward(x: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(model(x))


@brainstate.compile.jit
def sample(z: jax.Array) -> jax.Array:
    return model.generate(z)


# %%

for epoch in range(epochs):
    losses = []
    for step in range(steps_per_epoch):
        idxs = np.random.randint(0, len(X_train), size=(batch_size,))
        x_batch = X_train[idxs]

        loss = train_step(x_batch)
        losses.append(np.asarray(loss))

    print(f'Epoch {epoch} loss: {np.mean(losses)}')

# exit()
# %%
# get random samples
idxs = np.random.randint(0, len(X_test), size=(5,))
x_sample = X_test[idxs]

# get predictions
y_pred = forward(x_sample)

# plot reconstruction
figure = plt.figure(figsize=(3 * 5, 3 * 2))
plt.title('Reconstruction Samples')
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_sample[i], cmap='gray')
    plt.subplot(2, 5, 5 + i + 1)
    plt.imshow(y_pred[i], cmap='gray')
    # # tbwriter.add_figure("VAE Example", figure, epochs)

plt.show()

# %%
# plot generative samples
z_samples = np.random.normal(scale=1.5, size=(12, latent_size))
samples = sample(z_samples)

figure = plt.figure(figsize=(3 * 5, 3 * 2))
plt.title('Generative Samples')
for i in range(5):
    plt.subplot(2, 5, 2 * i + 1)
    plt.imshow(samples[i], cmap='gray')
    plt.subplot(2, 5, 2 * i + 2)
    plt.imshow(samples[i + 1], cmap='gray')

plt.show()

# %%
