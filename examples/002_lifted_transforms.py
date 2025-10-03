# The file is adapted from the Flax library (https://github.com/google/flax).
# The credit should go to the Flax authors.
#
# Copyright 2024 The Flax Authors & 2024 BDP Ecosystem.
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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import brainstate
import braintools


X = np.linspace(0, 1, 100)[:, None]
Y = 0.8 * X ** 2 + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
    while True:
        idx = np.random.choice(len(X), size=batch_size)
        yield X[idx], Y[idx]


class Linear(brainstate.nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.w = brainstate.ParamState(brainstate.random.rand(din, dout))
        self.b = brainstate.ParamState(jnp.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w.value + self.b.value


class Count(brainstate.State):
    pass


class MLP(brainstate.nn.Module):
    def __init__(self, din, dhidden, dout):
        super().__init__()
        self.count = Count(jnp.array(0))
        self.linear1 = Linear(din, dhidden)
        self.linear2 = Linear(dhidden, dout)

    def __call__(self, x):
        self.count.value += 1
        x = self.linear1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x


model = MLP(din=1, dhidden=32, dout=1)
optimizer = braintools.optim.SGD(1e-3)
optimizer.register_trainable_weights(model.states(brainstate.ParamState))


@brainstate.transform.jit
def train_step(batch):
    x, y = batch

    def loss_fn():
        return jnp.mean((y - model(x)) ** 2)

    grads = brainstate.transform.grad(loss_fn, optimizer.param_states.to_dict())()
    optimizer.update(grads)


@brainstate.transform.jit
def test_step(batch):
    x, y = batch
    y_pred = model(x)
    loss = jnp.mean((y - y_pred) ** 2)
    return {'loss': loss}


total_steps = 10_000
for step, batch in enumerate(dataset(32)):
    train_step(batch)

    if step % 1000 == 0:
        logs = test_step((X, Y))
        print(f"step: {step}, loss: {logs['loss']}")

    if step >= total_steps - 1:
        break

print('times called:', model.count.value)

y_pred = model(X)

plt.scatter(X, Y, color='blue')
plt.plot(X, y_pred, color='black')
plt.show()
