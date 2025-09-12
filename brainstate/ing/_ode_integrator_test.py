# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


import numpy as np
import jax.numpy as jnp
import brainstate



def test_ode_integrators_scalar_linear():
    # y' = a*y, a=1.0, y0 = 1.0
    a = 1.0
    def f(y, t):
        return a * y

    y0 = 1.0
    dt = 0.1

    with brainstate.environ.context(dt=dt):
        y_euler = brainstate.ing.ode_euler_step(f, y0, 0.0)
        y_rk2 = brainstate.ing.ode_rk2_step(f, y0, 0.0)
        y_rk3 = brainstate.ing.ode_rk3_step(f, y0, 0.0)
        y_rk4 = brainstate.ing.ode_rk4_step(f, y0, 0.0)

    # True solution: y = e^{a dt}
    y_true = np.exp(a * dt)
    assert np.allclose(y_euler, 1 + a * dt)
    assert abs(y_rk2 - y_true) < abs(y_euler - y_true)
    assert abs(y_rk3 - y_true) < abs(y_rk2 - y_true)
    assert abs(y_rk4 - y_true) < abs(y_rk3 - y_true)


def test_ode_integrators_vector_tree():
    # y' = A y for vector y
    A = jnp.array([[0.0, 1.0], [-1.0, 0.0]])  # rotation system
    def f(y, t):
        return A @ y

    y0 = jnp.array([1.0, 0.0])
    dt = 0.01

    with brainstate.environ.context(dt=dt):
        y1 = brainstate.ing.ode_rk4_step(f, y0, 0.0)

    assert y1.shape == (2,)

