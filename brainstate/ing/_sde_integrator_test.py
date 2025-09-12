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

import brainstate


def test_sde_euler_shape_and_variance():
    # d y = sigma dW (pure diffusion)
    sigma = 2.0

    def df(y, t):
        return 0.0

    def dg(y, t):
        return sigma

    y0 = 0.0
    dt = 0.2
    with brainstate.environ.context(dt=dt):
        y1 = brainstate.ing.sde_euler_step(df, dg, y0, 0.0)
        y2 = brainstate.ing.sde_euler_step(df, dg, y0, 0.0)

    # shapes (scalar array)
    assert np.shape(y1) == ()
    # stochastic - likely different
    assert not np.isclose(y1, y2)


def test_sde_milstein_basic():
    # d y = a y dt + b y dW (geometric brownian motion)
    a, b = 0.1, 0.3

    def df(y, t):
        return a * y

    def dg(y, t):
        return b * y

    y0 = 1.0
    dt = 0.05
    with brainstate.environ.context(dt=dt):
        y1 = brainstate.ing.sde_milstein_step(df, dg, y0, 0.0)

    assert np.isfinite(y1)
