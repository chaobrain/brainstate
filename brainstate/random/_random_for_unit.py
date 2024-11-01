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

from __future__ import annotations

import brainunit as bu
import jax
import jax.random as jr

from brainstate.typing import ArrayLike, Size, DTypeLike


def uniform_for_unit(
    key,
    shape: Size = (),
    dtype: DTypeLike = float,
    minval: ArrayLike = 0.,
    maxval: ArrayLike = 1.
) -> jax.Array | bu.Quantity:
    if isinstance(minval, bu.Quantity) and isinstance(maxval, bu.Quantity):
        maxval = maxval.in_unit(minval.unit)
        return bu.Quantity(jr.uniform(key, shape, dtype, minval.mantissa, maxval.mantissa), unit=minval.unit)
    elif isinstance(minval, bu.Quantity):
        assert minval.is_unitless, f'minval must be unitless when maxval is not a Quantity, got {minval}'
        minval = minval.mantissa
    elif isinstance(maxval, bu.Quantity):
        assert maxval.is_unitless, f'maxval must be unitless when minval is not a Quantity, got {maxval}'
        maxval = maxval.mantissa
    return jr.uniform(key, shape, dtype, minval, maxval)


def permutation_for_unit(
    key,
    x: int | ArrayLike,
    axis: int = 0,
    independent: bool = False
) -> jax.Array | bu.Quantity:
    if isinstance(x, bu.Quantity):
        return bu.Quantity(jr.permutation(key, x.mantissa, axis, independent=independent), unit=x.unit)
    return jr.permutation(key, x, axis, independent=independent)
