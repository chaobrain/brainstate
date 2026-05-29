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

"""Static type-assertion checks for brainstate's public typing surface.

This module contains no runtime logic. Everything lives under ``TYPE_CHECKING``
so it imposes zero import/runtime cost, while mypy (which treats
``TYPE_CHECKING`` as ``True``) fully type-checks every assertion below. It acts
as a regression guard: if a public type alias or signature silently changes,
mypy fails here.

Extend this file as each layer is typed (add ``assert_type`` checks on that
layer's public APIs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brainstate.typing import Axes, Shape, Size

    # --- typing.py alias guards (Phase 0) ---
    # Shape is Sequence[int]; a tuple of ints must be assignable.
    _shape: Shape = (1, 2, 3)
    # Size accepts a bare int or a sequence of ints.
    _size_scalar: Size = 5
    _size_seq: Size = [1, 2, 3]
    # Axes accepts an int or a sequence of ints.
    _axis_scalar: Axes = 0
    _axis_seq: Axes = (0, 1)

    # NOTE: As layers are typed, add real public-API checks here, e.g.:
    #     from typing import assert_type
    #     import jax
    #     import brainstate
    #     assert_type(brainstate.random.rand(3), jax.Array)
