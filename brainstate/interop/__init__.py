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

"""Convert standard-layer models between ``brainstate.nn`` and ``flax.nnx`` / ``flax.linen`` /
``equinox``.

The public functions construct an architecturally-equivalent model in the target framework and
transfer weights, guaranteeing numerical output-equivalence for everything that converts (single
layers and ``Sequential`` stacks of registered layers). Unsupported layers / structures raise
informative errors rather than producing a silently-wrong model.

See Also
--------
register_layer_mapping : add a conversion for a custom layer type.
supported_layers : list the layers convertible for each framework.
"""

from ._api import (
    from_nnx, to_nnx,
    from_linen, to_linen,
    from_equinox, to_equinox,
)
from ._errors import (
    InteropError,
    MissingDependencyError,
    UnmappedLayerError,
    UnsupportedLayerError,
    UnsupportedStructureError,
    MissingShapeError,
    ConversionError,
)
from ._registry import LayerMapping, register_layer_mapping, supported_layers

__all__ = [
    'from_nnx', 'to_nnx',
    'from_linen', 'to_linen',
    'from_equinox', 'to_equinox',
    'register_layer_mapping', 'supported_layers', 'LayerMapping',
    'InteropError', 'MissingDependencyError', 'UnmappedLayerError',
    'UnsupportedLayerError', 'UnsupportedStructureError', 'MissingShapeError',
    'ConversionError',
]
