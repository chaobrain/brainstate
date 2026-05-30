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

"""Error types raised by :mod:`brainstate.interop`."""

from __future__ import annotations

from brainstate._error import BrainStateError

__all__ = [
    'InteropError',
    'MissingDependencyError',
    'UnmappedLayerError',
    'UnsupportedLayerError',
    'UnsupportedStructureError',
    'MissingShapeError',
    'ConversionError',
]


class InteropError(BrainStateError):
    """Base class for all :mod:`brainstate.interop` errors."""


class MissingDependencyError(InteropError):
    """Raised when an optional framework (``flax`` / ``equinox``) is not installed.

    Parameters
    ----------
    package : str
        The name of the missing package.
    install_hint : str
        A ``pip install`` command the user can run.
    """

    def __init__(self, package: str, install_hint: str):
        self.package = package
        super().__init__(
            f"Converting to/from this framework requires the optional dependency "
            f"'{package}', which is not installed. Install it with: {install_hint}"
        )


class UnmappedLayerError(InteropError):
    """Raised when no conversion mapping is registered for a *leaf* layer type."""

    def __init__(self, layer_type: type, framework: str):
        self.layer_type = layer_type
        self.framework = framework
        name = getattr(layer_type, '__name__', repr(layer_type))
        super().__init__(
            f"No interop mapping is registered for layer type `{name}` "
            f"(framework: {framework}). Register one with "
            f"`brainstate.interop.register_layer_mapping(...)`, or remove the layer "
            f"from the model before converting."
        )


class UnsupportedLayerError(InteropError):
    """Raised for a known layer type that is deliberately unsupported in this version.

    The message explains *why* (e.g. a mathematical variant mismatch) so the user is not
    left guessing.
    """


class UnsupportedStructureError(InteropError):
    """Raised when a container's forward logic cannot be reconstructed.

    Only single layers and ``Sequential`` stacks of registered layers convert; a module
    with custom ``__call__`` (skips, branching, attention) raises this error.
    """


class MissingShapeError(InteropError):
    """Raised when importing a spatial layer (``Conv``/``BatchNorm``) without a sample input.

    brainstate's ``Conv`` and spatial ``BatchNorm`` carry a concrete ``in_size`` (including
    spatial dimensions) that a framework-agnostic foreign layer does not encode, so a
    ``sample_input`` is required to materialize them.
    """


class ConversionError(InteropError):
    """Raised when a weight transfer fails (shape/dtype/unit mismatch).

    The message includes the offending role and the shapes involved.
    """
