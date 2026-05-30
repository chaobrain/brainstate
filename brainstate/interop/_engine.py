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

"""The generic recursive conversion engine.

``to_bst`` walks a foreign model and builds the brainstate equivalent; ``to_foreign`` does the
reverse. A node is resolved *leaf-first*: a registered leaf type is converted directly (even if
it internally holds sub-modules, e.g. an LSTM cell). Only unregistered nodes are considered for
container treatment, where a ``Sequential`` recurses and any other multi-module node raises an
informative ``UnsupportedStructureError``.
"""

from __future__ import annotations

import brainstate.nn as bnn

from ._common import Context, FrameworkAdapter
from ._errors import (UnmappedLayerError, UnsupportedLayerError,
                      UnsupportedStructureError)
from ._registry import (lookup_export, lookup_import, unsupported_bst_reason,
                        unsupported_foreign_reason)

__all__ = ['to_bst', 'to_foreign']


def to_bst(node, adapter: FrameworkAdapter, ctx: Context):
    """Convert a foreign ``node`` (leaf or ``Sequential``) into a brainstate module."""
    ftype = adapter.layer_type(node)
    mapping = lookup_import(adapter.name, ftype)
    if mapping is not None:
        bst_layer = mapping.to_bst(node, ctx)
        out_size = getattr(bst_layer, 'out_size', None)
        if out_size is not None:
            ctx.cur_size = tuple(out_size)
        return bst_layer

    reason = unsupported_foreign_reason(adapter.name, ftype)
    if reason is not None:
        raise UnsupportedLayerError(reason)

    if adapter.is_sequential(node):
        children = [to_bst(child, adapter, ctx) for _, child in adapter.iter_children(node)]
        if not children:
            raise UnsupportedStructureError("Cannot convert an empty Sequential container.")
        return bnn.Sequential(*children)

    if adapter.has_child_modules(node):
        raise UnsupportedStructureError(
            f"Cannot convert `{ftype.__name__}`: only single layers and Sequential stacks of "
            f"registered layers are supported. A container with custom forward logic "
            f"(skips/branching) cannot be reconstructed."
        )
    raise UnmappedLayerError(ftype, adapter.name)


def to_foreign(node, adapter: FrameworkAdapter, ctx: Context):
    """Convert a brainstate ``node`` (leaf or ``Sequential``) into a foreign module."""
    btype = type(node)
    mapping = lookup_export(btype, adapter.name)
    if mapping is not None:
        return mapping.to_foreign(node, ctx)

    reason = unsupported_bst_reason(btype, adapter.name)
    if reason is not None:
        raise UnsupportedLayerError(reason)

    if isinstance(node, bnn.Sequential):
        children = [to_foreign(child, adapter, ctx) for child in node.layers]
        if not children:
            raise UnsupportedStructureError("Cannot convert an empty Sequential container.")
        return adapter.build_sequential(children)

    if _bst_has_child_modules(node):
        raise UnsupportedStructureError(
            f"Cannot convert `{btype.__name__}`: only single layers and Sequential stacks of "
            f"registered layers are supported. A module with custom forward logic cannot be "
            f"reconstructed."
        )
    raise UnmappedLayerError(btype, adapter.name)


def _bst_has_child_modules(node) -> bool:
    for v in vars(node).values():
        if isinstance(v, bnn.Module):
            return True
        if isinstance(v, (list, tuple)) and any(isinstance(x, bnn.Module) for x in v):
            return True
        if isinstance(v, dict) and any(isinstance(x, bnn.Module) for x in v.values()):
            return True
    return False
