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

"""flax.linen adapter + layer mappings.

linen is *functional*: a model is a ``(module, variables)`` pair rather than a single object
holding its own weights. The adapter wraps both into a lightweight :class:`_LinenNode` so the
generic engine can traverse it uniformly. Each node carries the slice of the variables tree that
belongs to its module, so leaf mappings simply read/write ``node.variables['params'][...]``.

linen shares brainstate's tensor layout (Linear ``(in, out)``, Conv ``(*k, in, out)`` NHWC), so
the weight maths is identical to the nnx adapter; only the param-tree plumbing differs. linen
``nn.Sequential`` names its children ``layers_0``, ``layers_1``, ... by position, which is how
children are sliced on import and re-assembled on export.
"""

from __future__ import annotations

import brainunit as u

from .. import _common as C
from .._common import FrameworkAdapter, lazy_import
from .._errors import ConversionError
from .._registry import (LayerMapping, register_layer_mapping,
                         register_unsupported_foreign)

linen = lazy_import('flax.linen')
_flax_core = lazy_import('flax.core')
import brainstate.nn as bnn  # noqa: E402

nn = linen


class _LinenNode:
    """A linen ``module`` paired with the slice of the variables tree it owns.

    Parameters
    ----------
    module : flax.linen.Module
        The (unbound) linen module describing this node's architecture.
    variables : dict
        A plain (unfrozen) variables mapping for *this* module, e.g.
        ``{'params': {...}, 'batch_stats': {...}}``. Collections absent for the module are simply
        omitted.
    """

    __slots__ = ('module', 'variables')

    def __init__(self, module, variables):
        self.module = module
        self.variables = variables


def _params(node: _LinenNode) -> dict:
    return node.variables.get('params', {}) or {}


def _split4(arr, axis=-1):
    return list(u.math.split(arr, 4, axis=axis))


# ---------------------------------------------------------------------------
# Linear (nn.Dense)
# ---------------------------------------------------------------------------

def _linear_to_bst(node, ctx):
    p = _params(node)
    w = p['kernel']
    b = p.get('bias')
    layer = C.build_linear(w.shape[0], w.shape[1], b is not None)
    C.bst_set_linear(layer, w, b)
    return layer


def _linear_to_foreign(layer, ctx):
    w, b = C.bst_get_linear(layer)
    module = nn.Dense(features=w.shape[1], use_bias=b is not None)
    params = {'kernel': w}
    if b is not None:
        params['bias'] = b
    return _LinenNode(module, {'params': params})


# ---------------------------------------------------------------------------
# Embedding (nn.Embed)
# ---------------------------------------------------------------------------

def _embed_to_bst(node, ctx):
    table = _params(node)['embedding']
    layer = C.build_embedding(table.shape[0], table.shape[1])
    C.bst_set_embedding(layer, table)
    return layer


def _embed_to_foreign(layer, ctx):
    table = C.bst_get_embedding(layer)
    module = nn.Embed(num_embeddings=table.shape[0], features=table.shape[1])
    return _LinenNode(module, {'params': {'embedding': table}})


# ---------------------------------------------------------------------------
# LayerNorm / RMSNorm / GroupNorm  (1-D affine vectors -> identity)
# ---------------------------------------------------------------------------

def _norm_num_from_params(*arrays):
    """Extract the feature count from the first non-None affine parameter."""
    for a in arrays:
        if a is not None:
            return int(a.shape[0])
    return None


def _layernorm_to_bst(node, ctx):
    p = _params(node)
    scale = p.get('scale')
    bias = p.get('bias')
    num = _norm_num_from_params(scale, bias)
    if num is None:
        raise ConversionError(
            "Cannot determine feature count for linen LayerNorm with no affine parameters "
            "and no explicit feature size."
        )
    layer = C.build_layernorm((num,), scale is not None, bias is not None,
                              float(node.module.epsilon))
    C.bst_set_norm(layer, 'weight', scale, bias)
    return layer


def _layernorm_to_foreign(layer, ctx):
    scale, offset = C.bst_get_norm(layer, 'weight', has_offset=True)
    module = nn.LayerNorm(epsilon=float(layer.epsilon),
                          use_scale=scale is not None, use_bias=offset is not None)
    params = {}
    if scale is not None:
        params['scale'] = scale
    if offset is not None:
        params['bias'] = offset
    return _LinenNode(module, {'params': params})


def _rmsnorm_to_bst(node, ctx):
    scale = _params(node).get('scale')
    if scale is None:
        raise ConversionError(
            "Cannot determine feature count for linen RMSNorm with no scale parameter."
        )
    layer = C.build_rmsnorm((scale.shape[0],), True, float(node.module.epsilon))
    C.bst_set_norm(layer, 'scale', scale, None)
    return layer


def _rmsnorm_to_foreign(layer, ctx):
    scale, _ = C.bst_get_norm(layer, 'scale', has_offset=False)
    has_scale = scale is not None
    module = nn.RMSNorm(epsilon=float(layer.epsilon), use_scale=has_scale)
    params = {}
    if scale is not None:
        params['scale'] = scale
    return _LinenNode(module, {'params': params})


def _groupnorm_to_bst(node, ctx):
    p = _params(node)
    scale = p.get('scale')
    bias = p.get('bias')
    num = _norm_num_from_params(scale, bias)
    if num is None:
        raise ConversionError(
            "Cannot determine feature count for linen GroupNorm with no affine parameters."
        )
    layer = C.build_groupnorm((num,), int(node.module.num_groups),
                              scale is not None, bias is not None, float(node.module.epsilon))
    C.bst_set_norm(layer, 'weight', scale, bias)
    return layer


def _groupnorm_to_foreign(layer, ctx):
    scale, offset = C.bst_get_norm(layer, 'weight', has_offset=True)
    num = int(layer.in_size[-1])
    module = nn.GroupNorm(num_groups=int(layer.num_groups), epsilon=float(layer.epsilon),
                          use_scale=scale is not None, use_bias=offset is not None)
    params = {}
    if scale is not None:
        params['scale'] = scale
    if offset is not None:
        params['bias'] = offset
    return _LinenNode(module, {'params': params})


# ---------------------------------------------------------------------------
# Dropout (stateless; prob[keep] <-> rate[drop])
# ---------------------------------------------------------------------------

def _dropout_to_bst(node, ctx):
    return C.build_dropout(1.0 - float(node.module.rate))


def _dropout_to_foreign(layer, ctx):
    # No parameters. ``deterministic=True`` is baked in so the exported module (and any
    # ``nn.Sequential`` containing it, which forwards call kwargs to every child) is directly
    # applicable and output-equivalent to brainstate's eval-mode Dropout. Cross-framework RNG
    # streams cannot be matched, so equivalence for Dropout is only defined in eval mode.
    return _LinenNode(nn.Dropout(rate=1.0 - float(layer.prob), deterministic=True), {})


# ---------------------------------------------------------------------------
# Conv  (bst (*k,in,out) NHWC <-> linen (*k,in,out) NHWC: kernel identity, bias reshape)
# ---------------------------------------------------------------------------

def _conv_to_bst(node, ctx):
    m = node.module
    p = _params(node)
    w = p['kernel']                       # (*k, in//g, out)
    kernel_size = tuple(w.shape[:-2])
    nd = len(kernel_size)
    out_channels = w.shape[-1]
    if any(d != 1 for d in _as_tuple(getattr(m, 'input_dilation', 1) or 1, nd)):
        raise ConversionError(
            "linen Conv with `input_dilation` != 1 (transposed convolution) is not supported "
            "by this converter."
        )
    in_size = ctx.require_size('Conv')
    b = p.get('bias')
    layer = C.build_conv(
        nd, in_size, out_channels, kernel_size,
        stride=getattr(m, 'strides', 1) or 1,
        padding=getattr(m, 'padding', 'SAME'),
        rhs_dilation=getattr(m, 'kernel_dilation', 1) or 1,
        groups=int(getattr(m, 'feature_group_count', 1)),
        has_bias=b is not None,
    )
    bias = u.math.reshape(b, (1,) * nd + (out_channels,)) if b is not None else None
    C.bst_set_linear(layer, w, bias)
    return layer


def _conv_to_foreign(layer, ctx):
    w, b = C.bst_get_linear(layer)
    nd = len(layer.kernel_size)
    module = nn.Conv(
        features=layer.out_channels,
        kernel_size=tuple(layer.kernel_size),
        strides=tuple(layer.stride),
        padding=layer.padding,
        input_dilation=tuple(layer.lhs_dilation),
        kernel_dilation=tuple(layer.rhs_dilation),
        feature_group_count=layer.groups,
        use_bias=b is not None,
    )
    params = {'kernel': w}
    if b is not None:
        params['bias'] = u.math.reshape(b, (b.shape[-1],))
    return _LinenNode(module, {'params': params})


# ---------------------------------------------------------------------------
# BatchNorm  (bst (1,...,1,C) <-> linen (C,); running stats in batch_stats collection)
# ---------------------------------------------------------------------------

def _batchnorm_to_bst(node, ctx):
    p = _params(node)
    bs = node.variables.get('batch_stats', {})
    scale = p.get('scale')
    bias = p.get('bias')
    mean = bs['mean']
    var = bs['var']
    in_size = ctx.require_size('BatchNorm')
    nd = len(in_size) - 1
    affine = scale is not None
    layer = C.build_batchnorm(nd, in_size, float(node.module.epsilon),
                              float(node.module.momentum), affine)

    def _r(v):
        return u.math.reshape(v, (1,) * nd + (v.shape[-1],))

    C.bst_set_batchnorm(
        layer,
        _r(scale) if scale is not None else None,
        _r(bias) if bias is not None else None,
        _r(mean), _r(var),
    )
    return layer


def _batchnorm_to_foreign(layer, ctx):
    scale, offset, mean, var = C.bst_get_batchnorm(layer)

    def _f(v):
        return u.math.reshape(v, (v.shape[-1],))

    module = nn.BatchNorm(
        use_running_average=True,
        momentum=float(layer.momentum),
        epsilon=float(layer.epsilon),
        use_scale=scale is not None,
        use_bias=offset is not None,
    )
    params = {}
    if scale is not None:
        params['scale'] = _f(scale)
    if offset is not None:
        params['bias'] = _f(offset)
    return _LinenNode(module, {'params': params,
                               'batch_stats': {'mean': _f(mean), 'var': _f(var)}})


# ---------------------------------------------------------------------------
# LSTMCell  (fused (in+h,4h) [i,g,f,o] + forget+1 fold  <->  8 Denses [i,f,g,o])
# ---------------------------------------------------------------------------

def _lstm_to_bst(node, ctx):
    p = _params(node)
    ii, if_, ig, io = (p['ii']['kernel'], p['if']['kernel'],
                       p['ig']['kernel'], p['io']['kernel'])
    hi, hf, hg, ho = (p['hi']['kernel'], p['hf']['kernel'],
                      p['hg']['kernel'], p['ho']['kernel'])
    bi, bf, bg, bo = (p['hi']['bias'], p['hf']['bias'], p['hg']['bias'], p['ho']['bias'])
    num_in = ii.shape[0]
    h = ii.shape[1]
    Wi = u.math.concatenate([ii, ig, if_, io], axis=-1)   # back to brainstate order i,g,f,o
    Wh = u.math.concatenate([hi, hg, hf, ho], axis=-1)
    W = u.math.concatenate([Wi, Wh], axis=0)
    bias = u.math.concatenate([bi, bg, bf - 1.0, bo], axis=-1)
    layer = C.build_lstm(num_in, h)
    C.bst_set_lstm(layer, W, bias)
    return layer


def _lstm_to_foreign(layer, ctx):
    W, bias = C.bst_get_lstm(layer)
    num_in = W.shape[0] - layer.num_out
    h = layer.num_out
    Wi, Wh = W[:num_in], W[num_in:]
    ii, ig, if_, io = _split4(Wi)            # brainstate order i,g,f,o
    hi, hg, hf, ho = _split4(Wh)
    bi, bg, bf, bo = _split4(bias)
    module = nn.LSTMCell(features=h)
    params = {
        'ii': {'kernel': ii}, 'if': {'kernel': if_},
        'ig': {'kernel': ig}, 'io': {'kernel': io},
        'hi': {'kernel': hi, 'bias': bi}, 'hf': {'kernel': hf, 'bias': bf + 1.0},
        'hg': {'kernel': hg, 'bias': bg}, 'ho': {'kernel': ho, 'bias': bo},
    }
    return _LinenNode(module, {'params': params})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_tuple(v, nd):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v,) * nd


# ---------------------------------------------------------------------------
# Adapter + registration
# ---------------------------------------------------------------------------

class LinenAdapter(FrameworkAdapter):
    """Structural plumbing for the functional flax.linen framework.

    A ``node`` here is a :class:`_LinenNode` bundling a module with the slice of the variables
    tree it owns. ``wrap`` builds the root node from a user ``(module, variables)`` pair; the
    per-leaf mappings return ``_LinenNode`` objects on export; ``finalize`` unwraps the root back
    into a ``(module, FrozenDict)`` pair.
    """

    name = 'linen'

    def wrap(self, module, variables):
        """Bundle a linen ``module`` and its ``variables`` into a traversable node."""
        return _LinenNode(module, _flax_core.unfreeze(variables))

    def finalize(self, node):
        """Return the ``(module, FrozenDict)`` pair represented by an export ``node``."""
        return node.module, _flax_core.freeze(node.variables)

    def is_sequential(self, node) -> bool:
        return isinstance(node.module, nn.Sequential)

    def iter_children(self, node):
        children = []
        for i, layer in enumerate(node.module.layers):
            key = f'layers_{i}'
            sub = {}
            for coll, tree in node.variables.items():
                if isinstance(tree, dict) and key in tree:
                    sub[coll] = tree[key]
            children.append((i, _LinenNode(layer, sub)))
        return children

    def has_child_modules(self, node) -> bool:
        # A leaf's param collection maps names -> arrays; a container maps names -> sub-trees.
        return any(isinstance(v, dict) for v in _params(node).values())

    def layer_type(self, node) -> type:
        return type(node.module)

    def build_sequential(self, children: list):
        layers = [c.module for c in children]
        merged: dict = {}
        for i, c in enumerate(children):
            key = f'layers_{i}'
            for coll, tree in c.variables.items():
                merged.setdefault(coll, {})[key] = tree
        return _LinenNode(nn.Sequential(layers), merged)


def _register():
    pairs = [
        (bnn.Linear, nn.Dense, _linear_to_bst, _linear_to_foreign),
        (bnn.Embedding, nn.Embed, _embed_to_bst, _embed_to_foreign),
        (bnn.LayerNorm, nn.LayerNorm, _layernorm_to_bst, _layernorm_to_foreign),
        (bnn.RMSNorm, nn.RMSNorm, _rmsnorm_to_bst, _rmsnorm_to_foreign),
        (bnn.GroupNorm, nn.GroupNorm, _groupnorm_to_bst, _groupnorm_to_foreign),
        (bnn.Dropout, nn.Dropout, _dropout_to_bst, _dropout_to_foreign),
        (bnn.Conv1d, nn.Conv, _conv_to_bst, _conv_to_foreign),
        (bnn.Conv2d, nn.Conv, _conv_to_bst, _conv_to_foreign),
        (bnn.Conv3d, nn.Conv, _conv_to_bst, _conv_to_foreign),
        (bnn.BatchNorm1d, nn.BatchNorm, _batchnorm_to_bst, _batchnorm_to_foreign),
        (bnn.BatchNorm2d, nn.BatchNorm, _batchnorm_to_bst, _batchnorm_to_foreign),
        (bnn.BatchNorm3d, nn.BatchNorm, _batchnorm_to_bst, _batchnorm_to_foreign),
        (bnn.LSTMCell, nn.LSTMCell, _lstm_to_bst, _lstm_to_foreign),
    ]
    for bst_type, foreign_type, to_bst, to_foreign in pairs:
        register_layer_mapping(LayerMapping(bst_type, 'linen', foreign_type, to_bst, to_foreign))

    register_unsupported_foreign(
        'linen', nn.GRUCell,
        "GRUCell conversion is unsupported: brainstate's GRU uses the Cho-2014 variant (reset "
        "applied before the hidden matmul, `Wh([x, r*h])`) while flax.linen uses the cuDNN "
        "variant (reset applied after, `r * (W_hn h + b_hn)`). These are mathematically distinct "
        "and cannot be matched by transferring weights."
    )
    C.register_bst_unsupported()


_register()
