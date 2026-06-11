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

"""flax.nnx adapter + layer mappings.

nnx is reference-semantic like brainstate and shares its tensor layout (Linear ``(in, out)``,
Conv ``(*k, in, out)`` NHWC), so most weight transfers are the identity; the differences are
bias shape (conv/batch-norm) and the fused-vs-split LSTM kernel + gate order.
"""

from __future__ import annotations

import brainunit as u

from .. import _common as C
from .._common import FrameworkAdapter, lazy_import, new_key
from .._errors import ConversionError
from .._registry import (LayerMapping, register_layer_mapping,
                         register_unsupported_bst, register_unsupported_foreign)

nnx = lazy_import('flax.nnx')
import brainstate.nn as bnn  # noqa: E402

# brainstate LSTM stores gates [i, g, f, o]; nnx/torch use [i, f, g, o]. Self-inverse swap.


def _get(var):
    """Read an nnx Variable's array (new ``[...]`` API, falling back to ``.value``)."""
    try:
        return var[...]
    except Exception:  # pragma: no cover - version fallback
        return var.value


def _set(var, arr):
    """Write an nnx Variable's array."""
    try:
        var[...] = arr
    except Exception:  # pragma: no cover - version fallback
        var.value = arr


def _rngs(ctx):
    return ctx.rngs if ctx.rngs is not None else nnx.Rngs(0)


def _split4(arr, axis=-1):
    return list(u.math.split(arr, 4, axis=axis))


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

def _linear_to_bst(m, ctx):
    w = _get(m.kernel)
    has_bias = m.bias is not None
    layer = C.build_linear(w.shape[0], w.shape[1], has_bias)
    C.bst_set_linear(layer, w, _get(m.bias) if has_bias else None)
    return layer


def _linear_to_foreign(layer, ctx):
    w, b = C.bst_get_linear(layer)
    m = nnx.Linear(w.shape[0], w.shape[1], use_bias=b is not None, rngs=_rngs(ctx))
    _set(m.kernel, w)
    if b is not None:
        _set(m.bias, b)
    return m


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_to_bst(m, ctx):
    table = _get(m.embedding)
    layer = C.build_embedding(table.shape[0], table.shape[1])
    C.bst_set_embedding(layer, table)
    return layer


def _embed_to_foreign(layer, ctx):
    table = C.bst_get_embedding(layer)
    m = nnx.Embed(table.shape[0], table.shape[1], rngs=_rngs(ctx))
    _set(m.embedding, table)
    return m


# ---------------------------------------------------------------------------
# LayerNorm / RMSNorm / GroupNorm  (1-D affine vectors -> identity)
# ---------------------------------------------------------------------------

def _layernorm_to_bst(m, ctx):
    scale = None if m.scale is None else _get(m.scale)
    bias = None if m.bias is None else _get(m.bias)
    num = int(m.num_features)
    layer = C.build_layernorm((num,), scale is not None, bias is not None, float(m.epsilon))
    C.bst_set_norm(layer, 'weight', scale, bias)
    return layer


def _layernorm_to_foreign(layer, ctx):
    scale, offset = C.bst_get_norm(layer, 'weight', has_offset=True)
    num = int(layer.in_size[-1])
    m = nnx.LayerNorm(num, epsilon=float(layer.epsilon),
                      use_scale=scale is not None, use_bias=offset is not None, rngs=_rngs(ctx))
    if scale is not None:
        _set(m.scale, scale)
    if offset is not None:
        _set(m.bias, offset)
    return m


def _rmsnorm_to_bst(m, ctx):
    scale = None if m.scale is None else _get(m.scale)
    num = int(m.num_features)
    layer = C.build_rmsnorm((num,), scale is not None, float(m.epsilon))
    C.bst_set_norm(layer, 'scale', scale, None)
    return layer


def _rmsnorm_to_foreign(layer, ctx):
    scale, _ = C.bst_get_norm(layer, 'scale', has_offset=False)
    num = int(layer.in_size[-1])
    m = nnx.RMSNorm(num, epsilon=float(layer.epsilon),
                     use_scale=scale is not None, rngs=_rngs(ctx))
    if scale is not None:
        _set(m.scale, scale)
    return m


def _groupnorm_to_bst(m, ctx):
    scale = None if m.scale is None else _get(m.scale)
    bias = None if m.bias is None else _get(m.bias)
    num = int(m.num_groups) * int(m.group_size)
    layer = C.build_groupnorm((num,), int(m.num_groups), scale is not None, bias is not None,
                              float(m.epsilon))
    C.bst_set_norm(layer, 'weight', scale, bias)
    return layer


def _groupnorm_to_foreign(layer, ctx):
    scale, offset = C.bst_get_norm(layer, 'weight', has_offset=True)
    num = int(layer.in_size[-1])
    m = nnx.GroupNorm(num, num_groups=int(layer.num_groups), epsilon=float(layer.epsilon),
                      use_scale=scale is not None, use_bias=offset is not None, rngs=_rngs(ctx))
    if scale is not None:
        _set(m.scale, scale)
    if offset is not None:
        _set(m.bias, offset)
    return m


# ---------------------------------------------------------------------------
# Dropout (stateless; prob[keep] <-> rate[drop])
# ---------------------------------------------------------------------------

def _dropout_to_bst(m, ctx):
    return C.build_dropout(1.0 - float(m.rate))


def _dropout_to_foreign(layer, ctx):
    return nnx.Dropout(1.0 - float(layer.prob), rngs=_rngs(ctx))


# ---------------------------------------------------------------------------
# Conv
# ---------------------------------------------------------------------------

def _conv_bias_reshape_to_bst(b, nd):
    # nnx (out,) -> brainstate (1,...,1,out)
    return u.math.reshape(b, (1,) * nd + (b.shape[-1],))


def _conv_bias_reshape_to_foreign(b):
    # brainstate (1,...,1,out) -> nnx (out,)
    return u.math.reshape(b, (b.shape[-1],))


def _as_tuple(v, nd):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v,) * nd


def _conv_to_bst(m, ctx):
    w = _get(m.kernel)  # (*k, in//g, out)
    kernel_size = tuple(w.shape[:-2])
    nd = len(kernel_size)
    if any(d != 1 for d in _as_tuple(getattr(m, 'input_dilation', 1) or 1, nd)):
        raise ConversionError(
            "nnx Conv with `input_dilation` != 1 (transposed convolution) is not supported "
            "by this converter."
        )
    out_channels = w.shape[-1]
    in_size = ctx.require_size('Conv')
    has_bias = m.bias is not None
    groups = int(getattr(m, 'feature_group_count', 1))
    layer = C.build_conv(
        nd, in_size, out_channels, kernel_size,
        stride=getattr(m, 'strides', 1) or 1,
        padding=getattr(m, 'padding', 'SAME'),
        rhs_dilation=getattr(m, 'kernel_dilation', 1) or 1,
        groups=groups,
        has_bias=has_bias,
    )
    b = _conv_bias_reshape_to_bst(_get(m.bias), nd) if has_bias else None
    C.bst_set_linear(layer, w, b)
    return layer


def _conv_to_foreign(layer, ctx):
    w, b = C.bst_get_linear(layer)
    nd = len(layer.kernel_size)
    m = nnx.Conv(
        layer.in_channels, layer.out_channels, tuple(layer.kernel_size),
        strides=tuple(layer.stride),
        padding=layer.padding,
        kernel_dilation=tuple(layer.rhs_dilation),
        input_dilation=tuple(layer.lhs_dilation),
        feature_group_count=layer.groups,
        use_bias=b is not None,
        rngs=_rngs(ctx),
    )
    _set(m.kernel, w)
    if b is not None:
        _set(m.bias, _conv_bias_reshape_to_foreign(b))
    return m


# ---------------------------------------------------------------------------
# BatchNorm
# ---------------------------------------------------------------------------

def _bn_reshape_to_bst(v, nd):
    return u.math.reshape(v, (1,) * nd + (v.shape[-1],))


def _bn_reshape_to_foreign(v):
    return u.math.reshape(v, (v.shape[-1],))


def _batchnorm_to_bst(m, ctx):
    scale = None if m.scale is None else _get(m.scale)
    bias = None if m.bias is None else _get(m.bias)
    mean = _get(m.mean)
    var = _get(m.var)
    in_size = ctx.require_size('BatchNorm')
    nd = len(in_size) - 1
    affine = scale is not None
    layer = C.build_batchnorm(nd, in_size, float(m.epsilon), float(m.momentum), affine)
    C.bst_set_batchnorm(
        layer,
        _bn_reshape_to_bst(scale, nd) if scale is not None else None,
        _bn_reshape_to_bst(bias, nd) if bias is not None else None,
        _bn_reshape_to_bst(mean, nd),
        _bn_reshape_to_bst(var, nd),
    )
    return layer


def _batchnorm_to_foreign(layer, ctx):
    scale, offset, mean, var = C.bst_get_batchnorm(layer)
    num = (scale if scale is not None else mean).shape[-1]
    # ``use_running_average=True`` so the exported layer normalizes with the transferred running
    # statistics by default, matching brainstate's eval-mode forward (the only mode in which
    # BatchNorm output is framework-equivalent). Flip it for training.
    m = nnx.BatchNorm(num, momentum=float(layer.momentum), epsilon=float(layer.epsilon),
                      use_scale=scale is not None, use_bias=offset is not None,
                      use_running_average=True, rngs=_rngs(ctx))
    if scale is not None:
        _set(m.scale, _bn_reshape_to_foreign(scale))
    if offset is not None:
        _set(m.bias, _bn_reshape_to_foreign(offset))
    _set(m.mean, _bn_reshape_to_foreign(mean))
    _set(m.var, _bn_reshape_to_foreign(var))
    return m


# ---------------------------------------------------------------------------
# LSTMCell  (fused (in+h,4h) [i,g,f,o] + forget+1 fold  <->  8 Denses [i,f,g,o])
# ---------------------------------------------------------------------------

def _lstm_to_foreign(layer, ctx):
    W, bias = C.bst_get_lstm(layer)              # W: (in+h, 4h), bias: (4h,)
    num_in = W.shape[0] - layer.num_out
    h = layer.num_out
    Wi, Wh = W[:num_in], W[num_in:]
    ii, ig, if_, io = _split4(Wi)                # brainstate order i,g,f,o
    hi, hg, hf, ho = _split4(Wh)
    bi, bg, bf, bo = _split4(bias)
    m = nnx.LSTMCell(num_in, h, rngs=_rngs(ctx))
    _set(m.ii.kernel, ii); _set(m.if_.kernel, if_); _set(m.ig.kernel, ig); _set(m.io.kernel, io)
    _set(m.hi.kernel, hi); _set(m.hf.kernel, hf); _set(m.hg.kernel, hg); _set(m.ho.kernel, ho)
    _set(m.hi.bias, bi); _set(m.hf.bias, bf + 1.0); _set(m.hg.bias, bg); _set(m.ho.bias, bo)
    return m


def _lstm_to_bst(m, ctx):
    ii, if_, ig, io = _get(m.ii.kernel), _get(m.if_.kernel), _get(m.ig.kernel), _get(m.io.kernel)
    hi, hf, hg, ho = _get(m.hi.kernel), _get(m.hf.kernel), _get(m.hg.kernel), _get(m.ho.kernel)
    bi, bf, bg, bo = _get(m.hi.bias), _get(m.hf.bias), _get(m.hg.bias), _get(m.ho.bias)
    num_in = ii.shape[0]
    h = ii.shape[1]
    Wi = u.math.concatenate([ii, ig, if_, io], axis=-1)   # back to brainstate order i,g,f,o
    Wh = u.math.concatenate([hi, hg, hf, ho], axis=-1)
    W = u.math.concatenate([Wi, Wh], axis=0)
    bias = u.math.concatenate([bi, bg, bf - 1.0, bo], axis=-1)
    layer = C.build_lstm(num_in, h)
    C.bst_set_lstm(layer, W, bias)
    return layer


# ---------------------------------------------------------------------------
# Adapter + registration
# ---------------------------------------------------------------------------

class NnxAdapter(FrameworkAdapter):
    """Structural plumbing for flax.nnx."""

    name = 'nnx'

    def is_sequential(self, node) -> bool:
        return isinstance(node, getattr(nnx, 'Sequential', ()))

    def iter_children(self, node):
        return list(enumerate(node.layers))

    def has_child_modules(self, node) -> bool:
        for v in vars(node).values():
            if isinstance(v, nnx.Module):
                return True
            if isinstance(v, (list, tuple)) and any(isinstance(x, nnx.Module) for x in v):
                return True
        return False

    def layer_type(self, node) -> type:
        return type(node)

    def build_sequential(self, children: list):
        return nnx.Sequential(*children)


def _register():
    pairs = [
        (bnn.Linear, nnx.Linear, _linear_to_bst, _linear_to_foreign),
        (bnn.Embedding, nnx.Embed, _embed_to_bst, _embed_to_foreign),
        (bnn.LayerNorm, nnx.LayerNorm, _layernorm_to_bst, _layernorm_to_foreign),
        (bnn.RMSNorm, nnx.RMSNorm, _rmsnorm_to_bst, _rmsnorm_to_foreign),
        (bnn.GroupNorm, nnx.GroupNorm, _groupnorm_to_bst, _groupnorm_to_foreign),
        (bnn.Dropout, nnx.Dropout, _dropout_to_bst, _dropout_to_foreign),
        (bnn.Conv1d, nnx.Conv, _conv_to_bst, _conv_to_foreign),
        (bnn.Conv2d, nnx.Conv, _conv_to_bst, _conv_to_foreign),
        (bnn.Conv3d, nnx.Conv, _conv_to_bst, _conv_to_foreign),
        (bnn.BatchNorm1d, nnx.BatchNorm, _batchnorm_to_bst, _batchnorm_to_foreign),
        (bnn.BatchNorm2d, nnx.BatchNorm, _batchnorm_to_bst, _batchnorm_to_foreign),
        (bnn.BatchNorm3d, nnx.BatchNorm, _batchnorm_to_bst, _batchnorm_to_foreign),
        (bnn.LSTMCell, nnx.LSTMCell, _lstm_to_bst, _lstm_to_foreign),
    ]
    for bst_type, foreign_type, to_bst, to_foreign in pairs:
        register_layer_mapping(LayerMapping(bst_type, 'nnx', foreign_type, to_bst, to_foreign))

    # nnx.Conv is one class for all spatial dims; map import to the right brainstate class by
    # spatial-dim count handled in _conv_to_bst (it picks Conv{nd}d via build_conv).
    register_unsupported_foreign(
        'nnx', nnx.GRUCell,
        "GRUCell conversion is unsupported: brainstate's GRU uses the Cho-2014 variant "
        "(reset applied before the hidden matmul, `Wh([x, r*h])`) while flax/nnx use the cuDNN "
        "variant (reset applied after, `r * (W_hn h + b_hn)`). These are mathematically distinct "
        "and cannot be matched by transferring weights."
    )
    C.register_bst_unsupported()


_register()
