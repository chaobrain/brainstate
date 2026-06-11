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

"""equinox adapter + layer mappings.

equinox is torch-like: Linear weight is ``(out, in)`` and Conv weight is ``(out, in, *k)``
(NCHW), so kernels need transpose/permute relative to brainstate's flax-style layout. equinox
modules are immutable pytrees, so parameters are written via ``eqx.tree_at``.
"""

from __future__ import annotations

import brainunit as u

from .. import _common as C
from .._common import FrameworkAdapter, lazy_import, new_key
from .._errors import ConversionError
from .._registry import (LayerMapping, register_layer_mapping,
                         register_unsupported_bst_for, register_unsupported_foreign)

eqx = lazy_import('equinox')
import brainstate.nn as bnn  # noqa: E402

_CONV = {1: eqx.nn.Conv1d, 2: eqx.nn.Conv2d, 3: eqx.nn.Conv3d}


def _set(module, **fields):
    """Return ``module`` with the named array fields replaced (immutable update)."""
    names = list(fields.keys())
    return eqx.tree_at(lambda m: [getattr(m, n) for n in names], module,
                       [fields[n] for n in names])


def _ctx_key(ctx):
    """Return the user-supplied PRNG key, or a throwaway one if none was given.

    A PRNG key is a JAX array, so ``ctx.key or new_key()`` would evaluate the
    truthiness of an array and raise; test against ``None`` explicitly instead.
    The key only seeds the foreign layer's construction — its weights are
    overwritten immediately — so reusing one key across layers is harmless.
    """
    return new_key() if ctx.key is None else ctx.key


def _split4(arr, axis=-1):
    return list(u.math.split(arr, 4, axis=axis))


# ---------------------------------------------------------------------------
# Linear  (bst (in,out) <-> eqx (out,in))
# ---------------------------------------------------------------------------

def _linear_to_bst(m, ctx):
    w = u.math.transpose(m.weight, (1, 0))   # (out,in) -> (in,out)
    has_bias = m.bias is not None
    layer = C.build_linear(w.shape[0], w.shape[1], has_bias)
    C.bst_set_linear(layer, w, m.bias if has_bias else None)
    return layer


def _linear_to_foreign(layer, ctx):
    w, b = C.bst_get_linear(layer)
    m = eqx.nn.Linear(w.shape[0], w.shape[1], use_bias=b is not None, key=_ctx_key(ctx))
    m = _set(m, weight=u.math.transpose(w, (1, 0)))
    if b is not None:
        m = _set(m, bias=b)
    return m


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_to_bst(m, ctx):
    table = m.weight
    layer = C.build_embedding(table.shape[0], table.shape[1])
    C.bst_set_embedding(layer, table)
    return layer


def _embed_to_foreign(layer, ctx):
    table = C.bst_get_embedding(layer)
    m = eqx.nn.Embedding(table.shape[0], table.shape[1], key=_ctx_key(ctx))
    return _set(m, weight=table)


# ---------------------------------------------------------------------------
# LayerNorm / RMSNorm / GroupNorm  (1-D affine vectors -> identity)
# ---------------------------------------------------------------------------

def _layernorm_to_bst(m, ctx):
    scale = m.weight
    bias = m.bias
    num = int(m.shape[0])
    layer = C.build_layernorm((num,), scale is not None, bias is not None, float(m.eps))
    C.bst_set_norm(layer, 'weight', scale, bias)
    return layer


def _layernorm_to_foreign(layer, ctx):
    scale, offset = C.bst_get_norm(layer, 'weight', has_offset=True)
    num = int(layer.in_size[-1])
    m = eqx.nn.LayerNorm(num, eps=float(layer.epsilon),
                         use_weight=scale is not None, use_bias=offset is not None)
    if scale is not None:
        m = _set(m, weight=scale)
    if offset is not None:
        m = _set(m, bias=offset)
    return m


def _rmsnorm_to_bst(m, ctx):
    scale = m.weight
    num = int(m.shape[0])
    layer = C.build_rmsnorm((num,), scale is not None, float(m.eps))
    C.bst_set_norm(layer, 'scale', scale, None)
    return layer


def _rmsnorm_to_foreign(layer, ctx):
    scale, _ = C.bst_get_norm(layer, 'scale', has_offset=False)
    num = int(layer.in_size[-1])
    m = eqx.nn.RMSNorm(num, eps=float(layer.epsilon),
                        use_weight=scale is not None, use_bias=False)
    if scale is not None:
        m = _set(m, weight=scale)
    return m


def _groupnorm_to_bst(m, ctx):
    scale = m.weight
    bias = m.bias
    num = int(m.channels)
    layer = C.build_groupnorm((num,), int(m.groups), scale is not None, bias is not None,
                              float(m.eps))
    C.bst_set_norm(layer, 'weight', scale, bias)
    return layer


def _groupnorm_to_foreign(layer, ctx):
    scale, offset = C.bst_get_norm(layer, 'weight', has_offset=True)
    num = int(layer.in_size[-1])
    m = eqx.nn.GroupNorm(int(layer.num_groups), num, eps=float(layer.epsilon),
                         channelwise_affine=(scale is not None or offset is not None))
    if scale is not None:
        m = _set(m, weight=scale)
    if offset is not None:
        m = _set(m, bias=offset)
    return m


# ---------------------------------------------------------------------------
# Dropout (prob[keep] <-> p[drop])
# ---------------------------------------------------------------------------

def _dropout_to_bst(m, ctx):
    return C.build_dropout(1.0 - float(m.p))


def _dropout_to_foreign(layer, ctx):
    return eqx.nn.Dropout(1.0 - float(layer.prob))


# ---------------------------------------------------------------------------
# Conv  (bst (*k,in,out) NHWC <-> eqx (out,in,*k) NCHW)
# ---------------------------------------------------------------------------

def _kernel_perm_to_eqx(nd):
    # bst (*k, in, out) -> eqx (out, in, *k)
    return (nd + 1, nd) + tuple(range(nd))


def _conv_to_bst(m, ctx):
    w = m.weight                          # (out, in//g, *k)
    nd = w.ndim - 2
    perm = _kernel_perm_to_eqx(nd)
    inv = tuple(int(i) for i in __import__('numpy').argsort(perm))
    w_bst = u.math.transpose(w, inv)      # -> (*k, in//g, out)
    out_channels = w.shape[0]
    kernel_size = tuple(w.shape[2:])
    in_size = ctx.require_size('Conv')
    has_bias = m.bias is not None
    layer = C.build_conv(
        nd, in_size, out_channels, kernel_size,
        stride=m.stride, padding=m.padding, rhs_dilation=m.dilation,
        groups=m.groups, has_bias=has_bias,
    )
    b = u.math.reshape(m.bias, (1,) * nd + (out_channels,)) if has_bias else None
    C.bst_set_linear(layer, w_bst, b)
    return layer


def _conv_to_foreign(layer, ctx):
    w, b = C.bst_get_linear(layer)        # w: (*k, in//g, out)
    nd = len(layer.kernel_size)
    if any(d != 1 for d in layer.lhs_dilation):
        raise ConversionError(
            "equinox Conv does not support input (lhs) dilation; cannot export this layer."
        )
    cls = _CONV[nd]
    m = cls(layer.in_channels, layer.out_channels, tuple(layer.kernel_size),
            stride=tuple(layer.stride), padding=layer.padding,
            dilation=tuple(layer.rhs_dilation), groups=layer.groups,
            use_bias=b is not None, key=_ctx_key(ctx))
    w_eqx = u.math.transpose(w, _kernel_perm_to_eqx(nd))   # (out, in//g, *k)
    m = _set(m, weight=w_eqx)
    if b is not None:
        out_channels = layer.out_channels
        m = _set(m, bias=u.math.reshape(b, (out_channels,) + (1,) * nd))
    return m


# ---------------------------------------------------------------------------
# LSTMCell  (fused (in+h,4h) [i,g,f,o] + forget+1 fold  <->  weight_ih/hh (4h,*) [i,f,g,o])
# ---------------------------------------------------------------------------

def _lstm_to_foreign(layer, ctx):
    W, bias = C.bst_get_lstm(layer)
    num_in = W.shape[0] - layer.num_out
    h = layer.num_out
    Wi, Wh = W[:num_in], W[num_in:]
    ii, ig, if_, io = _split4(Wi)            # bst order i,g,f,o
    hi, hg, hf, ho = _split4(Wh)
    bi, bg, bf, bo = _split4(bias)
    # assemble in std order i,f,g,o, then transpose to (4h, *)
    weight_ih = u.math.transpose(u.math.concatenate([ii, if_, ig, io], axis=-1), (1, 0))
    weight_hh = u.math.transpose(u.math.concatenate([hi, hf, hg, ho], axis=-1), (1, 0))
    bias_std = u.math.concatenate([bi, bf + 1.0, bg, bo], axis=-1)   # forget +1 fold
    m = eqx.nn.LSTMCell(num_in, h, use_bias=True, key=_ctx_key(ctx))
    return _set(m, weight_ih=weight_ih, weight_hh=weight_hh, bias=bias_std)


def _lstm_to_bst(m, ctx):
    Wih = u.math.transpose(m.weight_ih, (1, 0))   # (in, 4h) std order i,f,g,o
    Whh = u.math.transpose(m.weight_hh, (1, 0))   # (h, 4h)
    si, sf, sg, so = _split4(Wih)
    ti, tf, tg, to = _split4(Whh)
    bi, bf, bg, bo = _split4(m.bias)
    num_in = Wih.shape[0]
    h = m.weight_hh.shape[1]
    Wi = u.math.concatenate([si, sg, sf, so], axis=-1)        # back to bst order i,g,f,o
    Wh = u.math.concatenate([ti, tg, tf, to], axis=-1)
    W = u.math.concatenate([Wi, Wh], axis=0)
    bias = u.math.concatenate([bi, bg, bf - 1.0, bo], axis=-1)
    layer = C.build_lstm(num_in, h)
    C.bst_set_lstm(layer, W, bias)
    return layer


# ---------------------------------------------------------------------------
# Adapter + registration
# ---------------------------------------------------------------------------

class EquinoxAdapter(FrameworkAdapter):
    """Structural plumbing for equinox."""

    name = 'equinox'

    def is_sequential(self, node) -> bool:
        return isinstance(node, eqx.nn.Sequential)

    def iter_children(self, node):
        return list(enumerate(node.layers))

    def has_child_modules(self, node) -> bool:
        import dataclasses
        for f in dataclasses.fields(node):
            v = getattr(node, f.name, None)
            if isinstance(v, eqx.Module):
                return True
            if isinstance(v, (list, tuple)) and any(isinstance(x, eqx.Module) for x in v):
                return True
            if isinstance(v, dict) and any(isinstance(x, eqx.Module) for x in v.values()):
                return True
        return False

    def layer_type(self, node) -> type:
        return type(node)

    def build_sequential(self, children: list):
        return eqx.nn.Sequential(children)


def _register():
    pairs = [
        (bnn.Linear, eqx.nn.Linear, _linear_to_bst, _linear_to_foreign),
        (bnn.Embedding, eqx.nn.Embedding, _embed_to_bst, _embed_to_foreign),
        (bnn.LayerNorm, eqx.nn.LayerNorm, _layernorm_to_bst, _layernorm_to_foreign),
        (bnn.RMSNorm, eqx.nn.RMSNorm, _rmsnorm_to_bst, _rmsnorm_to_foreign),
        (bnn.GroupNorm, eqx.nn.GroupNorm, _groupnorm_to_bst, _groupnorm_to_foreign),
        (bnn.Dropout, eqx.nn.Dropout, _dropout_to_bst, _dropout_to_foreign),
        (bnn.Conv1d, eqx.nn.Conv1d, _conv_to_bst, _conv_to_foreign),
        (bnn.Conv2d, eqx.nn.Conv2d, _conv_to_bst, _conv_to_foreign),
        (bnn.Conv3d, eqx.nn.Conv3d, _conv_to_bst, _conv_to_foreign),
        (bnn.LSTMCell, eqx.nn.LSTMCell, _lstm_to_bst, _lstm_to_foreign),
    ]
    for bst_type, foreign_type, to_bst, to_foreign in pairs:
        register_layer_mapping(LayerMapping(bst_type, 'equinox', foreign_type, to_bst, to_foreign))

    _bn_reason = (
        "equinox BatchNorm is unsupported: equinox stores running statistics in an external "
        "`eqx.nn.State` object (threaded through the forward pass under a named vmap axis), "
        "which cannot be expressed through the simple from_equinox/to_equinox(model) API."
    )
    for bn in (eqx.nn.BatchNorm,):
        register_unsupported_foreign('equinox', bn, _bn_reason)
    for bn in (bnn.BatchNorm0d, bnn.BatchNorm1d, bnn.BatchNorm2d, bnn.BatchNorm3d):
        register_unsupported_bst_for('equinox', bn, _bn_reason)
    register_unsupported_foreign('equinox', eqx.nn.GRUCell, C._GRU_REASON)
    C.register_bst_unsupported()


_register()
