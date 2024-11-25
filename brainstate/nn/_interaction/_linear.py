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

# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, Union, Optional

import brainunit as u
import jax
import jax.numpy as jnp
from jax.experimental.sparse.coo import coo_matvec_p, coo_matmat_p, COOInfo
from jax.experimental.sparse.csr import csr_matvec_p, csr_matmat_p

from brainstate import init, functional
from brainstate._state import ParamState
from brainstate.nn._module import Module
from brainstate.typing import ArrayLike, Size

__all__ = [
    'Linear',
    'ScaledWSLinear',
    'SignedWLinear',
    'CSRLinear',
    'CSCLinear',
    'COOLinear',
    'AllToAll',
    'OneToOne',
]


class Linear(Module):
    """
    Linear layer.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
        b_init: Optional[Union[Callable, ArrayLike]] = init.ZeroInit(),
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size
        assert self.in_size[:-1] == self.out_size[:-1], ('The first n-1 dimensions of "in_size" '
                                                         'and "out_size" must be the same.')

        # w_mask
        self.w_mask = init.param(w_mask, self.in_size + self.out_size)

        # weights
        params = dict(weight=init.param(w_init, (self.in_size[-1], self.out_size[-1]), allow_none=False))
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)
        self.weight = ParamState(params)

    def update(self, x):
        params = self.weight.value
        weight = params['weight']
        if self.w_mask is not None:
            weight = weight * self.w_mask
        y = u.linalg.dot(x, weight)
        if 'bias' in params:
            y = y + params['bias']
        return y


class SignedWLinear(Module):
    """
    Linear layer with signed weights.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
        w_sign: Optional[ArrayLike] = None,
        name: Optional[str] = None,

    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size
        assert self.in_size[:-1] == self.out_size[:-1], ('The first n-1 dimensions of "in_size" '
                                                         'and "out_size" must be the same.')

        # w_mask
        self.w_sign = w_sign

        # weights
        weight = init.param(w_init, self.in_size + self.out_size, allow_none=False)
        self.weight = ParamState(weight)

    def update(self, x):
        w = self.weight.value
        if self.w_sign is None:
            return u.math.matmul(x, u.math.abs(w))
        else:
            return u.math.matmul(x, u.math.abs(w) * self.w_sign)


class ScaledWSLinear(Module):
    """
    Linear Layer with Weight Standardization.

    Applies weight standardization to the weights of the linear layer.

    Parameters
    ----------
    in_size: int, sequence of int
      The input size.
    out_size: int, sequence of int
      The output size.
    w_init: Callable, ArrayLike
      The initializer for the weights.
    b_init: Callable, ArrayLike
      The initializer for the bias.
    w_mask: ArrayLike, Callable
      The optional mask of the weights.
    ws_gain: bool
      Whether to use gain for the weights. The default is True.
    eps: float
      The epsilon value for the weight standardization.
    name: str
      The name of the object.

    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        w_init: Callable = init.KaimingNormal(),
        b_init: Callable = init.ZeroInit(),
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        ws_gain: bool = True,
        eps: float = 1e-4,
        name: str = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size
        assert self.in_size[:-1] == self.out_size[:-1], ('The first n-1 dimensions of "in_size" '
                                                         'and "out_size" must be the same.')

        # w_mask
        self.w_mask = init.param(w_mask, (self.in_size[0], 1))

        # parameters
        self.eps = eps

        # weights
        params = dict(weight=init.param(w_init, self.in_size + self.out_size, allow_none=False))
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size, allow_none=False)
        # gain
        if ws_gain:
            s = params['weight'].shape
            params['gain'] = jnp.ones((1,) * (len(s) - 1) + (s[-1],), dtype=params['weight'].dtype)
        self.weight = ParamState(params)

    def update(self, x):
        params = self.weight.value
        w = params['weight']
        w = functional.weight_standardization(w, self.eps, params.get('gain', None))
        if self.w_mask is not None:
            w = w * self.w_mask
        y = u.linalg.dot(x, w)
        if 'bias' in params:
            y = y + params['bias']
        return y


def csr_matmat(data, indices, indptr, B: jax.Array, *, shape, transpose: bool = False) -> jax.Array:
    """Product of CSR sparse matrix and a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      B : array of shape ``(mat.shape[0] if transpose else mat.shape[1], cols)`` and
        dtype ``mat.dtype``
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      C : array of shape ``(mat.shape[1] if transpose else mat.shape[0], cols)``
        representing the matrix vector product.
    """
    return csr_matmat_p.bind(data, indices, indptr, B, shape=shape, transpose=transpose)


def csr_matvec(data, indices, indptr, v, *, shape, transpose=False) -> jax.Array:
    """Product of CSR sparse matrix and a dense vector.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      v : array of shape ``(shape[0] if transpose else shape[1],)``
        and dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
        the matrix vector product.
    """
    return csr_matvec_p.bind(data, indices, indptr, v, shape=shape, transpose=transpose)


class CSRLinear(Module):
    """
    Linear layer with Compressed Sparse Row (CSR) matrix.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        indptr: ArrayLike,
        indices: ArrayLike,
        weight: Union[Callable, ArrayLike],
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size
        assert self.in_size[:-1] == self.out_size[:-1], ('The first n-1 dimensions of "in_size" '
                                                         'and "out_size" must be the same.')

        # CSR data structure
        indptr = jnp.asarray(indptr)
        indices = jnp.asarray(indices)
        assert indptr.ndim == 1, f"indptr must be 1D. Got: {indptr.ndim}"
        assert indices.ndim == 1, f"indices must be 1D. Got: {indices.ndim}"
        assert indptr.size == self.in_size[-1] + 1, f"indptr must have size {self.in_size[-1] + 1}. Got: {indptr.size}"
        with jax.ensure_compile_time_eval():
            self.indptr = u.math.asarray(indptr)
            self.indices = u.math.asarray(indices)

        # weights
        weight = init.param(weight, (len(indices),), allow_none=False, allow_scalar=False)
        params = dict(weight=weight)
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)
        self.weight = ParamState(params)

    def update(self, x):
        data = self.weight.value['weight']
        data, w_unit = u.get_mantissa(data), u.get_unit(data)
        x, x_unit = u.get_mantissa(x), u.get_unit(x)
        shape = [self.in_size[-1], self.out_size[-1]]
        if x.ndim == 1:
            y = csr_matvec(data, self.indices, self.indptr, x, shape=shape)
        elif x.ndim == 2:
            y = csr_matmat(data, self.indices, self.indptr, x, shape=shape)
        else:
            raise NotImplementedError(f"matmul with object of shape {x.shape}")
        y = u.maybe_decimal(u.Quantity(y, unit=w_unit * x_unit))
        if 'bias' in self.weight.value:
            y = y + self.weight.value['bias']
        return y


class CSCLinear(Module):
    """
    Linear layer with Compressed Sparse Column (CSC) matrix.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        indptr: ArrayLike,
        indices: ArrayLike,
        weight: Union[Callable, ArrayLike],
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size
        assert self.in_size[:-1] == self.out_size[:-1], ('The first n-1 dimensions of "in_size" '
                                                         'and "out_size" must be the same.')

        # CSR data structure
        indptr = jnp.asarray(indptr)
        indices = jnp.asarray(indices)
        assert indptr.ndim == 1, f"indptr must be 1D. Got: {indptr.ndim}"
        assert indices.ndim == 1, f"indices must be 1D. Got: {indices.ndim}"
        assert indptr.size == self.in_size[-1] + 1, f"indptr must have size {self.in_size[-1] + 1}. Got: {indptr.size}"
        with jax.ensure_compile_time_eval():
            self.indptr = u.math.asarray(indptr)
            self.indices = u.math.asarray(indices)

        # weights
        weight = init.param(weight, (len(indices),), allow_none=False, allow_scalar=False)
        params = dict(weight=weight)
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)
        self.weight = ParamState(params)

    def update(self, x):
        data = self.weight.value['weight']
        data, w_unit = u.get_mantissa(data), u.get_unit(data)
        x, x_unit = u.get_mantissa(x), u.get_unit(x)
        shape = [self.out_size[-1], self.in_size[-1]]
        if x.ndim == 1:
            y = csr_matvec(data, self.indices, self.indptr, x, shape=shape, transpose=True)
        elif x.ndim == 2:
            y = csr_matmat(data, self.indices, self.indptr, x, shape=shape, transpose=True)
        else:
            raise NotImplementedError(f"matmul with object of shape {x.shape}")
        y = u.maybe_decimal(u.Quantity(y, unit=w_unit * x_unit))
        if 'bias' in self.weight.value:
            y = y + self.weight.value['bias']
        return y


def coo_matvec(
    data: jax.Array,
    row: jax.Array,
    col: jax.Array,
    v: jax.Array, *,
    spinfo: COOInfo,
    transpose: bool = False
) -> jax.Array:
    """Product of COO sparse matrix and a dense vector.

    Args:
      data : array of shape ``(nse,)``.
      row : array of shape ``(nse,)``
      col : array of shape ``(nse,)`` and dtype ``row.dtype``
      v : array of shape ``(shape[0] if transpose else shape[1],)`` and
        dtype ``data.dtype``
      spinfo : COOInfo object containing the shape of the matrix and the dtype
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
        the matrix vector product.
    """
    return coo_matvec_p.bind(data, row, col, v, spinfo=spinfo, transpose=transpose)


def coo_matmat(
    data: jax.Array, row: jax.Array, col: jax.Array, B: jax.Array, *,
    spinfo: COOInfo, transpose: bool = False
) -> jax.Array:
    """Product of COO sparse matrix and a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      row : array of shape ``(nse,)``
      col : array of shape ``(nse,)`` and dtype ``row.dtype``
      B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
        dtype ``data.dtype``
      spinfo : COOInfo object containing the shape of the matrix and the dtype
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      C : array of shape ``(shape[1] if transpose else shape[0], cols)``
        representing the matrix vector product.
    """
    return coo_matmat_p.bind(data, row, col, B, spinfo=spinfo, transpose=transpose)


class COOLinear(Module):

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        row: ArrayLike,
        col: ArrayLike,
        weight: Union[Callable, ArrayLike],
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        rows_sorted: bool = False,
        cols_sorted: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size
        assert self.in_size[:-1] == self.out_size[:-1], ('The first n-1 dimensions of "in_size" '
                                                         'and "out_size" must be the same.')

        # COO data structure
        row = jnp.asarray(row)
        col = jnp.asarray(col)
        assert row.ndim == 1, f"row must be 1D. Got: {row.ndim}"
        assert col.ndim == 1, f"col must be 1D. Got: {col.ndim}"
        assert row.size == col.size, f"row and col must have the same size. Got: {row.size} and {col.size}"
        with jax.ensure_compile_time_eval():
            self.row = u.math.asarray(row)
            self.col = u.math.asarray(col)

        # COO structure information
        self.rows_sorted = rows_sorted
        self.cols_sorted = cols_sorted

        # weights
        weight = init.param(weight, (len(row),), allow_none=False, allow_scalar=False)
        params = dict(weight=weight)
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)
        self.weight = ParamState(params)

    def update(self, x):
        data = self.weight.value['weight']
        data, w_unit = u.get_mantissa(data), u.get_unit(data)
        x, x_unit = u.get_mantissa(x), u.get_unit(x)
        spinfo = COOInfo(
            shape=(self.in_size[-1], self.out_size[-1]),
            rows_sorted=self.rows_sorted,
            cols_sorted=self.cols_sorted
        )
        if x.ndim == 1:
            y = coo_matvec(data, self.row, self.col, x, spinfo=spinfo, transpose=False)
        elif x.ndim == 2:
            y = coo_matmat(data, self.row, self.col, x, spinfo=spinfo, transpose=False)
        else:
            raise NotImplementedError(f"matmul with object of shape {x.shape}")
        y = u.maybe_decimal(u.Quantity(y, unit=w_unit * x_unit))
        if 'bias' in self.weight.value:
            y = y + self.weight.value['bias']
        return y


class AllToAll(Module):
    """
    Synaptic matrix multiplication with All-to-All connections.

    Args:
      in_size: Size. The number of neurons in the pre-synaptic neuron group.
      out_size: Size. The number of neurons in the postsynaptic neuron group.
      w_init: The synaptic weight initializer.
      include_self: bool. Whether connect the neuron with at the same position.
      name: str. The object name.
    """

    def __init__(
        self,
        in_size: Size,
        out_size: Size,
        w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        include_self: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size
        assert self.in_size[:-1] == self.out_size[:-1], ('The first n-1 dimensions of "in_size" '
                                                         'and "out_size" must be the same.')

        # others
        self.include_self = include_self

        # weights
        weight = init.param(w_init, (self.in_size[-1], self.out_size[-1]), allow_none=False)
        params = dict(weight=weight)
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)
        self.weight = ParamState(params)

    def update(self, pre_val):
        params = self.weight.value
        pre_val, pre_unit = u.get_mantissa(pre_val), u.get_unit(pre_val)
        w_val, w_unit = u.get_mantissa(params['weight']), u.get_unit(params['weight'])

        if u.math.ndim(w_val) == 0:  # weight is a scalar
            if pre_val.ndim == 1:
                post_val = u.math.sum(pre_val)
            else:
                post_val = u.math.sum(pre_val, keepdims=True, axis=-1)
            if not self.include_self:
                if self.in_size == self.out_size:
                    post_val = post_val - pre_val
                elif self.in_size[-1] > self.out_size[-1]:
                    val = pre_val[..., :self.out_size[-1]]
                    post_val = post_val - val
                else:
                    size = list(self.out_size)
                    size[-1] = self.out_size[-1] - self.in_size[-1]
                    val = u.math.concatenate([pre_val, u.math.zeros(size, dtype=pre_val.dtype)])
                    post_val = post_val - val
            post_val = w_val * post_val

        else:  # weight is a matrix
            assert u.math.ndim(w_val) == 2, '"weight" must be a 2D matrix.'
            if not self.include_self:
                post_val = pre_val @ u.math.fill_diagonal(w_val, 0.)
            else:
                post_val = pre_val @ w_val

        post_val = u.maybe_decimal(u.Quantity(post_val, unit=w_unit * pre_unit))
        if 'bias' in params:
            post_val = post_val + params['bias']
        return post_val


class OneToOne(Module):
    """
    Synaptic matrix multiplication with One2One connection.

    Args:
        in_size: Size. The number of neurons in the pre-synaptic neuron group.
        w_init: The synaptic weight initializer.
        b_init: The synaptic bias initializer.
        name: str. The object name.
    """

    def __init__(
        self,
        in_size: Size,
        w_init: Union[Callable, ArrayLike] = init.Normal(),
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = in_size

        # weights
        param = dict(weight=init.param(w_init, self.in_size, allow_none=False))
        if b_init is not None:
            param['bias'] = init.param(b_init, self.out_size, allow_none=False)
        self.weight = param

    def update(self, pre_val):
        pre_val, pre_unit = u.get_mantissa(pre_val), u.get_unit(pre_val)
        w_val, w_unit = u.get_mantissa(self.weight['weight']), u.get_unit(self.weight['weight'])
        post_val = pre_val * w_val
        post_val = u.maybe_decimal(u.Quantity(post_val, unit=w_unit * pre_unit))
        if 'bias' in self.weight:
            post_val = post_val + self.weight['bias']
        return post_val
