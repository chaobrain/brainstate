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

import operator
from typing import Callable

import brainunit as u
import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import numpy as np
from brainunit.sparse._csr import (
    _csr_matvec as csr_matvec,
    _csr_matmat as csr_matmat,
    _csr_to_coo as csr_to_coo
)
from jax.experimental.sparse import JAXSparse
from jax.interpreters import ad

from brainstate.typing import Shape
from ._xla_custom_op import XLACustomOp

__all__ = [
    'CSR',
    'CSC',
]


@jax.tree_util.register_pytree_node_class
class CSR(u.sparse.SparseMatrix):
    """
    Event-driven and Unit-aware CSR matrix.
    """
    data: jax.Array | u.Quantity
    indices: jax.Array
    indptr: jax.Array
    shape: tuple[int, int]
    nse = property(lambda self: self.data.size)
    dtype = property(lambda self: self.data.dtype)
    _bufs = property(lambda self: (self.data, self.indices, self.indptr))

    def __init__(self, args, *, shape):
        self.data, self.indices, self.indptr = map(u.math.asarray, args)
        super().__init__(args, shape=shape)

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=np.int32):
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        return u.sparse.csr_fromdense(mat, nse=nse, index_dtype=index_dtype)

    def with_data(self, data: jax.Array | u.Quantity) -> CSR:
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return CSR((data, self.indices, self.indptr), shape=self.shape)

    def todense(self):
        return u.sparse.csr_todense(self)

    def transpose(self, axes=None):
        assert axes is None, "transpose does not support axes argument."
        return CSC((self.data, self.indices, self.indptr), shape=self.shape[::-1])

    def __abs__(self):
        return CSR((abs(self.data), self.indices, self.indptr), shape=self.shape)

    def __neg__(self):
        return CSR((-self.data, self.indices, self.indptr), shape=self.shape)

    def __pos__(self):
        return CSR((self.data.__pos__(), self.indices, self.indptr), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    (op(self.data, other.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    (op(other.data, self.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: jax.Array | u.Quantity) -> CSR:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: jax.Array | u.Quantity) -> CSR:
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: jax.Array | u.Quantity) -> CSR:
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: jax.Array | u.Quantity) -> CSR:
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> CSR:
        return self.__div__(other)

    def __rtruediv__(self, other) -> CSR:
        return self.__rdiv__(other)

    def __add__(self, other) -> CSR:
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> CSR:
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> CSR:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> CSR:
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> CSR:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> CSR:
        return self._binary_rop(other, operator.mod)

    def __matmul__(self, other):
        # csr @ other
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = u.math.asarray(other)
        data = self.data
        # data, other = u.math.promote_dtypes(self.data, other)
        if other.ndim == 1:
            return _csr_matvec(
                data,
                self.indices,
                self.indptr,
                other,
                shape=self.shape
            )
        elif other.ndim == 2:
            return _csr_matmat(
                data,
                self.indices,
                self.indptr,
                other,
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        # other @ csr
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = u.math.asarray(other)
        data = self.data
        # data, other = u.math.promote_dtypes(self.data, other)
        if other.ndim == 1:
            return _csr_matvec(
                data,
                self.indices,
                self.indptr,
                other,
                shape=self.shape,
                transpose=True
            )
        elif other.ndim == 2:
            other = other.T
            r = _csr_matmat(
                data,
                self.indices,
                self.indptr,
                other,
                shape=self.shape,
                transpose=True
            )
            return r.T
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def tree_flatten(self):
        return (self.data,), {"shape": self.shape, "indices": self.indices, "indptr": self.indptr}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.data, = children
        if aux_data.keys() != {'shape', 'indices', 'indptr'}:
            raise ValueError(f"CSR.tree_unflatten: invalid {aux_data=}")
        obj.__dict__.update(**aux_data)
        return obj


@jax.tree_util.register_pytree_node_class
class CSC(u.sparse.SparseMatrix):
    """
    Event-driven and Unit-aware CSC matrix.
    """
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    shape: tuple[int, int]
    nse = property(lambda self: self.data.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(self, args, *, shape):
        self.data, self.indices, self.indptr = map(u.math.asarray, args)
        super().__init__(args, shape=shape)

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=np.int32):
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        return u.sparse.csr_fromdense(mat.T, nse=nse, index_dtype=index_dtype).T

    @classmethod
    def _empty(cls, shape, *, dtype=None, index_dtype='int32'):
        """Create an empty CSC instance. Public method is sparse.empty()."""
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError(f"CSC must have ndim=2; got {shape=}")
        data = jnp.empty(0, dtype)
        indices = jnp.empty(0, index_dtype)
        indptr = jnp.zeros(shape[1] + 1, index_dtype)
        return cls((data, indices, indptr), shape=shape)

    @classmethod
    def _eye(cls, N, M, k, *, dtype=None, index_dtype='int32'):
        return CSR._eye(M, N, -k, dtype=dtype, index_dtype=index_dtype).T

    def with_data(self, data: jax.Array | u.Quantity) -> CSC:
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return CSC((data, self.indices, self.indptr), shape=self.shape)

    def todense(self):
        return u.sparse.csr_todense(self.T).T

    def transpose(self, axes=None):
        assert axes is None
        return CSR((self.data, self.indices, self.indptr), shape=self.shape[::-1])

    def __abs__(self):
        return CSC((abs(self.data), self.indices, self.indptr), shape=self.shape)

    def __neg__(self):
        return CSC((-self.data, self.indices, self.indptr), shape=self.shape)

    def __pos__(self):
        return CSC((self.data.__pos__(), self.indices, self.indptr), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(self.data, other.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(other.data, self.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: jax.Array | u.Quantity) -> 'CSC':
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: jax.Array | u.Quantity) -> 'CSC':
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: jax.Array | u.Quantity) -> CSC:
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: jax.Array | u.Quantity) -> CSC:
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> CSC:
        return self.__div__(other)

    def __rtruediv__(self, other) -> CSC:
        return self.__rdiv__(other)

    def __add__(self, other) -> CSC:
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> CSC:
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> CSC:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> CSC:
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> CSC:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> CSC:
        return self._binary_rop(other, operator.mod)

    def __matmul__(self, other):
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = u.math.asarray(other)
        data, other = u.math.promote_dtypes(self.data, other)
        if other.ndim == 1:
            return _csr_matvec(
                data,
                self.indices,
                self.indptr,
                other,
                shape=self.shape[::-1],
                transpose=True
            )
        elif other.ndim == 2:
            return _csr_matmat(
                data,
                self.indices,
                self.indptr,
                other,
                shape=self.shape[::-1],
                transpose=True
            )
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = u.math.asarray(other)
        data, other = u.math.promote_dtypes(self.data, other)
        if other.ndim == 1:
            return _csr_matvec(
                data,
                self.indices,
                self.indptr,
                other,
                shape=self.shape[::-1],
                transpose=False
            )
        elif other.ndim == 2:
            other = other.T
            r = _csr_matmat(
                data,
                self.indices,
                self.indptr, other,
                shape=self.shape[::-1],
                transpose=False
            )
            return r.T
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def tree_flatten(self):
        return (self.data,), {"shape": self.shape, "indices": self.indices, "indptr": self.indptr}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.data, = children
        if aux_data.keys() != {'shape', 'indices', 'indptr'}:
            raise ValueError(f"CSR.tree_unflatten: invalid {aux_data=}")
        obj.__dict__.update(**aux_data)
        return obj


def _csr_matvec(
    data: jax.Array | u.Quantity,
    indices: jax.Array,
    indptr: jax.Array,
    v: jax.Array | u.Quantity,
    *,
    shape: Shape,
    transpose: bool = False,
    float_as_event: bool = True,
) -> jax.Array | u.Quantity:
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
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = event_csrmv_p_call(
        data, indices, indptr, v,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


def _csr_matmat(
    data: jax.Array | u.Quantity,
    indices: jax.Array,
    indptr: jax.Array,
    B: jax.Array | u.Quantity,
    *,
    shape: Shape,
    transpose: bool = False,
    float_as_event: bool = True,
) -> jax.Array | u.Quantity:
    """
    Product of CSR sparse matrix and a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
        dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      C : array of shape ``(shape[1] if transpose else shape[0], cols)``
        representing the matrix-matrix product.
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = event_csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


Kernel = Callable


def event_csrmv_cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

            elif float_as_event:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for i in range(v.shape[0]):
                        sp = v[i]
                        if sp != 0.:
                            wsp = w * sp
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += wsp

        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[()]
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += w
                        posts[i] = r

            elif float_as_event:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[()]
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += w
                        posts[i] = r

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[()]
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            c = v[indices[j]]
                            if c != 0.:
                                r += w * c
                        posts[i] = r

    else:
        if transpose:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

            elif float_as_event:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        sp = v[i]
                        if sp != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j] * sp

        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += weights[j]
                        posts[i] = r

            elif float_as_event:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += weights[j]
                        posts[i] = r

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            c = v[indices[j]]
                            if c != 0.:
                                r += weights[j] * c
                        posts[i] = r

    return mv


def event_csrmv_gpu_kernel_generator(
    block_size: int,
    shape: Shape,
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    if transpose:
        # v @ csr   (shape[0], ) @ (shape[0], shape[1]) -> (shape[1], )
        if weight_info.size == 1:
            # 每个block处理一个[block_size,]的indptr
            def _kernel(
                weights_ref,    # [num_nonzeros]
                indices_ref,    # [num_nonzeros]
                indptr_ref,     # [n_rows + 1], here is [block_size]
                v_ref,          # [n_rows]
                posts_ref       # [n_cols]
            ):
                def process_row(row_i):
                    weight_0 = weights_ref[0]
                    if spike_info.dtype == jnp.bool_:
                        # Check if the input vector has a non-zero value at this row
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the row
                            def process_element(j, _):
                                col_idx = indices_ref[j]
                                # Atomic add to the output vector
                                pl.atomic_add(posts_ref, col_idx, weight_0)
                            
                            # Iterate over the non-zero elements in the row
                            jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], process_element, None)
                            return None
                        
                        jax.lax.cond(
                            v_ref[row_i],
                            true_fn,
                            lambda _: None,
                            None
                        )
                    elif float_as_event:
                        # Check if the input vector has a non-zero value at this row
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the row
                            def process_element(j, _):
                                col_idx = indices_ref[j]
                                # Atomic add to the output vector
                                pl.atomic_add(posts_ref, col_idx, weight_0)
                            
                            # Iterate over the non-zero elements in the row
                            jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], process_element, None)
                            return None
                        jax.lax.cond(
                            v_ref[row_i] != 0.,
                            true_fn,
                            lambda _: None,
                            None
                        )
                    else:
                        # Check if the input vector has a non-zero value at this row
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the row
                            def process_element(j, _):
                                col_idx = indices_ref[j]
                                # Atomic add to the output vector
                                pl.atomic_add(posts_ref, col_idx, weight_0 * v_ref[row_i])
                            
                            # Iterate over the non-zero elements in the row
                            jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], process_element, None)
                            return None
                        jax.lax.cond(
                            v_ref[row_i] != 0.,
                            true_fn,
                            lambda _: None,
                            None
                        )
                # Iterate over the rows in the block
                jax.lax.fori_loop(
                    0,
                    indptr_ref.shape[0] - 1,
                    lambda i, _: process_row(i),
                    None
                )
            
            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct([shape[1]], weight_info.dtype),
                ],
                out_specs=[
                    pl.BlockSpec((shape[1],), lambda i, j: j),
                ],
                in_specs=[
                    pl.BlockSpec((weight_info.size,), lambda i, j: j),
                    pl.BlockSpec((indices_info.size,), lambda i, j: j),
                    pl.BlockSpec((block_size,), lambda i, j: j),
                    pl.BlockSpec((shape[0],), lambda i, j: j)
                ],
                grid=(
                    pl.cdiv(shape[0], block_size),
                    1,
                ),
                interpret=False,
            )
        else:
            # 每个block处理一个[block_size,]的indptr
            def _kernel(
                weights_ref,    # [num_nonzeros]
                indices_ref,    # [num_nonzeros]
                indptr_ref,     # [n_rows + 1], here is [block_size]
                v_ref,          # [n_rows]
                posts_ref       # [n_cols]
            ):
                def process_row(row_i):
                    if spike_info.dtype == jnp.bool_:
                        # Check if the input vector has a non-zero value at this row
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the row
                            def process_element(j, _):
                                col_idx = indices_ref[j]
                                # Atomic add to the output vector
                                pl.atomic_add(posts_ref, col_idx, weights_ref[j])
                            
                            # Iterate over the non-zero elements in the row
                            jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], process_element, None)
                            return None
                        
                        jax.lax.cond(
                            v_ref[row_i],
                            true_fn,
                            lambda _: None,
                            None
                        )
                    elif float_as_event:
                        # Check if the input vector has a non-zero value at this row
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the row
                            def process_element(j, _):
                                col_idx = indices_ref[j]
                                # Atomic add to the output vector
                                pl.atomic_add(posts_ref, col_idx, weights_ref[j])
                            
                            # Iterate over the non-zero elements in the row
                            jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], process_element, None)
                            return None
                        jax.lax.cond(
                            v_ref[row_i] != 0.,
                            true_fn,
                            lambda _: None,
                            None
                        )
                    else:
                        # Check if the input vector has a non-zero value at this row
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the row
                            def process_element(j, _):
                                col_idx = indices_ref[j]
                                # Atomic add to the output vector
                                pl.atomic_add(posts_ref, col_idx, weights_ref[j] * v_ref[row_i])
                            
                            # Iterate over the non-zero elements in the row
                            jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], process_element, None)
                            return None
                        jax.lax.cond(
                            v_ref[row_i] != 0.,
                            true_fn,
                            lambda _: None,
                            None
                        )
                # Iterate over the rows in the block
                jax.lax.fori_loop(
                    0,
                    indptr_ref.shape[0] - 1,
                    lambda i, _: process_row(i),
                    None
                )
            
            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct([shape[1]], weight_info.dtype),
                ],
                out_specs=[
                    pl.BlockSpec((shape[1],), lambda i, j: j)
                ],
                in_specs=[
                    pl.BlockSpec((weight_info.size,), lambda i, j: j),
                    pl.BlockSpec((indices_info.size,), lambda i, j: j),
                    pl.BlockSpec((block_size,), lambda i, j: j),
                    pl.BlockSpec((shape[0],), lambda i, j: j)
                ],
                grid=(
                    pl.cdiv(shape[0], block_size),
                    1,
                ),
                interpret=False,
            )
        
    else:
        # TODO: Need test
        # csr @ v   (shape[0], shape[1]) @ (shape[1], ) -> (shape[0], )
        if weight_info.size == 1:
            # 每个block处理一个[block_size,]的indptr
            def _kernel(
                weights_ref,    # [1] (scalar weight for homogeneous matrix)
                indices_ref,    # [num_nonzeros] (column indices of non-zero values)
                indptr_ref,     # [n_rows + 1] (row pointers)
                v_ref,          # [n_cols] (input vector)
                posts_ref       # [n_rows] (output vector)
            ):
                weight_0 = weights_ref[0]

                def process_row(row_i):
                    # Initialize the result for this row
                    r = 0.0

                    # Define a function to process a single non-zero element in the row
                    def process_element(j, r_):
                        col_idx = indices_ref[j]
                        if spike_info.dtype == jnp.bool_:
                            r_ = jax.lax.cond(
                                v_ref[col_idx],
                                lambda: r_ + 1,
                                lambda: r_
                            )
                        elif float_as_event:
                            r_ = jax.lax.cond(
                                v_ref[col_idx] != 0.,
                                lambda: r_ + 1,
                                lambda: r_
                            )
                        else:
                            r_ = jax.lax.cond(
                                v_ref[col_idx] != 0.,
                                lambda: r_ + v_ref[col_idx],
                                lambda: r_
                            )
                        return r_

                    # Iterate over the non-zero elements in the row
                    r = jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], process_element, r) * weight_0

                    # Store the result in the output vector
                    posts_ref[row_i] = r

                # Iterate over the rows in the block
                jax.lax.fori_loop(
                    0,
                    indptr_ref.shape[0] - 1,
                    lambda i, _: process_row(i),
                    None
                )

            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct([shape[0]], weight_info.dtype),
                ],
                out_specs=[
                    pl.BlockSpec((shape[0],), lambda i, j: i),
                ],
                in_specs=[
                    pl.BlockSpec((weight_info.size,), lambda i, j: j),
                    pl.BlockSpec((indices_info.size,), lambda i, j: j),
                    pl.BlockSpec((block_size,), lambda i, j: i),
                    pl.BlockSpec((shape[1],), lambda i, j: j),
                ],
                grid=(
                    pl.cdiv(shape[0], block_size),
                    1,
                ),
                interpret=False,
            )
        else:
            # 每个block处理一个[block_size,]的indptr
            def _kernel(
                weights_ref,    # [num_nonzeros] (non-zero values)
                indices_ref,    # [num_nonzeros] (column indices of non-zero values)
                indptr_ref,     # [n_rows + 1] (row pointers)
                v_ref,          # [n_cols] (input vector)
                posts_ref       # [n_rows] (output vector)
            ):
                def process_row(row_i):
                    # Initialize the result for this row
                    r = 0.0

                    # Define a function to process a single non-zero element in the row
                    def process_element(j, r_):
                        col_idx = indices_ref[j]
                        if spike_info.dtype == jnp.bool_:
                            r_ = jax.lax.cond(
                                v_ref[col_idx],
                                lambda: r_ + weights_ref[j],
                                lambda: r_
                            )
                        elif float_as_event:
                            r_ = jax.lax.cond(
                                v_ref[col_idx] != 0.,
                                lambda: r_ + weights_ref[j],
                                lambda: r_
                            )
                        else:
                            r_ = jax.lax.cond(
                                v_ref[col_idx] != 0.,
                                lambda: r_ + weights_ref[j] * v_ref[col_idx],
                                lambda: r_
                            )
                        return r_

                    # Iterate over the non-zero elements in the row
                    r = jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], process_element, r)

                    # Store the result in the output vector
                    posts_ref[row_i] = r

                # Iterate over the rows in the block
                jax.lax.fori_loop(
                    0,
                    indptr_ref.shape[0] - 1,
                    lambda i, _: process_row(i),
                    None
                )

            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct([shape[0]], weight_info.dtype),
                ],
                out_specs=[
                    pl.BlockSpec((shape[0],), lambda i, j: i),
                ],
                in_specs=[
                    pl.BlockSpec((weight_info.size,), lambda i, j: j),
                    pl.BlockSpec((indices_info.size,), lambda i, j: j),
                    pl.BlockSpec((block_size,), lambda i, j: i),
                    pl.BlockSpec((shape[1],), lambda i, j: j),
                ],
                grid=(
                    pl.cdiv(shape[0], block_size),
                    1,
                ),
                interpret=False,
            )
    
    return kernel
    

def event_csrmv_jvp_v(
    v_dot,
    data,
    indices,
    indptr,
    v,
    *,
    shape,
    transpose,
    **kwargs
):
    return [
        csr_matvec(
            data,
            indices,
            indptr,
            v_dot,
            shape=shape,
            transpose=transpose
        )
    ]


def event_csrmv_jvp_weights(
    data_dot,
    data,
    indices,
    indptr,
    v,
    *,
    shape,
    transpose,
    float_as_event,
    **kwargs
):
    return event_csrmv_p_call(
        data_dot,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
    )


def event_csrmv_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    events,
    *,
    shape,
    float_as_event,
    transpose,
    **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(events):
        ct_events = csr_matvec(
            data,
            indices,
            indptr,
            ct,
            shape=shape,
            transpose=not transpose
        )[0]
        return data, indices, indptr, (ad.Zero(events) if type(ct) is ad.Zero else ct_events)
    else:
        if type(ct[0]) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = event_csrmv_p_call(
                    jnp.ones(1, dtype=data.dtype),
                    indices,
                    indptr,
                    events,
                    shape=shape,
                    transpose=transpose,
                    float_as_event=float_as_event,
                )[0]
                ct_values = jnp.inner(ct, ct_values)
            else:  # heterogeneous values
                row, col = csr_to_coo(indices, indptr)
                ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
        return ct_values, indices, indptr, events


def event_csrmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = event_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = event_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        return r, [1]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven CSR matrix-vector product.")


event_csrmv_p = XLACustomOp(
    'event_csrmv',
    cpu_kernel_or_generator=event_csrmv_cpu_kernel_generator,
    gpu_kernel_or_generator=event_csrmv_gpu_kernel_generator,
)
event_csrmv_p.defjvp(event_csrmv_jvp_weights, None, None, event_csrmv_jvp_v)
event_csrmv_p.def_transpose_rule(event_csrmv_transpose_rule)
event_csrmv_p.def_batching_rule(event_csrmv_batching)


def event_csrmv_p_call(
    weights,
    indices,
    indptr,
    v,
    *,
    shape: Shape,
    transpose: bool,
    float_as_event: bool,
    block_size: int = 64
):
    if jax.default_backend() != 'cpu':
        weights = jnp.atleast_1d(weights)
    return event_csrmv_p(
        weights,
        indices,
        indptr,
        v,
        outs=[
            jax.ShapeDtypeStruct([shape[1]], weights.dtype)
            if transpose else
            jax.ShapeDtypeStruct([shape[0]], weights.dtype),
        ],
        # block_size=block_size,
        float_as_event=float_as_event,
        shape=shape,
        transpose=transpose,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        spike_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        block_size=block_size,
    )
    # else:
    #     return [
    #         csr_matvec(
    #             weights,
    #             indices,
    #             indptr,
    #             v,
    #             shape=shape,
    #             transpose=transpose
    #         )
    #     ]


def event_csrmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = event_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, batch_size, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = event_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 2):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = event_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        r = jnp.reshape(r[0], [r[0].shape[0], n, batch_size])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven CSR matrix-vector product.")


def event_csrmm_cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            # csr.T @ B

            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True, parallel=False)
                def mv(weights, indices, indptr, B, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

            elif float_as_event:
                @numba.njit(fastmath=True, parallel=False)
                def mv(weights, indices, indptr, B, posts):
                    posts[:] = 0.
                    B = B != 0.
                    w = weights[()]
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

            else:
                @numba.njit(fastmath=True, parallel=False)
                def mv(weights, indices, indptr, B, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            sp = B[i, k]
                            if sp != 0.:
                                wsp = w * sp
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += wsp

        else:
            # csr @ B
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, B, posts):
                    w = weights[()]
                    for i in range(indptr.shape[0] - 1):
                        r = np.zeros(B.shape[1], dtype=weights.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            index = indices[j]
                            for k in range(B.shape[1]):
                                if B[index, k]:
                                    r[k] += w
                        posts[i] = r

            elif float_as_event:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, B, posts):
                    w = weights[()]
                    B = B != 0.
                    for i in range(indptr.shape[0] - 1):
                        r = np.zeros(B.shape[1], dtype=weights.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            index = indices[j]
                            for k in range(B.shape[1]):
                                if B[index, k]:
                                    r[k] += w
                        posts[i] = r

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, B, posts):
                    w = weights[()]
                    for i in range(indptr.shape[0] - 1):
                        for k in range(B.shape[1]):
                            r = 0.
                            for j in range(indptr[i], indptr[i + 1]):
                                c = B[indices[j], k]
                                if c != 0.:
                                    r += w * c
                            posts[i, k] = r

    else:
        if transpose:
            # csr.T @ B

            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True, parallel=False)
                def mv(weights, indices, indptr, B, posts):
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

            elif float_as_event:
                @numba.njit(fastmath=True, parallel=False)
                def mv(weights, indices, indptr, B, posts):
                    posts[:] = 0.
                    B = B != 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

            else:
                @numba.njit(fastmath=True, parallel=False)
                def mv(weights, indices, indptr, B, posts):
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            sp = B[i, k]
                            if sp != 0.:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j] * sp

        else:
            # csr @ B

            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, B, posts):
                    for i in range(indptr.shape[0] - 1):
                        for k in range(B.shape[1]):
                            r = 0.
                            for j in range(indptr[i], indptr[i + 1]):
                                if B[indices[j], k]:
                                    r += weights[j]
                            posts[i, k] = r

            elif float_as_event:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, B, posts):
                    B = B != 0.
                    for i in range(indptr.shape[0] - 1):
                        for k in range(B.shape[1]):
                            r = 0.
                            for j in range(indptr[i], indptr[i + 1]):
                                if B[indices[j], k]:
                                    r += weights[j]
                            posts[i, k] = r

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, B, posts):
                    for i in range(indptr.shape[0] - 1):
                        for k in range(B.shape[1]):
                            r = 0.
                            for j in range(indptr[i], indptr[i + 1]):
                                c = B[indices[j], k]
                                if c != 0.:
                                    r += weights[j] * c
                            posts[i, k] = r

    return mv


def event_csrmm_gpu_kernel_generator(
    block_size: int,
    shape: Shape,
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    # TODO: Fix csrmm gpu kernel
    if transpose:
        # B @ csr   (B.shape[1], shape[0]) @ (shape[0], shape[1]) -> (B.shape[1], shape[1])
        if weight_info.size == 1:
            # 每个block处理一个[block_size, block_size]的output
            def _kernel(
                weights_ref,    # [num_nonzeros]
                indices_ref,    # [num_nonzeros]
                indptr_ref,     # [n_rows + 1], here is [block_size]
                B_ref,          # [B.shape[0], n_rows]
                posts_ref       # [B.shape[0], n_cols]
            ):
                weight_0 = weights_ref[0]
                
                # Get the row index in the output matrix
                row_k = pl.program_id(0)    # Row index in the output matrix
                col_i = pl.program_id(1)    # Col index in the output matrix
                
                def process_row(row_j, _):
                    # Check if the dense matrix element is non-zero
                    if spike_info.dtype == jnp.bool_:
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the CSR row
                            def process_element(j, _):
                                def true_fn_element(_):
                                    pl.atomic_add(posts_ref, (row_k, col_i), weight_0)
                                    
                                    return None
                                jax.lax.cond(
                                    indices_ref[j] == row_k,
                                    true_fn_element,
                                    lambda _: None,
                                    None
                                )
                                return None
                            
                            # Iterate over the non-zero elements in the CSR row
                            jax.lax.fori_loop(indptr_ref[row_j], indptr_ref[row_j + 1], process_element, None)
                        
                        jax.lax.cond(
                            B_ref[row_j, col_i],
                            true_fn,
                            lambda _: None,
                            None
                        )
                    elif float_as_event:
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the CSR row
                            def process_element(j, _):
                                def true_fn_element(_):
                                    pl.atomic_add(posts_ref, (row_k, col_i), weight_0)
                                    
                                    return None
                                jax.lax.cond(
                                    indices_ref[j] == row_k,
                                    true_fn_element,
                                    lambda _: None,
                                    None
                                )
                                return None
                            
                            # Iterate over the non-zero elements in the CSR row
                            jax.lax.fori_loop(indptr_ref[row_j], indptr_ref[row_j + 1], process_element, None)
                        
                        jax.lax.cond(
                            B_ref[row_j, col_i] != 0.,
                            true_fn,
                            lambda _: None,
                            None
                        )
                    else:
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the CSR row
                            def process_element(j, _):
                                def true_fn_element(_):
                                    pl.atomic_add(posts_ref, (row_k, col_i), weight_0 * B_ref[row_j, col_i])
                                    
                                    return None
                                jax.lax.cond(
                                    indices_ref[j] == row_k,
                                    true_fn_element,
                                    lambda _: None,
                                    None
                                )
                                return None
                            
                            # Iterate over the non-zero elements in the CSR row
                            jax.lax.fori_loop(indptr_ref[row_j], indptr_ref[row_j + 1], process_element, None)
                        
                        jax.lax.cond(
                            B_ref[row_j, col_i] != 0.,
                            true_fn,
                            lambda _: None,
                            None
                        )
                            
                # Iterate over the rows of the dense matrix
                jax.lax.fori_loop(0, spike_info.shape[0], process_row, None)
            
            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct([shape[1], spike_info.shape[1]], weight_info.dtype),
                ],
                out_specs=[
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),
                ],
                in_specs=[
                    pl.BlockSpec((weight_info.size,), lambda i, j: j),
                    pl.BlockSpec((indices_info.size,), lambda i, j: j),
                    pl.BlockSpec((shape[0] + 1,), lambda i, j: j),
                    pl.BlockSpec((spike_info.shape[0], spike_info.shape[1]), lambda i, j: (i, j))
                ],
                grid=(
                    pl.cdiv(shape[1], block_size),
                    pl.cdiv(spike_info.shape[1], block_size),
                ),
                interpret=False,
            )
        else:
            # 每个block处理一个[block_size, block_size]的output
            def _kernel(
                weights_ref,    # [num_nonzeros]
                indices_ref,    # [num_nonzeros]
                indptr_ref,     # [n_rows + 1], here is [block_size]
                B_ref,          # [B.shape[0], n_rows]
                posts_ref       # [B.shape[0], n_cols]
            ):
                
                # Get the row index in the output matrix
                row_k = pl.program_id(0)    # Row index in the output matrix
                col_i = pl.program_id(1)    # Col index in the output matrix
                
                def process_row(row_j, _):
                    # Check if the dense matrix element is non-zero
                    if spike_info.dtype == jnp.bool_:
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the CSR row
                            def process_element(j, _):
                                def true_fn_element(_):
                                    pl.atomic_add(posts_ref, (row_k, col_i), weights_ref[j])
                                    
                                    return None
                                jax.lax.cond(
                                    indices_ref[j] == row_k,
                                    true_fn_element,
                                    lambda _: None,
                                    None
                                )
                                return None
                            
                            # Iterate over the non-zero elements in the CSR row
                            jax.lax.fori_loop(indptr_ref[row_j], indptr_ref[row_j + 1], process_element, None)
                        
                        jax.lax.cond(
                            B_ref[row_j, col_i],
                            true_fn,
                            lambda _: None,
                            None
                        )
                    elif float_as_event:
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the CSR row
                            def process_element(j, _):
                                def true_fn_element(_):
                                    pl.atomic_add(posts_ref, (row_k, col_i), weights_ref[j])
                                    
                                    return None
                                jax.lax.cond(
                                    indices_ref[j] == row_k,
                                    true_fn_element,
                                    lambda _: None,
                                    None
                                )
                                return None
                            
                            # Iterate over the non-zero elements in the CSR row
                            jax.lax.fori_loop(indptr_ref[row_j], indptr_ref[row_j + 1], process_element, None)
                        
                        jax.lax.cond(
                            B_ref[row_j, col_i] != 0.,
                            true_fn,
                            lambda _: None,
                            None
                        )
                    else:
                        def true_fn(_):
                            # Define a function to process a single non-zero element in the CSR row
                            def process_element(j, _):
                                def true_fn_element(_):
                                    pl.atomic_add(posts_ref, (row_k, col_i), weights_ref[j] * B_ref[row_j, col_i])
                                    
                                    return None
                                jax.lax.cond(
                                    indices_ref[j] == row_k,
                                    true_fn_element,
                                    lambda _: None,
                                    None
                                )
                                return None
                            
                            # Iterate over the non-zero elements in the CSR row
                            jax.lax.fori_loop(indptr_ref[row_j], indptr_ref[row_j + 1], process_element, None)
                        
                        jax.lax.cond(
                            B_ref[row_j, col_i] != 0.,
                            true_fn,
                            lambda _: None,
                            None
                        )
                            
                # Iterate over the rows of the dense matrix
                jax.lax.fori_loop(0, spike_info.shape[0], process_row, None)
            
            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct([shape[1], spike_info.shape[1]], weight_info.dtype),
                ],
                out_specs=[
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),
                ],
                in_specs=[
                    pl.BlockSpec((weight_info.size,), lambda i, j: j),
                    pl.BlockSpec((indices_info.size,), lambda i, j: j),
                    pl.BlockSpec((shape[0] + 1,), lambda i, j: j),
                    pl.BlockSpec((spike_info.shape[0], spike_info.shape[1]), lambda i, j: (i, j))
                ],
                grid=(
                    pl.cdiv(shape[1], block_size),
                    pl.cdiv(spike_info.shape[1], block_size),
                ),
                interpret=False,
            )
        
    else:
        if weight_info.size == 1:
            def _kernel(
                weights_ref,    # [1] (scalar weight for homogeneous matrix)
                indices_ref,    # [num_nonzeros] (column indices of non-zero values)
                indptr_ref,     # [n_rows + 1] (row pointers)
                B_ref,          # [n_rows, n_cols] (input matrix)
                posts_ref       # [n_rows, n_cols] (output matrix)
            ):
                weight_0 = weights_ref[0]
                
                row_i = pl.program_id(0)
                col_k = pl.program_id(1)
                
                # Initialize the result for this row
                r = 0.0
                
                def true_fn(row_j, r_):
                    if spike_info.dtype == jnp.bool_:
                        r_ = jax.lax.cond(
                            B_ref[indices_ref[row_j], col_k],
                            lambda: r_ + 1.,
                            lambda: r_
                        )
                    elif float_as_event:
                        r_ = jax.lax.cond(
                            B_ref[indices_ref[row_j], col_k] != 0.,
                            lambda: r_ + 1.,
                            lambda: r_
                        )
                    else:
                        r_ = jax.lax.cond(
                            B_ref[indices_ref[row_j], col_k] != 0.,
                            lambda: r_ + B_ref[indices_ref[row_j], col_k],
                            lambda: r_,
                        )
                    return r_
                
                r = jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], true_fn, r) * weight_0
                posts_ref[row_i, col_k] = r
            
            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct([shape[0], spike_info.shape[1]], weight_info.dtype),
                ],
                out_specs=[
                    pl.BlockSpec([block_size, block_size], lambda i, j: (i, j)),
                ],
                in_specs=[
                    pl.BlockSpec((weight_info.size,), lambda i, j: j),
                    pl.BlockSpec((indices_info.size,), lambda i, j: j),
                    pl.BlockSpec((shape[0] + 1,), lambda i, j: j),
                    pl.BlockSpec(spike_info.shape, lambda i, j: (i, j))
                ],
                grid=(
                    pl.cdiv(shape[0], block_size),
                    pl.cdiv(spike_info.shape[1], block_size),
                ),
                interpret=False,
            )
        else:
            def _kernel(
                weights_ref,    # [num_nonzeros] (non-zero values)
                indices_ref,    # [num_nonzeros] (column indices of non-zero values)
                indptr_ref,     # [n_rows + 1] (row pointers)
                B_ref,          # [n_rows, n_cols] (input matrix)
                posts_ref       # [n_rows, n_cols] (output matrix)
            ):
                row_i = pl.program_id(0)
                col_k = pl.program_id(1)
                
                # Initialize the result for this row
                r = 0.0
                
                def true_fn(row_j, r_):
                    if spike_info.dtype == jnp.bool_:
                        r_ = jax.lax.cond(
                            B_ref[indices_ref[row_j], col_k],
                            lambda: r_ + weights_ref[row_j],
                            lambda: r_
                        )
                    elif float_as_event:
                        r_ = jax.lax.cond(
                            B_ref[indices_ref[row_j], col_k] != 0.,
                            lambda: r_ + weights_ref[row_j],
                            lambda: r_
                        )
                    else:
                        r_ = jax.lax.cond(
                            B_ref[indices_ref[row_j], col_k] != 0.,
                            lambda: r_ + weights_ref[row_j] * B_ref[indices_ref[row_j], col_k],
                            lambda: r_
                        )
                    return r_
                
                r = jax.lax.fori_loop(indptr_ref[row_i], indptr_ref[row_i + 1], true_fn, r)
                posts_ref[row_i, col_k] = r
                
            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct([shape[0], spike_info.shape[1]], weight_info.dtype),
                ],
                out_specs=[
                    pl.BlockSpec([block_size, block_size], lambda i, j: (i, j)),
                ],
                in_specs=[
                    pl.BlockSpec((weight_info.size,), lambda i, j: j),
                    pl.BlockSpec((indices_info.size,), lambda i, j: j),
                    pl.BlockSpec((shape[0] + 1,), lambda i, j: j),
                    pl.BlockSpec((spike_info.shape[0], spike_info.shape[1]), lambda i, j: (i, j))
                ],
                grid=(
                    pl.cdiv(shape[0], block_size),
                    pl.cdiv(spike_info.shape[1], block_size),
                ),
                interpret=False,
            )
    
    return kernel


event_csrmm_p = XLACustomOp(
    'event_csrmm',
    cpu_kernel_or_generator=event_csrmm_cpu_kernel_generator,
    gpu_kernel_or_generator=event_csrmm_gpu_kernel_generator,
)
event_csrmm_p.def_batching_rule(event_csrmm_batching)


def event_csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    *,
    shape: Shape,
    transpose: bool,
    float_as_event: bool,
    block_size: int = 64
):
    if jax.default_backend() != 'cpu':
        weights = jnp.atleast_1d(weights)
    return event_csrmm_p(
        weights,
        indices,
        indptr,
        B,
        outs=[
            jax.ShapeDtypeStruct([shape[1], B.shape[1]], weights.dtype)
            if transpose else
            jax.ShapeDtypeStruct([shape[0], B.shape[1]], weights.dtype),
        ],
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        spike_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        block_size=block_size,
    )
    # else:
    #     return [
    #         csr_matmat(
    #             weights,
    #             indices,
    #             indptr,
    #             B,
    #             shape=shape,
    #             transpose=transpose
    #         )
    #     ]
