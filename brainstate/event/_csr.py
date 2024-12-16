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

from typing import Callable

import brainunit as u
import jax
import jax.numpy as jnp
from brainunit.sparse.csr import _csr_matvec as csr_matvec, _csr_matmat as csr_matmat
from brainunit.sparse.csr import _csr_to_coo as csr_to_coo
from jax.experimental.sparse import JAXSparse
from jax.interpreters import ad

from brainstate.typing import Shape
from ._xla_custom_op import XLACustomOp

__all__ = [
    'CSR',
    'CSC',
]


class CSR(u.sparse.CSR):
    """
    Event-driven sparse matrix in CSR format.
    """

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


class CSC(u.sparse.CSC):
    """
    Event-driven sparse matrix in CSC format.
    """

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
    # res = csr_matvec_p.bind(data, indices, indptr, v, shape=shape, transpose=transpose)
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
    transpose: bool = False
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
    res = csr_matmat_p.bind(data, indices, indptr, B, shape=shape, transpose=transpose)
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
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

            elif float_as_event:
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[()]
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

            else:
                @numba.njit
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
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    w = weights[()]
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += w
                        posts[i] = r

            elif float_as_event:
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    w = weights[()]
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += w
                        posts[i] = r

            else:
                @numba.njit
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
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

            elif float_as_event:
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

            else:
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        sp = v[i]
                        if sp != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j] * sp

        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += weights[j]
                        posts[i] = r

            elif float_as_event:
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += weights[j]
                        posts[i] = r

            else:
                @numba.njit
                def mv(weights, indices, indptr, v, posts):
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            c = v[indices[j]]
                            if c != 0.:
                                r += weights[j] * c
                        posts[i] = r

    return mv


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
        return 0, event_csrmm_p_call(*args, **kwargs)
    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven CSR matrix-vector product.")


event_csrmv_p = XLACustomOp(
    'event_csrmv',
    cpu_kernel_or_generator=event_csrmv_cpu_kernel_generator,
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
    shape,
    transpose,
    float_as_event,
):
    if jax.default_backend() == 'cpu':
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
        )
    else:
        return [
            csr_matvec(
                weights,
                indices,
                indptr,
                v,
                shape=shape,
                transpose=transpose
            )
        ]


def event_csrmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        batch_shape = args[3].shape[:-1]
        B = jnp.reshape(args[3], (-1, args[3].shape[-1:]))
        r = event_csrmm_p_call(args[0], args[1], args[2], B, **kwargs)
        return 0, [jnp.reshape(r[0], batch_shape + r.shape[-1:])]
    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven CSR matrix-vector product.")


event_csrmm_p = XLACustomOp(
    'event_csrmm',
    cpu_kernel_or_generator=event_csrmv_cpu_kernel_generator,
)
event_csrmm_p.def_batching_rule(event_csrmm_batching)


def event_csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    *,
    shape,
    transpose,
    float_as_event,
):
    if jax.default_backend() == 'cpu':
        return event_csrmm_p(
            weights,
            indices,
            indptr,
            B,
            outs=[
                jax.ShapeDtypeStruct([shape[0], B.shape[1]], weights.dtype)
                if transpose else
                jax.ShapeDtypeStruct([shape[1], B.shape[1]], weights.dtype),
            ],
            # block_size=block_size,
            float_as_event=float_as_event,
            weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
            spike_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        )
    else:
        return [
            csr_matmat(
                weights,
                indices,
                indptr,
                B,
                shape=shape,
                transpose=transpose
            )
        ]
