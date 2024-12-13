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

import functools

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from brainstate.nn._module import Module

__all__ = [
    'BlockCSR',
    'BlockCSRLinear',
]


class BlockCSRLinear(Module):
    def __init__(
        self
    ):
        pass


@jax.tree_util.register_pytree_node_class
class BlockCSR:
    blocks: jax.Array  # float32[n_blocks, *block_size]
    indptr: jax.Array  # int32[n_block_rows + 1]
    indices: jax.Array  # int32[n_blocks]
    shape: tuple[int, int]  # (n_block_rows * block_size[0], n_block_cols * block_size[1])

    ndim: int = property(lambda self: len(self.shape))
    num_blocks = property(lambda self: self.blocks.shape[0])
    block_size = property(lambda self: self.blocks.shape[1:])
    dtype = property(lambda self: self.blocks.dtype)

    def __init__(self, blocks, indptr, indices, *, shape):
        self.blocks = blocks
        self.indptr = indptr
        self.indices = indices
        self.shape = shape

    def tree_flatten(self):
        return (self.blocks,), (self.indptr, self.indices, self.shape,)

    @classmethod
    def tree_unflatten(cls, data, xs):
        blocks, = xs
        indptr, indices, shape = data
        return BlockCSR(blocks, indptr, indices, shape=shape)

    def _validate(self):
        _nblocks, n, m = self.blocks.shape
        nrows = self.indptr.shape[0] - 1
        assert self.indices.shape[0] == _nblocks
        assert len(self.shape) == 2
        assert self.shape[0] == n * nrows
        assert self.shape[1] % m == 0

    @jax.jit
    def todense(self) -> jax.Array:
        self._validate()
        return _sdd_todense(self)

    @classmethod
    def fromdense(cls, dense: jax.Array, *, block_size) -> 'BlockCSR':
        raise NotImplementedError

    def __matmul__(self, other) -> jax.Array:
        self._validate()
        return sdd_matmul(self, other)


@jax.jit
def _sdd_todense(mat: BlockCSR) -> jax.Array:
    _, n, m = mat.blocks.shape
    nrows = mat.shape[0] // n
    unit = u.get_unit(mat.blocks)
    blocks = u.get_mantissa(mat.blocks)

    def i_body(i_row, out):  # each row
        def j_body(x):  # each block in the row
            i_block, val = x
            i_col = mat.indices[i_block]
            val = jax.lax.dynamic_update_slice(val, blocks[i_block], (i_row * n, i_col * m))
            return i_block + 1, val

        return jax.lax.while_loop(
            lambda x: x[0] < mat.indptr[i_row + 1],
            j_body,
            (mat.indptr[i_row], out)
        )[1]

    dense = jax.lax.fori_loop(0, nrows, i_body, jnp.zeros(mat.shape, mat.dtype))
    return u.maybe_decimal(u.Quantity(dense, unit=unit))


def _check_shape_consistency(x, y):
    assert isinstance(y, jax.Array), f"Only support jax.Array. But got unsupported type {type(y)}"
    assert x.ndim == y.ndim == 2
    assert x.shape[1] == y.shape[0], f"Dimension mismatch: {x.shape} @ {y.shape}"


def _sdd_kernel(
    x_ref,  # [n_blocks, bm, bn]
    indices_ref,  # [n_block]
    indptr_ref,  # [n_rows + 1]
    y_ref,  # [n, k]
    o_ref,  # [m, k]
    *,
    bm: int,
    bn: int,
    bk: int,
):
    i_m = pl.program_id(axis=0)
    i_k = pl.program_id(axis=1)
    i_start = indptr_ref[i_m]
    i_end = indptr_ref[i_m + 1]

    def body(x):
        val, i_block = x
        i_x_col = indices_ref[i_block]
        block = pl.load(x_ref, (i_block, pl.dslice(None), pl.dslice(None)))  # [bm, bn]
        chunk = pl.load(y_ref, (pl.dslice(i_x_col * bn, bn), pl.dslice(i_k * bk, bk)))  # [bn, bk]
        return val + jnp.dot(block, chunk).astype(o_ref.dtype), i_block + 1

    acc = jax.lax.while_loop(
        lambda x: x[1] < i_end,
        body,
        (jnp.zeros([bm, bk], dtype=o_ref.dtype), i_start)
    )[0]
    pl.store(o_ref, (pl.dslice(bm * i_m, bm), pl.dslice(bk * i_k, bk)), acc)  # [bm, bk]


@functools.partial(jax.jit, static_argnames=["debug", 'interpret', 'block_size'])
def sdd_matmul(
    mat1: BlockCSR,
    mat2: jax.Array,
    *,
    debug: bool = False,
    interpret: bool = False,
    block_size: int = 256,
) -> jax.Array:
    _check_shape_consistency(mat1, mat2)

    # shape and dtype
    m, n, k = mat1.shape[0], mat1.shape[1], mat2.shape[1]
    _, bm, bn = mat1.blocks.shape
    dtype = jnp.result_type(mat1.dtype, mat2.dtype)

    # kernel
    fn = pl.pallas_call(
        functools.partial(_sdd_kernel, bm=bm, bn=bn, bk=block_size),
        out_shape=jax.ShapeDtypeStruct(shape=(m, k), dtype=dtype),
        grid=(pl.cdiv(m, bm), pl.cdiv(k, block_size)),
        debug=debug,
        interpret=interpret
    )

    # call
    unita = u.get_unit(mat1.blocks)
    unitb = u.get_unit(mat2)
    blocks = u.get_mantissa(mat1.blocks)
    r = fn(blocks, mat1.indices, mat1.indptr, u.get_mantissa(mat2))
    return u.maybe_decimal(u.Quantity(r, unit=unita * unitb))


@jax.jit
def native_sdd_matmul(
    mat1: BlockCSR,
    mat2: jax.Array,
):
    _check_shape_consistency(mat1, mat2)

    dtype = jnp.result_type(mat1.dtype, mat2.dtype)
    _, n, m = mat1.blocks.shape

    nrows = mat1.shape[0] // n

    def i_body(i):  # each row
        def k_body(x):
            i_block, val = x
            i_col = mat1.indices[i_block]
            chunk = jax.lax.dynamic_slice(mat2, [i_col * m, 0], (m, mat2.shape[1]))  # [m, mat2.shape[1]]
            block = blocks[i_block]
            return i_block + 1, val + block.dot(chunk)

        acc = jax.lax.while_loop(
            lambda x: x[0] < mat1.indptr[i + 1],
            k_body,
            (mat1.indptr[i], jnp.zeros((n, mat2.shape[1]), dtype=jnp.float32))
        )[1]
        return acc.astype(dtype)

    unita = u.get_unit(mat1.blocks)
    unitb = u.get_unit(mat2)
    blocks = u.get_mantissa(mat1.blocks)
    mat2 = u.get_mantissa(mat2)

    out = jax.vmap(i_body)(jnp.arange(nrows)).reshape((mat1.shape[0], mat2.shape[1]))
    return u.maybe_decimal(u.Quantity(out, unit=unita * unitb))


def sample_sparse_matrix(
    m, n, bm, bn, *,
    sparse_prob=0.2,
    dtype=jnp.float32
) -> BlockCSR:
    num_rows = m // bm  # number of rows in the Block-ELL matrix
    num_cols = n // bn  # number of columns in the Block-ELL matrix
    blocks_per_row = np.random.binomial(num_cols, sparse_prob,
                                        size=[num_rows])  # [n_rows], number of blocks in each row
    num_blocks = blocks_per_row.sum()
    blocks = np.random.randn(num_blocks, bm, bn).astype(dtype)  # [n_blocks, bm, bk], block values

    # [n_rows + 1], row pointers
    indptr = np.zeros(num_rows + 1, dtype=np.int32)  # [n_rows + 1], row pointers
    indptr[1:] = np.cumsum(blocks_per_row)

    # [n_block], block indices
    indices = []
    for i in range(num_rows):
        indices.extend(np.random.choice(num_cols, blocks_per_row[i], replace=False))
    indices = jnp.array(indices)  # [n_rows, max_num_blocks_per_row, 2], block indices

    return BlockCSR(jnp.asarray(blocks), jnp.asarray(indptr), jnp.asarray(indices), shape=(m, n))
