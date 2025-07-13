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

from typing import Callable, Union, Optional

import brainunit as u
import jax.numpy as jnp

from brainstate import init, functional
from brainstate._state import ParamState
from brainstate.typing import ArrayLike, Size
from ._module import Module

__all__ = [
    'Linear',
    'ScaledWSLinear',
    'SignedWLinear',
    'SparseLinear',
    'AllToAll',
    'OneToOne',
    'LoRA',
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
        param_type: type = ParamState,
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
        self.weight = param_type(params)

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
        param_type: type = ParamState,
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
        self.weight = param_type(weight)

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
        param_type: type = ParamState,
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
        self.weight = param_type(params)

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


class SparseLinear(Module):
    """
    Linear layer with Sparse Matrix (can be ``brainunit.sparse.CSR``,
    ``brainunit.sparse.CSC``, ``brainunit.sparse.COO``, or any other sparse matrix).

    Args:
        spar_mat: SparseMatrix. The sparse weight matrix.
        in_size: Size. The input size.
        name: str. The object name.
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        spar_mat: u.sparse.SparseMatrix,
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        in_size: Size = None,
        name: Optional[str] = None,
        param_type: type = ParamState,
    ):
        super().__init__(name=name)

        # input and output shape
        if in_size is not None:
            self.in_size = in_size
        self.out_size = spar_mat.shape[-1]
        if in_size is not None:
            assert self.in_size[:-1] == self.out_size[:-1], (
                'The first n-1 dimensions of "in_size" '
                'and "out_size" must be the same.'
            )

        # weights
        assert isinstance(spar_mat, u.sparse.SparseMatrix), '"weight" must be a SparseMatrix.'
        self.spar_mat = spar_mat
        params = dict(weight=spar_mat.data)
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)
        self.weight = param_type(params)

    def update(self, x):
        data = self.weight.value['weight']
        y = x @ self.spar_mat.with_data(data)
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
        param_type: type = ParamState,
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
        self.weight = param_type(params)

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
        param_type: type = ParamState,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = in_size

        # weights
        param = dict(weight=init.param(w_init, self.in_size, allow_none=False))
        if b_init is not None:
            param['bias'] = init.param(b_init, self.out_size, allow_none=False)
        self.weight = param_type(param)

    def update(self, pre_val):
        post_val = pre_val * self.weight.value['weight']
        if 'bias' in self.weight.value:
            post_val = post_val + self.weight.value['bias']
        return post_val


class LoRA(Module):
    """A standalone LoRA layer.

    Example usage::

        >>> import brainstate as brainstate
        >>> import jax, jax.numpy as jnp
        >>> layer = brainstate.nn.LoRA(3, 2, 4)
        >>> layer.weight.value
    {'lora_a': Array([[ 0.25141352, -0.09826107],
            [ 0.2328382 ,  0.38869813],
            [ 0.27069277,  0.7678282 ]], dtype=float32),
     'lora_b': Array([[-0.8372317 ,  0.21012013, -0.52999765, -0.31939325],
            [ 0.64234126, -0.42980042,  1.2549229 , -0.47134295]],      dtype=float32)}
        >>> # Wrap around existing layer
        >>> linear = brainstate.nn.Linear(3, 4)
        >>> wrapper = brainstate.nn.LoRA(3, 2, 4, base_module=linear)
        >>> assert wrapper.base_module == linear
        >>> y = layer(jnp.ones((16, 3)))
        >>> y.shape
        (16, 4)

    Args:
        in_features: the number of input features.
        lora_rank: the rank of the LoRA dimension.
        out_features: the number of output features.
        base_module: a base module to call and substitute, if possible.
        kernel_init: initializer function for the weight matrices.
        param_type: the type of the LoRA params.
    """

    def __init__(
        self,
        in_features: int,
        lora_rank: int,
        out_features: int,
        *,
        base_module: Optional[Module] = None,
        kernel_init: Union[Callable, ArrayLike] = init.LecunNormal(),
        param_type: type = ParamState,
    ):
        super().__init__()

        # input and output shape
        self.in_size = in_features
        self.out_size = out_features
        self.in_features = in_features
        self.out_features = out_features

        # others
        self.base_module = base_module

        # weights
        param = dict(
            lora_a=kernel_init((in_features, lora_rank)),
            lora_b=kernel_init((lora_rank, out_features))
        )
        self.weight = param_type(param)

    def __call__(self, x: ArrayLike):
        out = x @ self.weight.value['lora_a'] @ self.weight.value['lora_b']
        if self.base_module is not None:
            if not callable(self.base_module):
                raise ValueError('`self.base_module` must be callable.')
            out += self.base_module(x)
        return out
