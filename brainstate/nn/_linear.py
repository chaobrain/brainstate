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

# -*- coding: utf-8 -*-

from typing import Callable, Union, Optional

import brainunit as u
import jax.numpy as jnp

from brainstate._state import ParamState
from brainstate.typing import ArrayLike, Size
from . import init as init
from ._module import Module
from ._normalizations import weight_standardization

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
    Linear transformation layer.

    Applies a linear transformation to the incoming data: :math:`y = xW + b`

    Parameters
    ----------
    in_size : int or tuple of int
        The input feature size.
    out_size : int or tuple of int
        The output feature size.
    w_init : Callable or ArrayLike, optional
        Weight initializer. Default is ``KaimingNormal()``.
    b_init : Callable, ArrayLike, or None, optional
        Bias initializer. If ``None``, no bias is added. Default is ``ZeroInit()``.
    w_mask : ArrayLike, Callable, or None, optional
        Optional mask for the weights. If provided, weights will be element-wise
        multiplied by this mask.
    name : str, optional
        Name of the module.
    param_type : type, optional
        Type of parameter state. Default is ``ParamState``.

    Attributes
    ----------
    in_size : tuple
        Input feature size.
    out_size : tuple
        Output feature size.
    w_mask : ArrayLike or None
        Weight mask if provided.
    weight : ParamState
        Parameter state containing 'weight' and optionally 'bias'.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a linear layer
        >>> layer = brainstate.nn.Linear((10,), (5,))
        >>> x = jnp.ones((32, 10))
        >>> y = layer(x)
        >>> y.shape
        (32, 5)
        >>>
        >>> # Linear layer without bias
        >>> layer = brainstate.nn.Linear((10,), (5,), b_init=None)
        >>> y = layer(x)
        >>> y.shape
        (32, 5)
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
    Linear layer with signed absolute weights.

    This layer uses absolute values of weights multiplied by a sign matrix,
    ensuring all effective weights have controlled signs.

    Parameters
    ----------
    in_size : int or tuple of int
        The input feature size.
    out_size : int or tuple of int
        The output feature size.
    w_init : Callable or ArrayLike, optional
        Weight initializer. Default is ``KaimingNormal()``.
    w_sign : ArrayLike or None, optional
        Sign matrix for the weights. If ``None``, all weights are positive
        (absolute values used). If provided, should have the same shape as
        the weight matrix.
    name : str, optional
        Name of the module.
    param_type : type, optional
        Type of parameter state. Default is ``ParamState``.

    Attributes
    ----------
    in_size : tuple
        Input feature size.
    out_size : tuple
        Output feature size.
    w_sign : ArrayLike or None
        Sign matrix for weights.
    weight : ParamState
        Parameter state containing the weight values.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a signed weight linear layer with all positive weights
        >>> layer = brainstate.nn.SignedWLinear((10,), (5,))
        >>> x = jnp.ones((32, 10))
        >>> y = layer(x)
        >>> y.shape
        (32, 5)
        >>>
        >>> # With custom sign matrix (e.g., inhibitory connections)
        >>> w_sign = jnp.ones((10, 5)) * -1.0  # all negative
        >>> layer = brainstate.nn.SignedWLinear((10,), (5,), w_sign=w_sign)
        >>> y = layer(x)
        >>> y.shape
        (32, 5)
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
    Linear layer with weight standardization.

    Applies weight standardization [1]_ to normalize weights before the linear
    transformation, which can improve training stability and performance.

    Parameters
    ----------
    in_size : int or tuple of int
        The input feature size.
    out_size : int or tuple of int
        The output feature size.
    w_init : Callable, optional
        Weight initializer. Default is ``KaimingNormal()``.
    b_init : Callable, optional
        Bias initializer. Default is ``ZeroInit()``.
    w_mask : ArrayLike, Callable, or None, optional
        Optional mask for the weights.
    ws_gain : bool, optional
        Whether to use a learnable gain parameter for weight standardization.
        Default is ``True``.
    eps : float, optional
        Small constant for numerical stability in standardization.
        Default is ``1e-4``.
    name : str, optional
        Name of the module.
    param_type : type, optional
        Type of parameter state. Default is ``ParamState``.

    Attributes
    ----------
    in_size : tuple
        Input feature size.
    out_size : tuple
        Output feature size.
    w_mask : ArrayLike or None
        Weight mask if provided.
    eps : float
        Epsilon for numerical stability.
    weight : ParamState
        Parameter state containing 'weight', optionally 'bias' and 'gain'.

    References
    ----------
    .. [1] Qiao, S., Wang, H., Liu, C., Shen, W., & Yuille, A. (2019).
           Weight standardization. arXiv preprint arXiv:1903.10520.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a weight-standardized linear layer
        >>> layer = brainstate.nn.ScaledWSLinear((10,), (5,))
        >>> x = jnp.ones((32, 10))
        >>> y = layer(x)
        >>> y.shape
        (32, 5)
        >>>
        >>> # Without learnable gain
        >>> layer = brainstate.nn.ScaledWSLinear((10,), (5,), ws_gain=False)
        >>> y = layer(x)
        >>> y.shape
        (32, 5)
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
        w = weight_standardization(w, self.eps, params.get('gain', None))
        if self.w_mask is not None:
            w = w * self.w_mask
        y = u.linalg.dot(x, w)
        if 'bias' in params:
            y = y + params['bias']
        return y


class SparseLinear(Module):
    """
    Linear layer with sparse weight matrix.

    Supports sparse matrices from ``brainunit.sparse`` including CSR, CSC,
    and COO formats. Only the non-zero entries are stored and updated.

    Parameters
    ----------
    spar_mat : brainunit.sparse.SparseMatrix
        The sparse weight matrix defining the connectivity structure.
    b_init : Callable, ArrayLike, or None, optional
        Bias initializer. If ``None``, no bias is added.
    in_size : int or tuple of int, optional
        The input size. If not provided, inferred from ``spar_mat``.
    name : str, optional
        Name of the module.
    param_type : type, optional
        Type of parameter state. Default is ``ParamState``.

    Attributes
    ----------
    in_size : tuple
        Input feature size.
    out_size : int
        Output feature size.
    spar_mat : brainunit.sparse.SparseMatrix
        The sparse matrix structure.
    weight : ParamState
        Parameter state containing the sparse 'weight' data and optionally 'bias'.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a sparse linear layer with CSR matrix
        >>> indices = jnp.array([[0, 1], [1, 2], [2, 0]])
        >>> values = jnp.array([1.0, 2.0, 3.0])
        >>> spar_mat = u.sparse.CSR((values, indices[:, 1], indices[:, 0]),
        ...                          shape=(3, 3))
        >>> layer = brainstate.nn.SparseLinear(spar_mat, in_size=(3,))
        >>> x = jnp.ones((5, 3))
        >>> y = layer(x)
        >>> y.shape
        (5, 3)
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
    All-to-all connection layer.

    Performs matrix multiplication with optional exclusion of self-connections,
    commonly used in recurrent neural networks and graph neural networks.

    Parameters
    ----------
    in_size : int or tuple of int
        The number of neurons in the pre-synaptic group.
    out_size : int or tuple of int
        The number of neurons in the post-synaptic group.
    w_init : Callable or ArrayLike, optional
        Weight initializer. Default is ``KaimingNormal()``.
    b_init : Callable, ArrayLike, or None, optional
        Bias initializer. If ``None``, no bias is added.
    include_self : bool, optional
        Whether to include self-connections (diagonal elements).
        Default is ``True``.
    name : str, optional
        Name of the module.
    param_type : type, optional
        Type of parameter state. Default is ``ParamState``.

    Attributes
    ----------
    in_size : tuple
        Input size.
    out_size : tuple
        Output size.
    include_self : bool
        Whether self-connections are included.
    weight : ParamState
        Parameter state containing 'weight' and optionally 'bias'.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # All-to-all with self-connections
        >>> layer = brainstate.nn.AllToAll((10,), (10,), include_self=True)
        >>> x = jnp.ones((32, 10))
        >>> y = layer(x)
        >>> y.shape
        (32, 10)
        >>>
        >>> # All-to-all without self-connections (recurrent layer)
        >>> layer = brainstate.nn.AllToAll((10,), (10,), include_self=False)
        >>> y = layer(x)
        >>> y.shape
        (32, 10)
    """
    __module__ = 'brainstate.nn'

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
    One-to-one connection layer.

    Applies element-wise multiplication with a weight vector, implementing
    diagonal connectivity where each input unit connects only to its
    corresponding output unit.

    Parameters
    ----------
    in_size : int or tuple of int
        The number of neurons. Input and output sizes are the same.
    w_init : Callable or ArrayLike, optional
        Weight initializer. Default is ``Normal()``.
    b_init : Callable, ArrayLike, or None, optional
        Bias initializer. If ``None``, no bias is added.
    name : str, optional
        Name of the module.
    param_type : type, optional
        Type of parameter state. Default is ``ParamState``.

    Attributes
    ----------
    in_size : tuple
        Input size.
    out_size : tuple
        Output size (same as input size).
    weight : ParamState
        Parameter state containing 'weight' and optionally 'bias'.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # One-to-one connection
        >>> layer = brainstate.nn.OneToOne((10,))
        >>> x = jnp.ones((32, 10))
        >>> y = layer(x)
        >>> y.shape
        (32, 10)
        >>>
        >>> # With bias
        >>> layer = brainstate.nn.OneToOne((10,), b_init=braintools.init.Constant(0.1))
        >>> y = layer(x)
        >>> y.shape
        (32, 10)
    """
    __module__ = 'brainstate.nn'

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
    """
    Low-Rank Adaptation (LoRA) layer.

    Implements parameter-efficient fine-tuning using low-rank decomposition [1]_.
    Can be used standalone or as a wrapper around an existing module.

    Parameters
    ----------
    in_features : int
        The number of input features.
    lora_rank : int
        The rank of the low-rank decomposition. Lower rank means fewer parameters.
    out_features : int
        The number of output features.
    base_module : Module, optional
        A base module to wrap. If provided, the LoRA output will be added to
        the base module's output. Default is ``None``.
    kernel_init : Callable or ArrayLike, optional
        Initializer for the LoRA weight matrices. Default is ``LecunNormal()``.
    param_type : type, optional
        Type of parameter state. Default is ``ParamState``.

    Attributes
    ----------
    in_size : int
        Input feature size.
    out_size : int
        Output feature size.
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    base_module : Module or None
        The wrapped base module if provided.
    weight : ParamState
        Parameter state containing 'lora_a' and 'lora_b' matrices.

    References
    ----------
    .. [1] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
           Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large
           Language Models. arXiv preprint arXiv:2106.09685.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate as brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> # Standalone LoRA layer
        >>> layer = brainstate.nn.LoRA(in_features=10, lora_rank=2, out_features=5)
        >>> x = jnp.ones((32, 10))
        >>> y = layer(x)
        >>> y.shape
        (32, 5)
        >>>
        >>> # Wrap around existing linear layer
        >>> base = brainstate.nn.Linear((10,), (5,))
        >>> lora_layer = brainstate.nn.LoRA(in_features=10, lora_rank=2,
        ...                           out_features=5, base_module=base)
        >>> y = lora_layer(x)
        >>> y.shape
        (32, 5)
        >>>
        >>> # Check parameter count - LoRA has fewer parameters
        >>> # Base layer: 10 * 5 = 50 parameters
        >>> # LoRA: 10 * 2 + 2 * 5 = 30 parameters
    """
    __module__ = 'brainstate.nn'

    def __init__(
        self,
        in_features: int,
        lora_rank: int,
        out_features: int,
        *,
        base_module: Optional[Module] = None,
        kernel_init: Union[Callable, ArrayLike] = init.LecunNormal(),
        param_type: type = ParamState,
        in_size: Size = None,
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

        # in_size
        if in_size is not None:
            self.in_size = in_size
            self.out_size = tuple(self.in_size[:-1]) + (out_features,)

    def __call__(self, x: ArrayLike):
        out = x @ self.weight.value['lora_a'] @ self.weight.value['lora_b']
        if self.base_module is not None:
            if not callable(self.base_module):
                raise ValueError('`self.base_module` must be callable.')
            out += self.base_module(x)
        return out
