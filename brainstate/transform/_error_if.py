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

import functools
from functools import partial
from typing import Callable, Union

import jax

from brainstate._utils import set_module_as
from ._unvmap import unvmap

__all__ = [
    'jit_error_if',
]


def _err_jit_true_branch(err_fun, args, kwargs):
    jax.debug.callback(err_fun, *args, **kwargs)


def _err_jit_false_branch(args, kwargs):
    pass


def _error_msg(msg, *arg, **kwargs):
    if len(arg):
        msg = msg % arg
    if len(kwargs):
        msg = msg.format(**kwargs)
    raise ValueError(msg)


@set_module_as('brainstate.transform')
def jit_error_if(
    pred,
    error: Union[Callable, str],
    *err_args,
    **err_kwargs,
):
    """
    Check errors in a jit function.

    Parameters
    ----------
    pred : bool or Array
        The boolean prediction.
    error : callable or str
        The error function, which raise errors, or a string indicating the error message.
    *err_args
        The arguments which passed into the error function.
    **err_kwargs
        The keywords which passed into the error function.

    Examples
    --------
    It can give a function which receive arguments that passed from the JIT variables and raise errors.

    .. code-block:: python

        >>> def error(x):
        ...     raise ValueError(f'error {x}')
        >>> x = jax.random.uniform(jax.random.PRNGKey(0), (10,))
        >>> jit_error_if(x.sum() < 5., error, x)

    Or, it can be a simple string message.

    .. code-block:: python

        >>> x = jax.random.uniform(jax.random.PRNGKey(0), (10,))
        >>> jit_error_if(x.sum() < 5., "Error: the sum is less than 5. Got {s}", s=x.sum())
    """
    if isinstance(error, str):
        error = partial(_error_msg, error)

    jax.lax.cond(
        unvmap(pred, op='any'),
        partial(_err_jit_true_branch, error),
        _err_jit_false_branch,
        jax.tree.map(functools.partial(unvmap, op='none'), err_args),
        jax.tree.map(functools.partial(unvmap, op='none'), err_kwargs),
    )
