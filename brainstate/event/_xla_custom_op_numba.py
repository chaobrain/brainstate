# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import ctypes
import dataclasses
import functools
import importlib.util
import threading
from contextlib import contextmanager
from typing import Callable, Dict

import jax
from jax.interpreters import mlir
from jaxlib.hlo_helpers import custom_call

if jax.__version_info__ < (0, 4, 35):
    from jax.lib import xla_client
else:
    import jax.extend as je

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Primitive
else:
    from jax.extend.core import Primitive

__all__ = [
    'NumbaKernelGenerator',
    'set_numba_environ',
]

numba_installed = importlib.util.find_spec('numba') is not None


class EnvironContext(threading.local):
    def __init__(self):
        # default environment settings
        self.numba_parallel: bool = False
        self.numba_setting: dict = dict(nogil=True, fastmath=True)


numba_environ = EnvironContext()


@contextmanager
def set_numba_environ(
    parallel_if_possible: int | bool = None,
    **kwargs
) -> None:
    """
    Enable Numba parallel execution if possible.
    """
    old_parallel = numba_environ.numba_parallel
    old_setting = numba_environ.numba_setting.copy()

    try:
        numba_environ.numba_setting.update(kwargs)
        if parallel_if_possible is not None:
            if isinstance(parallel_if_possible, bool):
                numba_environ.numba_parallel = parallel_if_possible
            elif isinstance(parallel_if_possible, int):
                numba_environ.numba_parallel = True
                assert parallel_if_possible > 0, 'The number of threads must be a positive integer.'
                import numba  # pylint: disable=import-outside-toplevel
                numba.set_num_threads(parallel_if_possible)
            else:
                raise ValueError('The argument `parallel_if_possible` must be a boolean or an integer.')
    finally:
        numba_environ.numba_parallel = old_parallel
        numba_environ.numba_setting = old_setting


@dataclasses.dataclass(frozen=True)
class NumbaKernelGenerator:
    """
    The Numba kernel generator.

    Args:
        generator: Callable. The function defines the computation on CPU backend.
            It can be a function to generate the Numba jitted kernel.
        input_output_aliases: Dict[int, int]. The input-output aliases.
    """

    generator: Callable[..., Callable]
    input_output_aliases: Dict[int, int] = None

    def generate_kernel(self, **kwargs):
        return self.generator(**kwargs)


#                                         [void* pointer,
#                                          const char *name,
#                                          PyCapsule_Destructor destructor]
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def _shape_to_layout(shape):
    return tuple(range(len(shape) - 1, -1, -1))


def _numba_mlir_cpu_translation_rule(
    kernel_generator: NumbaKernelGenerator,
    debug: bool,
    ctx,
    *ins,
    **kwargs
):
    if not numba_installed:
        raise ImportError('Numba is required to compile the CPU kernel for the custom operator.')

    from numba import types, carray, cfunc  # pylint: disable=import-error
    from numba.core.dispatcher import Dispatcher  # pylint: disable=import-error

    kernel = kernel_generator.generate_kernel(**kwargs)
    assert isinstance(kernel, Dispatcher), f'The kernel should be a Numba dispatcher. But we got {kernel}'

    # output information
    outs = ctx.avals_out
    output_shapes = tuple([out.shape for out in outs])
    output_dtypes = tuple([out.dtype for out in outs])
    output_layouts = tuple([_shape_to_layout(out.shape) for out in outs])
    result_types = [mlir.aval_to_ir_type(out) for out in outs]

    # input information
    avals_in = ctx.avals_in
    input_layouts = [_shape_to_layout(a.shape) for a in avals_in]
    input_dtypes = tuple(inp.dtype for inp in avals_in)
    input_shapes = tuple(inp.shape for inp in avals_in)

    # compiling function
    code_scope = dict(
        func_to_call=kernel,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        carray=carray
    )
    args_in = [f'in{i} = carray(input_ptrs[{i}], input_shapes[{i}], dtype=input_dtypes[{i}])'
               for i in range(len(input_shapes))]
    if len(output_shapes) > 1:
        args_out = [f'out{i} = carray(output_ptrs[{i}], output_shapes[{i}], dtype=output_dtypes[{i}])'
                    for i in range(len(output_shapes))]
        sig = types.void(types.CPointer(types.voidptr), types.CPointer(types.voidptr))
    else:
        args_out = [f'out0 = carray(output_ptrs, output_shapes[0], dtype=output_dtypes[0])']
        sig = types.void(types.voidptr, types.CPointer(types.voidptr))
    args_call = [f'in{i}' for i in range(len(input_shapes))] + [f'out{i}' for i in range(len(output_shapes))]
    code_string = '''
def numba_cpu_custom_call_target(output_ptrs, input_ptrs):
    {args_in}
    {args_out}
    func_to_call({args_call})
      '''.format(args_in="\n    ".join(args_in),
                 args_out="\n    ".join(args_out),
                 args_call=", ".join(args_call))
    if debug:
        print(code_string)
    exec(compile(code_string.strip(), '', 'exec'), code_scope)
    new_f = code_scope['numba_cpu_custom_call_target']

    # register
    xla_c_rule = cfunc(sig)(new_f)
    target_name = f'numba_custom_call_{str(xla_c_rule.address)}'
    capsule = ctypes.pythonapi.PyCapsule_New(xla_c_rule.address, b"xla._CUSTOM_CALL_TARGET", None)
    if jax.__version_info__ < (0, 4, 35):
        xla_client.register_custom_call_target(target_name, capsule, "cpu")
    else:
        je.ffi.register_ffi_target(target_name, capsule, "cpu", api_version=0)

    # call
    return custom_call(
        call_target_name=target_name,
        operands=ins,
        operand_layouts=list(input_layouts),
        result_layouts=list(output_layouts),
        result_types=list(result_types),
        has_side_effect=False,
        operand_output_aliases=kernel_generator.input_output_aliases,
    ).results


def register_numba_mlir_cpu_translation_rule(
    primitive: Primitive,
    cpu_kernel: NumbaKernelGenerator,
    debug: bool = False
):
    """
    Register the Numba CPU translation rule for the custom operator.

    Args:
        primitive: Primitive. The custom operator.
        cpu_kernel: Callable. The function defines the computation on CPU backend.
            It can be a function to generate the Numba jitted kernel.
        debug: bool. Whether to print the generated code.
    """
    rule = functools.partial(_numba_mlir_cpu_translation_rule, cpu_kernel, debug)
    mlir.register_lowering(primitive, rule, platform='cpu')
