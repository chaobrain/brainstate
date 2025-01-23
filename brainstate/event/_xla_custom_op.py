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

import ctypes
import dataclasses
import functools
import importlib.util
from typing import Callable, Sequence, Tuple, Protocol, Dict, Union

import jax
import numpy as np
from jax import tree_util
from jax.interpreters import xla, mlir, batching, ad
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from brainstate.typing import PyTree

if jax.__version_info__ < (0, 4, 35):
    from jax.lib import xla_client
else:
    import jax.extend as je

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Primitive
else:
    from jax.extend.core import Primitive

__all__ = [
    'XLACustomKernel',
    'NumbaKernelGenerator',
    'WarpKernelGenerator',
    'PallasKernelGenerator',
]

numba_installed = importlib.util.find_spec('numba') is not None
warp_installed = importlib.util.find_spec('warp') is not None
_warp_gpu_capsule = False

if warp_installed:
    import warp  # pylint: disable=import-error, import-outside-toplevel
    import warp.context  # pylint: disable=import-error, import-outside-toplevel
    import warp.types  # pylint: disable=import-error, import-outside-toplevel

# Holder for the custom callback to keep it alive.
_registered_warp_gpu_kernels = [None]
_registered_warp_gpu_kernel_to_id = {}

#                                         [void* pointer,
#                                          const char *name,
#                                          PyCapsule_Destructor destructor]
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


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

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class WarpKernelGenerator:
    """
    The Warp kernel generator.

    Args:
        generator: Callable. The function defines the computation on GPU backend.
            It can be a function to generate the Warp kernel.
        input_output_aliases: Dict[int, int]. The input-output aliases.

    """

    generator: Callable[..., Callable]
    dim: int | Sequence[int] | Callable[..., Sequence[int]] | Callable[..., int]
    input_output_aliases: Dict[int, int] | Callable[..., Dict[int, int]] | None = None
    block_dim: int | Callable[..., int] | None = None

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class PallasKernelGenerator:
    """
    The JAX Pallas kernel generator.

    Args:
        generator: Callable. The function defines the computation on GPU/TPU backend using JAX Pallas.
            See the `JAX Pallas documentation <https://jax.readthedocs.io/en/latest/pallas/quickstart.html>`_
            for more details .
    """

    generator: Callable[..., Callable]

    def __call__(self, *args, **kwargs):
        return self.generator(*args, **kwargs)


def defjvp(primitive: Union[Primitive, 'XLACustomKernel'], *jvp_rules):
    """
    Define JVP rules for any JAX primitive.

    This function is similar to ``jax.interpreters.ad.defjvp``.
    However, the JAX one only supports primitive with ``multiple_results=False``.
    ``brainpy.math.defjvp`` enables to define the independent JVP rule for
    each input parameter no matter ``multiple_results=False/True``.

    For examples, please see ``test_ad_support.py``.

    Args:
      primitive: Primitive, XLACustomOp.
      *jvp_rules: The JVP translation rule for each primal.
    """
    if isinstance(primitive, XLACustomKernel):
        primitive = primitive.primitive
    assert isinstance(primitive, Primitive), f'The primitive should be a JAX primitive. But we got {primitive}'
    if primitive.multiple_results:
        ad.primitive_jvps[primitive] = functools.partial(_standard_jvp, jvp_rules, primitive)
    else:
        ad.primitive_jvps[primitive] = functools.partial(ad.standard_jvp, jvp_rules, primitive)


def _standard_jvp(jvp_rules, primitive: Primitive, primals, tangents, **params):
    assert primitive.multiple_results
    val_out = tuple(primitive.bind(*primals, **params))
    tree = tree_util.tree_structure(val_out)
    tangents_out = []
    for rule, t in zip(jvp_rules, tangents):
        if rule is not None and type(t) is not ad.Zero:
            r = tuple(rule(t, *primals, **params))
            tangents_out.append(r)
            assert tree_util.tree_structure(r) == tree
    r = functools.reduce(
        _add_tangents,
        tangents_out,
        tree_util.tree_map(
            # compatible with JAX 0.4.34
            lambda a: (
                ad.Zero.from_primal_value(a)
                if jax.__version_info__ >= (0, 4, 34) else
                ad.Zero.from_value(a)
            ),
            val_out
        )
    )
    return val_out, r


def _add_tangents(xs, ys):
    return tree_util.tree_map(ad.add_tangents, xs, ys, is_leaf=lambda a: isinstance(a, ad.Zero))


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

    kernel = kernel_generator(**kwargs)
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


def _warp_gpu_custom_callback(stream, buffers, opaque, opaque_len):
    # The descriptor is the form
    # <kernel-id>|<launch-dims>|<arg-dims-list>|<block-dim>
    # Example:  42|16,32|16,32;100;16,32|256
    kernel_id_str, dim_str, args_str, block_dim_str = opaque.decode().split("|")

    # Get the kernel from the registry.
    kernel_id = int(kernel_id_str)
    kernel = _registered_warp_gpu_kernels[kernel_id]

    # Parse launch dimensions.
    dims = [int(d) for d in dim_str.split(",")]
    bounds = warp.types.launch_bounds_t(dims)
    block_dim = int(block_dim_str)

    # Parse arguments.
    arg_strings = args_str.split(";")
    num_args = len(arg_strings)
    assert num_args == len(kernel.adj.args), "Incorrect number of arguments"

    # First param is the launch bounds.
    kernel_params = (ctypes.c_void_p * (1 + num_args))()
    kernel_params[0] = ctypes.addressof(bounds)

    # Parse array descriptors.
    args = []
    for i in range(num_args):
        dtype = kernel.adj.args[i].type.dtype
        shape = [int(d) for d in arg_strings[i].split(",")]
        strides = warp.types.strides_from_shape(shape, dtype)

        arr = warp.types.array_t(buffers[i], 0, len(shape), shape, strides)
        args.append(arr)  # keep a reference
        arg_ptr = ctypes.addressof(arr)

        kernel_params[i + 1] = arg_ptr

    # Get current device.
    device = warp.device_from_jax(_get_jax_device())

    # Get kernel hooks.
    # Note: module was loaded during jit lowering.
    hooks = kernel.module.get_kernel_hooks(kernel, device)
    assert hooks.forward, "Failed to find kernel entry point"

    # Launch the kernel.
    warp.context.runtime.core.cuda_launch_kernel(
        device.context,
        hooks.forward,
        bounds.size,
        0,  # max_blocks
        block_dim,  # threads_per_block
        hooks.forward_smem_bytes,
        kernel_params,
        stream
    )


# Create python-land custom call target.
warp_gpu_CCALL_FUNC = ctypes.CFUNCTYPE(
    ctypes.c_voidp,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_char_p,
    ctypes.c_size_t
)
warp_gpu_cc_callback = warp_gpu_CCALL_FUNC(_warp_gpu_custom_callback)
warp_gpu_ccall_address = ctypes.cast(warp_gpu_cc_callback, ctypes.c_void_p)


def _warp_cpu_single_out_call(output_ptrs, input_ptrs):
    kernel_id = int(kernel_id_str)
    kernel = _registered_warp_gpu_kernels[kernel_id]

    num_args = len(kernel.adj.args)

    # First param is the launch bounds.
    kernel_params = (ctypes.c_void_p * num_args)()

    # Parse array descriptors.
    args = []
    for i in range(num_args):
        dtype = kernel.adj.args[i].type.dtype
        shape = [int(d) for d in arg_strings[i].split(",")]
        strides = warp.types.strides_from_shape(shape, dtype)

        arr = warp.types.array_t(input_ptrs[i], 0, len(shape), shape, strides)
        args.append(arr)  # keep a reference
        arg_ptr = ctypes.addressof(arr)
        kernel_params[i] = arg_ptr

    # compile the kernel #
    # ------------------ #

    # Get current device.
    device = warp.device_from_jax(_get_jax_device())
    # Get kernel hooks.
    # Note: module was loaded during jit lowering.
    hooks = kernel.module.get_kernel_hooks(kernel, device)
    assert hooks.forward, "Failed to find kernel entry point"

    # Launch the kernel.
    hooks.forward(*kernel_params)


warp_cpu_CCALL_FUNC_single_out = ctypes.CFUNCTYPE(
    ctypes.c_voidp,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
)
warp_cpu_callback_single_out = warp_cpu_CCALL_FUNC_single_out(_warp_cpu_single_out_call)
warp_cpu_ccall_address_single_out = ctypes.cast(warp_cpu_callback_single_out, ctypes.c_void_p)


def _warp_cpu_multiple_outs_call(output_ptrs, input_ptrs):
    # The descriptor is the form
    # <kernel-id>|<launch-dims>|<arg-dims-list>
    # Example:  42|16,32|16,32;100;16,32
    kernel_id_str, dim_str, args_str = opaque.decode().split("|")

    # Get the kernel from the registry.
    kernel_id = int(kernel_id_str)
    kernel = _registered_warp_gpu_kernels[kernel_id]

    # Parse launch dimensions.
    dims = [int(d) for d in dim_str.split(",")]
    bounds = warp.types.launch_bounds_t(dims)

    # Parse arguments.
    arg_strings = args_str.split(";")
    num_args = len(arg_strings)
    assert num_args == len(kernel.adj.args), "Incorrect number of arguments"

    # First param is the launch bounds.
    kernel_params = (ctypes.c_void_p * (1 + num_args))()
    kernel_params[0] = ctypes.addressof(bounds)

    # Parse array descriptors.
    args = []
    for i in range(num_args):
        dtype = kernel.adj.args[i].type.dtype
        shape = [int(d) for d in arg_strings[i].split(",")]
        strides = warp.types.strides_from_shape(shape, dtype)

        arr = warp.types.array_t(buffers[i], 0, len(shape), shape, strides)
        args.append(arr)  # keep a reference
        arg_ptr = ctypes.addressof(arr)

        kernel_params[i + 1] = arg_ptr

    # Get current device.
    device = warp.device_from_jax(_get_jax_device())

    # Get kernel hooks.
    # Note: module was loaded during jit lowering.
    hooks = kernel.module.get_kernel_hooks(kernel, device)
    assert hooks.forward, "Failed to find kernel entry point"

    # Launch the kernel.
    warp.context.runtime.core.cuda_launch_kernel(
        device.context,
        hooks.forward,
        bounds.size,
        0,  # max_blocks
        256,  # threads_per_block
        hooks.forward_smem_bytes,
        kernel_params,
        stream
    )


warp_cpu_CCALL_FUNC_multi_outs = ctypes.CFUNCTYPE(
    ctypes.c_voidp,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
)
warp_cpu_callback_multi_out = warp_cpu_CCALL_FUNC_multi_outs(_warp_cpu_multiple_outs_call)
warp_cpu_ccall_address_multi_out = ctypes.cast(warp_cpu_callback_multi_out, ctypes.c_void_p)


def _warp_gpu_register_capsule():
    global _warp_gpu_capsule
    if _warp_gpu_capsule:
        return

    _warp_gpu_capsule = True

    # Put the custom call into a capsule, as required by XLA.
    warp_PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
    warp_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    warp_PyCapsule_New.restype = ctypes.py_object
    warp_PyCapsule_New.argtypes = (
        ctypes.c_void_p,
        ctypes.c_char_p,
        warp_PyCapsule_Destructor
    )
    warp_capsule = warp_PyCapsule_New(
        warp_gpu_ccall_address.value,
        b"xla._CUSTOM_CALL_TARGET",
        warp_PyCapsule_Destructor(0)
    )

    # Register the callback in XLA.
    if jax.__version_info__ < (0, 4, 35):
        xla_client.register_custom_call_target("brainstate_warp_gpu_call", warp_capsule, platform="gpu")
    else:
        je.ffi.register_ffi_target('brainstate_warp_gpu_call', warp_capsule, platform="gpu", api_version=0)


def _register_warp_kernel(wp_kernel) -> int:
    if wp_kernel not in _registered_warp_gpu_kernel_to_id:
        id_ = len(_registered_warp_gpu_kernels)
        _registered_warp_gpu_kernels.append(wp_kernel)
        _registered_warp_gpu_kernel_to_id[wp_kernel] = id_
    else:
        id_ = _registered_warp_gpu_kernel_to_id[wp_kernel]
    return id_


def _warp_get_vecmat_shape(warp_type):
    if hasattr(warp_type, 'dtype'):
        if hasattr(warp_type.dtype, "_shape_"):
            return warp_type.dtype._shape_
    return []


def _warp_strip_vecmat_dimensions(warp_arg, actual_shape):
    shape = _warp_get_vecmat_shape(warp_arg.type)
    for i, s in enumerate(reversed(shape)):
        item = actual_shape[-i - 1]
        if s != item:
            raise Exception(f"The vector/matrix shape for argument {warp_arg.label} does not match")
    return actual_shape[: len(actual_shape) - len(shape)]


def _warp_collapse_into_leading_dimension(warp_arg, actual_shape):
    if len(actual_shape) < warp_arg.type.ndim:
        raise Exception(f"Argument {warp_arg.label} has too few non-matrix/vector dimensions")
    index_rest = len(actual_shape) - warp_arg.type.ndim + 1
    leading_size = functools.reduce(lambda x, y: x * y, actual_shape[:index_rest])
    return [leading_size] + actual_shape[index_rest:]


# Infer array dimensions from input type.
def _warp_infer_dimensions(warp_arg, actual_shape):
    actual_shape = _warp_strip_vecmat_dimensions(warp_arg, actual_shape)
    return _warp_collapse_into_leading_dimension(warp_arg, actual_shape)


def _get_jax_device():
    # check if jax.default_device() context manager is active
    device = jax.config.jax_default_device
    # if default device is not set, use first device
    if device is None:
        device = jax.local_devices()[0]
    return device


def _warp_base_type_is_compatible(warp_type, jax_ir_type):
    jax_ir_to_warp = {
        "f16": warp.float16,
        "f32": warp.float32,
        "f64": warp.float64,
        "i8": warp.int8,
        "i16": warp.int16,
        "i32": warp.int32,
        "i64": warp.int64,
        "ui8": warp.uint8,
        "ui16": warp.uint16,
        "ui32": warp.uint32,
        "ui64": warp.uint64,
        "b1": warp.bool,
        "i1": warp.bool,
    }
    expected_warp_type = jax_ir_to_warp.get(str(jax_ir_type))
    if expected_warp_type is not None:
        if hasattr(warp_type, "_wp_scalar_type_"):
            return warp_type._wp_scalar_type_ == expected_warp_type
        else:
            return warp_type == expected_warp_type
    else:
        raise TypeError(f"Invalid or unsupported data type: {jax_ir_type}")


def _warp_gpu_lowering(
    kernel_generator: WarpKernelGenerator,
    ctx,
    *args,
    **kwargs,
):
    if not warp_installed:
        raise ImportError('Warp is required to compile the GPU kernel for the custom operator.')
    _warp_gpu_register_capsule()

    wp_kernel: warp.context.Kernel = kernel_generator(**kwargs)
    kernel_id = _register_warp_kernel(wp_kernel)

    # TODO: This may not be necessary, but it is perhaps better not to be
    #       mucking with kernel loading while already running the workload.
    module = wp_kernel.module
    device = warp.device_from_jax(_get_jax_device())
    if not module.load(device):
        raise Exception("Could not load kernel on device")

    # ------------------
    # launch dimensions
    # ------------------
    warp_dims = kernel_generator.dim
    if isinstance(warp_dims, int):
        warp_dims = (warp_dims,)
    assert isinstance(warp_dims, (tuple, list)), (
        f"Invalid launch dimensions, expected "
        f"tuple or list, got {warp_dims}"
    )

    # ------------------
    # block dimensions
    # ------------------
    block_dim = kernel_generator.block_dim
    if callable(block_dim):
        block_dim = block_dim(**kwargs)
    if isinstance(block_dim, int):
        pass
    elif block_dim is None:
        block_dim = 256
    else:
        raise ValueError(
            f"Invalid block dimensions, expected "
            f"int, got {block_dim}"
        )

    # ------
    # inputs
    # ------
    # Figure out the types and shapes of the input arrays.
    arg_strings = []
    operand_layouts = []
    for actual, warg in zip(args, wp_kernel.adj.args):
        rtt = ir.RankedTensorType(actual.type)
        _warp_strip_vecmat_dimensions(warg, rtt.shape)
        if hasattr(warg.type, 'ndim'):
            if len(rtt.shape) < warg.type.ndim:
                raise Exception(f"Argument {warg.label} has too few non-matrix/vector dimensions")
        arg_strings.append(",".join([str(d) for d in rtt.shape]))
        operand_layouts.append(_shape_to_layout(rtt.shape))

    # ------------------
    # output information
    # ------------------
    # Figure out the types and shapes of the output arrays.
    outs = ctx.avals_out
    result_layouts, result_types = [], []
    for out in outs:
        arg_strings.append(",".join([str(d) for d in out.shape]))
        result_layouts.append(_shape_to_layout(out.shape))
        result_types.append(mlir.aval_to_ir_type(out))

    # Build opaque descriptor for callback.
    dims_str = ",".join([str(d) for d in warp_dims])
    args_str = ";".join(arg_strings)
    descriptor = f"{kernel_id}|{dims_str}|{args_str}|{block_dim}"

    # ---------------------
    # input_output_aliases
    # ---------------------

    input_output_aliases = kernel_generator.input_output_aliases
    if callable(input_output_aliases):
        input_output_aliases = input_output_aliases(**kwargs)

    # custom call
    out = custom_call(
        b"brainstate_warp_gpu_call",
        result_types=result_types,
        operands=args,
        backend_config=descriptor.encode("utf-8"),
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
        operand_output_aliases=input_output_aliases,
    ).results
    return out


def register_warp_mlir_gpu_translation_rule(
    primitive: Primitive,
    kernel_generator: WarpKernelGenerator,
):
    """
    Register the Warp GPU translation rule for the custom operator.

    Args:
        primitive: Primitive. The custom operator.
        kernel_generator: Callable. The function defines the computation on GPU backend.
            It can be a function to generate the Warp kernel.
    """
    # register the lowering rule
    mlir.register_lowering(
        primitive,
        functools.partial(_warp_gpu_lowering, kernel_generator),
        platform="gpu",
    )


def register_pallas_mlir_gpu_translation_rule(
    primitive: Primitive,
    kernel_generator: PallasKernelGenerator,
):
    """
    Register the JAX Pallas GPU translation rule for the custom operator.

    Args:
        primitive: Primitive. The custom operator.
        kernel_generator: Callable. The function defines the computation on GPU backend.
            It can be a function to generate the JAX Pallas kernel.
    """
    lower = mlir.lower_fun(
        lambda *args, **kwargs: kernel_generator(**kwargs)(*args),
        multiple_results=True
    )
    mlir.register_lowering(primitive, lower, platform='cuda')
    mlir.register_lowering(primitive, lower, platform='tpu')


class ShapeDtype(Protocol):

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...


class XLACustomKernel:
    """Creating a XLA custom call operator.

    Args:
        cpu_kernel: Callable. The function defines the computation on CPU backend.
            It can be a function to generate the Numba jitted kernel.
        gpu_kernel: Callable. The function defines the computation on GPU backend.
            It can be a function to generate the JAX Pallas kernel.
        batching_translation: Callable. The batching translation rule of JAX.
        jvp_translation: Callable. The JVP translation rule of JAX.
        transpose_translation: Callable. The transpose translation rule of JAX.
        name: str. The primitive name.
    """

    def __init__(
        self,
        name: str,
        cpu_kernel: NumbaKernelGenerator = None,
        gpu_kernel: PallasKernelGenerator | WarpKernelGenerator = None,
        batching_translation: Callable = None,
        jvp_translation: Callable = None,
        transpose_translation: Callable = None,
    ):
        # primitive
        self.primitive = Primitive(name)
        self.primitive.multiple_results = True

        # abstract evaluation
        self.primitive.def_impl(functools.partial(xla.apply_primitive, self.primitive))
        self.primitive.def_abstract_eval(self._abstract_eval)

        # cpu kernel
        if cpu_kernel is not None:
            self.def_cpu_kernel(cpu_kernel)
        if gpu_kernel is not None:
            self.def_gpu_kernel(gpu_kernel)

        # batching rule
        if batching_translation is not None:
            batching.primitive_batchers[self.primitive] = batching_translation

        # jvp rule
        if jvp_translation is not None:
            ad.primitive_jvps[self.primitive] = jvp_translation

        # transpose rule
        if transpose_translation is not None:
            ad.primitive_transposes[self.primitive] = transpose_translation

    def _abstract_eval(self, *ins, outs: Sequence[jax.core.ShapedArray], **kwargs):
        return tuple(outs)

    def call(
        self,
        *ins,
        outs: PyTree[ShapeDtype],
        **kwargs,
    ):
        """
        Call the custom operator.
        """
        return self.__call__(*ins, outs=outs, **kwargs, )

    def __call__(
        self,
        *ins,
        outs: PyTree[ShapeDtype],
        **kwargs,
    ):
        """
        Call the custom operator.
        """
        outs = jax.tree.map(_transform_to_shapedarray, outs)
        outs, tree_def = jax.tree.flatten(outs)
        r = self.primitive.bind(
            *ins,
            **kwargs,
            outs=tuple(outs),
        )
        assert len(r) == len(outs), 'The number of outputs does not match the expected.'
        return tree_def.unflatten(r)

    def def_cpu_kernel(self, kernel_generator: NumbaKernelGenerator):
        """
        Define the CPU kernel using Numba.
        """
        if not isinstance(kernel_generator, NumbaKernelGenerator):
            raise TypeError('The `kernel_generator` should be an instance of `NumbaKernel`.')
        register_numba_mlir_cpu_translation_rule(self.primitive, kernel_generator)

    def def_gpu_kernel(self, kernel_generator: PallasKernelGenerator | WarpKernelGenerator):
        """
        Define the GPU kernel using the JAX Pallas or Warp.
        """

        if isinstance(kernel_generator, PallasKernelGenerator):
            register_pallas_mlir_gpu_translation_rule(self.primitive, kernel_generator)

        elif isinstance(kernel_generator, WarpKernelGenerator):
            register_warp_mlir_gpu_translation_rule(self.primitive, kernel_generator)

        else:
            raise TypeError('The `kernel_generator` should be an instance of `PallasKernel` or `WarpKernel`.')

    def def_batching_rule(self, fun):
        """Define the batching rule.

        Args:
          fun: The batching rule.
        """
        batching.primitive_batchers[self.primitive] = fun

    def def_jvp_rule(self, fun):
        """Define the JVP rule.

        Args:
          fun: The JVP rule.
        """
        ad.primitive_jvps[self.primitive] = fun

    def defjvp(self, *jvp_rules):
        """
        Define the JVP rule. Similar to ``jax.interpreters.ad.defjvp``,
        but supports the Primitive with multiple results.

        Args:
          jvp_rules: The JVP rules.
        """
        defjvp(self.primitive, *jvp_rules)

    def def_transpose_rule(self, fun):
        """Define the transpose rule.

        Args:
          fun: The transpose rule.
        """
        ad.primitive_transposes[self.primitive] = fun

    def def_xla_translation(self, platform, fun):
        """Define the XLA translation rule.

        Args:
          platform: str. The computing platform.
          fun: The XLA translation rule.
        """
        xla.backend_specific_translations[platform][self.primitive] = fun

    def def_mlir_lowering(self, platform, fun):
        """
        Define the MLIR lowering rule.

        Args:
          platform: str. The computing platform.
          fun: The lowering rule.
        """
        mlir.register_lowering(self.primitive, fun, platform)


def _transform_to_shapedarray(a):
    return jax.core.ShapedArray(a.shape, a.dtype)
