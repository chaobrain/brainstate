# -*- coding: utf-8 -*-

import ctypes
import functools
import importlib.util
from functools import partial
from typing import Callable, Sequence, Tuple, Protocol

import jax
import numpy as np
from jax import tree_util
from jax.core import Primitive
from jax.interpreters import batching, ad
from jax.interpreters import xla, mlir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

numba_installed = importlib.util.find_spec('numba') is not None

if numba_installed:
    import numba  # pylint: disable=import-error
    from numba import types, carray, cfunc  # pylint: disable=import-error
    from numba.core.dispatcher import Dispatcher  # pylint: disable=import-error
else:
    numba = None

__all__ = [
    'XLACustomOp',
]

#                                         [void* pointer,
#                                          const char *name,
#                                          PyCapsule_Destructor destructor]
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def defjvp(primitive, *jvp_rules):
    """Define JVP rules for any JAX primitive.

    This function is similar to ``jax.interpreters.ad.defjvp``.
    However, the JAX one only supports primitive with ``multiple_results=False``.
    ``brainpy.math.defjvp`` enables to define the independent JVP rule for
    each input parameter no matter ``multiple_results=False/True``.

    For examples, please see ``test_ad_support.py``.

    Args:
      primitive: Primitive, XLACustomOp.
      *jvp_rules: The JVP translation rule for each primal.
    """
    assert isinstance(primitive, Primitive)
    if primitive.multiple_results:
        ad.primitive_jvps[primitive] = partial(_standard_jvp, jvp_rules, primitive)
    else:
        ad.primitive_jvps[primitive] = partial(ad.standard_jvp, jvp_rules, primitive)


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
            lambda a: ad.Zero.from_primal_value(a) if jax.__version__ >= '0.4.34' else ad.Zero.from_value(a),
            val_out
        )
    )
    return val_out, r


def _add_tangents(xs, ys):
    return tree_util.tree_map(ad.add_tangents, xs, ys, is_leaf=lambda a: isinstance(a, ad.Zero))


def _shape_to_layout(shape):
    return tuple(range(len(shape) - 1, -1, -1))


def _numba_mlir_cpu_translation_rule(
    kernel,
    debug: bool,
    ctx,
    *ins,
    **kwargs
):
    if numba is None:
        raise ImportError('Numba is required to compile the CPU kernel for the custom operator.')

    if not isinstance(kernel, Dispatcher):
        kernel = kernel(**kwargs)
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
    code_scope = dict(func_to_call=kernel,
                      input_shapes=input_shapes,
                      input_dtypes=input_dtypes,
                      output_shapes=output_shapes,
                      output_dtypes=output_dtypes, carray=carray)
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
    xla_client.register_custom_call_target(target_name, capsule, "cpu")

    # call
    return custom_call(
        call_target_name=target_name,
        operands=ins,
        operand_layouts=list(input_layouts),
        result_layouts=list(output_layouts),
        result_types=list(result_types),
        has_side_effect=False,
    ).results


def register_numba_mlir_cpu_translation_rule(
    primitive: jax.core.Primitive,
    cpu_kernel: Callable,
    debug: bool = False
):
    rule = partial(_numba_mlir_cpu_translation_rule, cpu_kernel, debug)
    mlir.register_lowering(primitive, rule, platform='cpu')


class ShapeDtype(Protocol):

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...


class XLACustomOp:
    """Creating a XLA custom call operator.

    Args:
      cpu_kernel_generator: Callable. The function defines the computation on CPU backend.
      gpu_kernel_generator: Callable. The function defines the computation on GPU backend.
      batching_translation: Callable. The batching translation rule of JAX.
      jvp_translation: Callable. The JVP translation rule of JAX.
      transpose_translation: Callable. The transpose translation rule of JAX.
      name: str. The primitive name.
    """

    def __init__(
        self,
        name: str,
        cpu_kernel_generator: Callable,
        gpu_kernel_generator: Callable = None,
        batching_translation: Callable = None,
        jvp_translation: Callable = None,
        transpose_translation: Callable = None,
    ):
        # set cpu_kernel and gpu_kernel
        self.cpu_kernel = cpu_kernel_generator

        # primitive
        self.primitive = jax.core.Primitive(name)
        self.primitive.multiple_results = True

        # abstract evaluation
        self.primitive.def_impl(partial(xla.apply_primitive, self.primitive))

        # cpu kernel
        if cpu_kernel_generator is not None:
            self.def_cpu_kernel(cpu_kernel_generator)
        if gpu_kernel_generator is not None:
            self.def_gpu_kernel(gpu_kernel_generator)

        # batching rule
        if batching_translation is not None:
            batching.primitive_batchers[self.primitive] = batching_translation

        # jvp rule
        if jvp_translation is not None:
            ad.primitive_jvps[self.primitive] = jvp_translation

        # transpose rule
        if transpose_translation is not None:
            ad.primitive_transposes[self.primitive] = transpose_translation

    def __call__(self, *ins, outs: Sequence[ShapeDtype], **kwargs):
        assert isinstance(outs, (tuple, list)), 'The `outs` should be a tuple or list of shape-dtype pairs.'
        outs = jax.tree.map(_transform_to_shapedarray, outs)
        self.primitive.def_abstract_eval(functools.partial(_abstract_eval, outs))
        return self.primitive.bind(*ins, **kwargs)

    def def_cpu_kernel(self, kernel_generator: Callable):
        """
        Define the CPU kernel using Numba.
        """
        register_numba_mlir_cpu_translation_rule(self.primitive, kernel_generator)

    def def_gpu_kernel(self, kernel_generator: Callable):
        """
        Define the GPU kernel using the JAX Pallas language.
        """
        lower = mlir.lower_fun(
            lambda *args, **kwargs: kernel_generator(**kwargs)(*args),
            multiple_results=True
        )
        mlir.register_lowering(self.primitive, lower, platform='cuda')
        mlir.register_lowering(self.primitive, lower, platform='tpu')

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


def _abstract_eval(outs, *args, **kwargs):
    return [jax.core.ShapedArray(out.shape, out.dtype) for out in outs]


def _transform_to_shapedarray(a):
    return jax.core.ShapedArray(a.shape, a.dtype)
