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

import functools
from typing import Callable, Sequence, Tuple, Protocol, Union

import jax
import numpy as np
from jax import tree_util
from jax.interpreters import xla, mlir, batching, ad

from brainstate.typing import PyTree
from ._xla_custom_op_numba import NumbaKernelGenerator, register_numba_mlir_cpu_translation_rule
from ._xla_custom_op_pallas import PallasKernelGenerator, register_pallas_mlir_gpu_translation_rule
from ._xla_custom_op_warp import WarpKernelGenerator, register_warp_mlir_gpu_translation_rule

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Primitive
else:
    from jax.extend.core import Primitive

__all__ = [
    'XLACustomKernel',
]


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
