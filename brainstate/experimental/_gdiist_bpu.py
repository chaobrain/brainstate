# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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


from typing import Callable, Any

import jax
from jax.api_util import shaped_abstractify

from brainstate.transform._make_jaxpr import StatefulFunction, _make_hashable
from brainstate.util._cache import BoundedCache
from ._impl import register_jit_impl, register_forloop_impl
from .neuroir import compile_fn, CompiledGraphIR

__all__ = [
    'GdiistBPUParser',
]


class GdiistBPUParser:
    """Parser for BPU (second generation) operations and connections.
    
    This class is responsible for parsing the operations and connections in a BPU model.
    It provides comprehensive analysis capabilities including:
    
    - Operation and connection parsing
    - Statistics and metrics computation
    - Multiple display formats (text, summary, graph)
    - Export capabilities (dict, JSON)
    - Cache management
    """

    def __init__(
        self,
        fn: Callable,
        target: str = 'jit',
        cache_size: int = 128,
        jit_inline: bool = True,
        debug: bool = False,
    ):
        self.fn = fn
        self.debug = debug
        self.jit_inline = jit_inline
        self.stateful_fn = StatefulFunction(self.fn, ir_optimizations='dce')
        # self.stateful_fn = StatefulFunction(self.fn)
        if target not in ['jit', 'forloop']:
            raise ValueError(f"Target must be either 'jit' or 'forloop', got {target}")
        self.target = target
        self.compiled_graph = BoundedCache(maxsize=cache_size)

    def cache_key(self, *args, **kwargs) -> Any:
        if self.target == 'forloop':
            args, kwargs = jax.tree.map(lambda x: x[0], (args, kwargs))
        return _make_hashable(jax.tree.map(shaped_abstractify, (args, kwargs)))

    def compile(self, *args, verbose: bool = False, **kwargs) -> CompiledGraphIR:
        key = self.cache_key(*args, **kwargs)

        if key in self.compiled_graph:
            if verbose:
                print(f"Cache hit for key: {key}")
            compiled = self.compiled_graph.get(key)
        else:
            if verbose:
                print(f"Cache miss for key: {key}, parsing...")

            # Get the JAXpr and states from the stateful function
            parse_args, parse_kwargs = args, kwargs
            if self.target == 'forloop':
                parse_args, parse_kwargs = jax.tree.map(lambda x: x[0], (args, kwargs))

            # IR parsing
            compiled = compile_fn(self.stateful_fn, jit_inline=self.jit_inline)(*parse_args, **parse_kwargs)
            self.compiled_graph.set(key, compiled)

        return compiled

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self.compiled_graph.clear()

    def __call__(self, *args, **kwargs):
        compiled = self.compile(*args, **kwargs)
        if self.debug:
            return compiled.run_compiled(*args, **kwargs)
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        """String representation of the parser."""
        return (
            f"{self.__class__.__name__}("
            f"target='{self.target}', "
            f"cache_size={self.compiled_graph.maxsize}, "
            f"cached_graphs={len(self.compiled_graph)}"
            f")"
        )


def _jit_wrapper(fn, debug: bool = False, **kwargs):
    return GdiistBPUParser(fn, target='jit', debug=debug)


def _forloop_wrapper(fn, debug: bool = False, **kwargs):
    return GdiistBPUParser(fn, target='forloop', debug=debug)


register_jit_impl('bpu', _jit_wrapper)
register_forloop_impl('bpu', _forloop_wrapper)
