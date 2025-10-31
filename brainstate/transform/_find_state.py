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

from typing import Callable, Dict, Hashable, Literal, Sequence, Any

from brainstate._state import State
from brainstate.util.filter import Filter, to_predicate
from ._make_jaxpr import StatefulFunction

__all__ = [
    'StateFinder'
]


class StateFinder:
    """
    Discover :class:`~brainstate.State` instances touched by a callable.

    ``StateFinder`` wraps a function in :class:`~brainstate.transform.StatefulFunction`
    and exposes the collection of states the function reads or writes. The finder
    can filter states by predicates, request only read or write states, and return
    the result in several convenient formats.

    Parameters
    ----------
    fn : callable
        Function whose state usage should be inspected.
    filter : Filter, optional
        Predicate (see :mod:`brainstate.util.filter`) used to select states.
    usage : {'all', 'read', 'write', 'both'}, default 'all'
        Portion of the state trace to return. ``'both'`` returns a mapping with
        separate read and write entries.
    return_type : {'dict', 'list', 'tuple'}, default 'dict'
        Controls the container type returned for the selected states. When
        ``usage='both'``, the same container type is used for the ``'read'`` and
        ``'write'`` entries.
    key_fn : callable, optional
        Callable ``key_fn(index, state)`` that produces dictionary keys when
        ``return_type='dict'``. Defaults to using the positional index so existing
        code continues to receive integer keys.

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import jax.numpy as jnp
        >>>
        >>> param = brainstate.ParamState(jnp.ones(()), name='weights')
        >>> bias = brainstate.ParamState(jnp.zeros(()), name='bias')
        >>>
        >>> def forward(x):
        ...     _ = bias.value  # read-only
        ...     param.value = param.value * x  # read + write
        ...     return param.value + bias.value
        >>>
        >>> finder = brainstate.transform.StateFinder(
        ...     forward,
        ...     filter=brainstate.ParamState,
        ...     usage='both',
        ...     key_fn=lambda i, st: st.name or i,
        ... )
        >>> finder(2.0)['write']  # doctest: +ELLIPSIS
        {'weights': ParamState(...}

    Notes
    -----
    The underlying :class:`StatefulFunction` is cached, so subsequent calls with
    compatible arguments will reuse the compiled trace.
    """
    __module__ = 'brainstate.transform'

    _VALID_USAGE: tuple[str, ...] = ('all', 'read', 'write', 'both')
    _VALID_RETURN_TYPE: tuple[str, ...] = ('dict', 'list', 'tuple')

    def __init__(
        self,
        fn: Callable,
        filter: Filter = None,
        *,
        usage: Literal['all', 'read', 'write', 'both'] = 'all',
        return_type: Literal['dict', 'list', 'tuple'] = 'dict',
        key_fn: Callable[[int, State], Hashable] | None = None,
    ) -> None:
        if usage not in self._VALID_USAGE:
            raise ValueError(f"Invalid usage '{usage}'. Expected one of {self._VALID_USAGE}.")
        if return_type not in self._VALID_RETURN_TYPE:
            raise ValueError(
                f"Invalid return_type '{return_type}'. Expected one of {self._VALID_RETURN_TYPE}."
            )

        self.fn = fn
        self._usage = usage
        self._return_type = return_type
        self._key_fn = key_fn if key_fn is not None else self._default_key_fn
        self._filter = to_predicate(filter) if filter is not None else None
        self.stateful_fn = StatefulFunction(self.fn)

    def __call__(self, *args, **kwargs):
        """
        Invoke :meth:`find` to retrieve states touched by ``fn``.
        """
        return self.find(*args, **kwargs)

    def find(self, *args, **kwargs):
        """
        Execute the wrapped function symbolically and return the selected states.

        Parameters
        ----------
        *args, **kwargs
            Arguments forwarded to ``fn`` to determine the state trace.

        Returns
        -------
        Any
            Container holding the requested states as configured by ``usage`` and
            ``return_type``.
        """
        if self._usage == 'both':
            read_states = self._collect_states('read', *args, **kwargs)
            write_states = self._collect_states('write', *args, **kwargs)
            return {
                'read': self._format_states(read_states),
                'write': self._format_states(write_states),
            }

        states = self._collect_states(self._usage, *args, **kwargs)
        return self._format_states(states)

    def _collect_states(self, usage: str, *args, **kwargs) -> Sequence[State]:
        usage_map = {
            'all': self.stateful_fn.get_states,
            'read': self.stateful_fn.get_read_states,
            'write': self.stateful_fn.get_write_states,
        }
        collector = usage_map.get(usage)
        if collector is None:
            raise ValueError(f"Unsupported usage '{usage}'.")
        states = list(collector(*args, **kwargs))
        if self._filter is not None:
            states = [st for st in states if self._filter(tuple(), st)]
        return states

    def _format_states(self, states: Sequence[State]):
        if self._return_type == 'list':
            return list(states)
        if self._return_type == 'tuple':
            return tuple(states)
        return self._states_to_dict(states)

    def _states_to_dict(self, states: Sequence[State]) -> Dict[Hashable, State]:
        result: Dict[Hashable, State] = {}
        used_keys: set[Hashable] = set()
        for idx, state in enumerate(states):
            key = self._key_fn(idx, state)
            key = self._ensure_hashable(key)
            key = self._ensure_unique_key(key, idx, state, used_keys)
            result[key] = state
            used_keys.add(key)
        return result

    @staticmethod
    def _default_key_fn(idx: int, state: State) -> Hashable:
        return idx

    @staticmethod
    def _ensure_hashable(key: Any) -> Hashable:
        if key is None:
            return None
        try:
            hash(key)
        except TypeError:
            return str(key)
        return key

    @staticmethod
    def _ensure_unique_key(key: Hashable, idx: int, state: State, used: set[Hashable]) -> Hashable:
        if key is None or key in used:
            base_name = getattr(state, 'name', None)
            base = base_name if base_name not in (None, '') else f"state_{idx}"
            candidate = base
            suffix = 1
            while candidate in used:
                candidate = f"{base}_{suffix}"
                suffix += 1
            return candidate
        return key

