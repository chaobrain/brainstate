"""
Hierarchical data containers for parameter and state management.

This module provides the ``HiData`` class, a flexible container for hierarchical
data structures that supports dictionary-like and attribute-style access,
cloning, serialization, and composition.
"""

import dataclasses
from typing import Any, Dict

import brainstate
from brainstate.util import PrettyAttr, PrettyRepr, PrettyType, format_array_like, pretty_repr_object
from brainstate.util.struct import dataclass, field

Array = brainstate.typing.ArrayLike

__all__ = [
    'HiData',
]


def is_dataclass(cls):
    return hasattr(cls, '_brainstate_dataclass')


@dataclass
class HiData(PrettyRepr):
    """
    Hierarchical state container for composed dynamics.

    Stores child states in a dictionary where keys match the attribute names
    of child dynamics in the parent dynamics class.

    Supports two initialization styles:
        - Data(children={'key1': data1, 'key2': data2})
        - Data(key1=data1, key2=data2)

    And two access styles:
        - cd['key1'] or cd.key1

    Attributes
    ----------
    children
        Dict mapping child names to their states.

    Examples
    --------
        Create a simple Data object:

        >>> data = HiData(name='config', learning_rate=0.01, batch_size=32)
        >>> print(data)
        HiData(name='config', learning_rate=0.01, batch_size=32)

        Create nested Data objects:

        >>> import numpy as np
        >>> optimizer = HiData(name='optimizer', lr=0.001, momentum=0.9)
        >>> model = HiData(name='model', weights=np.array([1, 2, 3]))
        >>> config = HiData(name='config', optimizer=optimizer, model=model)
        >>> print(config)
        HiData(
          name='config',
          optimizer=HiData(name='optimizer', lr=0.001, momentum=0.9),
          model=HiData(name='model', weights=i64[3]),
        )

        Access children using attribute or dictionary syntax:

        >>> data = HiData(name='test', value=42)
        >>> data.value
        42
        >>> data['value']
        42

        Clone and modify:

        >>> original = HiData(name='original', x=1, y=2)
        >>> cloned = original.clone()
        >>> cloned['z'] = 3
    """

    name: str = field(pytree_node=False)
    children: Dict[str, Any] = dataclasses.field(default_factory=dict, kw_only=True)

    def __init__(self, children: Dict[str, Any] = None, name: str = None, **kwargs):
        object.__setattr__(self, 'children', dict(children) if children is not None else {})
        object.__setattr__(self, 'name', name)
        self.children.update(kwargs)

    def __len__(self):
        return len(self.children)

    def __getattr__(self, key: str) -> Any:
        """Get child state by attribute name."""
        try:
            return self.children[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __getitem__(self, key: str) -> Any:
        """Get child state by name."""
        return self.children[key]

    def __contains__(self, key: str) -> bool:
        """Check if child exists."""
        return key in self.children

    def keys(self):
        """Return child keys."""
        return self.children.keys()

    def items(self):
        """Return child items."""
        return self.children.items()

    def values(self):
        """Return child values."""
        return self.children.values()

    def __repr__(self) -> str:
        """Return the hierarchical pretty representation."""
        return pretty_repr_object(self)

    __repr__._uses_pretty_repr = True

    def __pretty_repr__(self):
        """Yield the pretty representation: ``name`` followed by the children.

        Nested ``HiData`` objects, containers and arrays are laid out by the
        shared pretty-repr engine; other values whose ``repr`` is longer than
        60 characters are truncated with ``...``.

        Yields
        ------
        PrettyType or PrettyAttr
            The type header followed by the items.
        """
        yield PrettyType(type(self))
        yield PrettyAttr('name', repr(self.name))
        for key, value in self.children.items():
            yield PrettyAttr(key, self._format_value(value))

    @staticmethod
    def _format_value(value: Any) -> Any:
        """Return ``value`` for the engine, or a truncated ``repr`` string."""
        if (
            isinstance(value, (PrettyRepr, list, tuple, dict))
            or format_array_like(value) is not None
        ):
            return value
        value_str = repr(value)
        if len(value_str) > 60:
            return f"{value_str[:57]}..."
        return value_str

    def clone(self) -> 'HiData':
        """
        Create a deep copy of the state, recursively cloning children.

        Returns
        -------
        New state instance with cloned tensors.
        """
        cloned_children = {}
        for k, v in self.children.items():
            if v is None:
                cloned_children[k] = None
            elif hasattr(v, 'clone'):
                cloned_children[k] = v.clone()
            else:
                cloned_children[k] = v
        return self.__class__(children=cloned_children, name=self.name)

    @property
    def state_size(self) -> int:
        """Number of state variables per node."""
        total = 0
        for v in self.children.values():
            if isinstance(v, HiData):
                total = total + v.state_size
            elif v is not None:
                total += 1
        return total

    @property
    def dtype(self):
        """Return dtype of first array child."""
        for v in self.children.values():
            if v is None:
                continue
            if isinstance(v, HiData):
                try:
                    return v.dtype
                except ValueError:
                    continue
            if hasattr(v, 'dtype'):
                return v.dtype
        raise ValueError("No array children found to determine dtype")

    def add(self, *args, **updates) -> 'HiData':
        children = {k: v for k, v in self.children.items()}
        for arg in args:
            assert isinstance(arg, (HiData, dict)), 'Argument must be of type HiData or Dict, got {}'.format(type(arg))
            for k, v in arg.items():
                children[k] = v
        for k in updates:
            children[k] = updates[k]
        return self.__class__(children=children, name=self.name)

    def pop(self, *args) -> 'HiData':
        children = {k: v for k, v in self.children.items()}
        for arg in args:
            children.pop(arg)
        return self.__class__(children=children, name=self.name)

    def replace(self, **updates) -> 'HiData':
        """
        Apply partial updates to child states.

        Parameters
        ----------
        updates
            Dictionary of child states to update.

        Returns
        -------
        New state instance with updated children.
        """
        children = {k: v for k, v in self.children.items()}
        for k in updates:
            children[k] = updates[k]
        return self.__class__(children=children, name=self.name)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary representation.

        Returns
        -------
        Dictionary mapping state variable names to tensors.
        """
        return {k: d.to_dict() if isinstance(d, HiData) else d for k, d in self.children.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> 'HiData':
        """
        Create state from dictionary.

        Parameters
        ----------
        d
            Dictionary mapping state variable names to tensors.

        Returns
        -------
        State instance.
        """
        return cls(children={k: cls.from_dict(v) if isinstance(v, dict) else v for k, v in d.items()})
