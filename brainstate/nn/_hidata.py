"""
Hierarchical data containers for parameter and state management.

This module provides the ``HiData`` class, a flexible container for hierarchical
data structures that supports dictionary-like and attribute-style access,
cloning, serialization, and composition.
"""

import dataclasses
from typing import Any, Dict

import brainstate
from brainstate.util.struct import dataclass, field

Array = brainstate.typing.ArrayLike

__all__ = [
    'HiData',
]


def is_dataclass(cls):
    return hasattr(cls, '_brainstate_dataclass')


@dataclass
class HiData:
    """
    Hierarchical state container for composed dynamics.

    Stores child states in a dictionary where keys match the attribute names
    of child dynamics in the parent dynamics class.

    Supports two initialization styles:
        - Data(children={'key1': data1, 'key2': data2})
        - Data(key1=data1, key2=data2)

    And two access styles:
        - cd['key1'] or cd.key1

    Attributes:
        children: Dict mapping child names to their states.

    Examples:
        Create a simple Data object:

        >>> data = HiData(name='config', learning_rate=0.01, batch_size=32)
        >>> print(data)
        ParamData(
          name='config',
          learning_rate=0.01,
          batch_size=32
        )

        Create nested Data objects:

        >>> import numpy as np
        >>> optimizer = HiData(name='optimizer', lr=0.001, momentum=0.9)
        >>> model = HiData(name='model', weights=np.array([1, 2, 3]))
        >>> config = HiData(name='config', optimizer=optimizer, model=model)
        >>> print(config)
        ParamData(
          name='config',
          optimizer=ParamData(
            name='optimizer',
            lr=0.001,
            momentum=0.9
          ),
          model=ParamData(
            name='model',
            weights=Array(shape=(3,), dtype=int64)
          )
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
        """Return hierarchical string representation."""
        return self._repr_recursive(indent=0)

    def _repr_recursive(self, indent: int = 0) -> str:
        """
        Generate hierarchical representation with indentation.

        Format:
            HiData(
              name='value',
              child1=value1,
              child2=HiData(
                name='nested',
                subchild=42
              ),
              child3=value3
            )

        Args:
            indent: Current indentation level.

        Returns:
            String representation of this HiData and its children.
        """
        indent_str = "  " * indent
        name_str = f"'{self.name}'" if self.name else "None"

        # All children use '=' separator
        separator = "="

        if not self.children:
            # Empty Data object
            return f"{indent_str}HiData(name={name_str})"

        # Start with Data( and name parameter
        lines = [f"{indent_str}HiData("]
        lines.append(f"{indent_str}  name={name_str},")

        # Add children as parameters
        child_items = list(self.children.items())
        for i, (key, value) in enumerate(child_items):
            is_last = (i == len(child_items) - 1)
            comma = "" if is_last else ","

            if isinstance(value, HiData):
                # Recursively format nested Data objects
                nested_repr = value._repr_recursive(indent + 1)
                # Remove the leading indent from nested_repr since we're adding it ourselves
                nested_lines = nested_repr.split('\n')
                nested_lines[0] = nested_lines[0].lstrip()
                nested_repr = '\n'.join(nested_lines)
                lines.append(f"{indent_str}  {key}{separator}{nested_repr}{comma}")
            else:
                # Format other values
                value_repr = self._format_value(value)
                lines.append(f"{indent_str}  {key}{separator}{value_repr}{comma}")

        # Close with parenthesis
        lines.append(f"{indent_str})")

        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        """
        Format a non-HiData value for display.

        Args:
            value: The value to format.

        Returns:
            Formatted string representation.
        """
        if value is None:
            return "None"

        # Handle arrays with shape information
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            return f"Array(shape={value.shape}, dtype={value.dtype})"

        # Handle other types
        value_str = repr(value)
        if len(value_str) > 60:
            return f"{value_str[:57]}..."
        return value_str

    def clone(self) -> 'HiData':
        """
        Create a deep copy of the state, recursively cloning children.

        Returns:
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
        return self.__class__(children=cloned_children)

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
        return HiData(children=children)

    def pop(self, *args) -> 'HiData':
        children = {k: v for k, v in self.children.items()}
        for arg in args:
            children.pop(arg)
        return HiData(children=children)

    def replace(self, **updates) -> 'HiData':
        """
        Apply partial updates to child states.

        Args:
            updates: Dictionary of child states to update.

        Returns:
            New state instance with updated children.
        """
        children = {k: v for k, v in self.children.items()}
        for k in updates:
            children[k] = updates[k]
        return self.__class__(children=children)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary mapping state variable names to tensors.
        """
        return {k: d.to_dict() if isinstance(d, HiData) else d for k, d in self.children.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> 'HiData':
        """
        Create state from dictionary.

        Args:
            d: Dictionary mapping state variable names to tensors.

        Returns:
            State instance.
        """
        return cls(children={k: cls.from_dict(v) if isinstance(v, dict) else v for k, v in d.items()})
