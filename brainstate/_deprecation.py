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

"""
Deprecation utilities for brainstate.
"""

import warnings


class DeprecatedModule:
    """
    A proxy class that mimics a module but shows deprecation warnings when accessed.

    This class allows for smooth deprecation of modules by forwarding attribute
    access to a replacement module while showing appropriate warnings.
    """

    def __init__(
        self,
        deprecated_name,
        replacement_module,
        replacement_name,
        version="0.1.11",
        removal_version=None,
        scoped_apis=None
    ):
        """
        Initialize a deprecated module proxy.

        Args:
            deprecated_name: Name of the deprecated module (e.g., 'brainstate.augment')
            replacement_module: The module to forward calls to
            replacement_name: Name of the replacement module (e.g., 'brainstate.transform')
            version: Version when deprecation started
            removal_version: Version when module will be removed (optional)
            scoped_apis: Dict mapping API names to their locations, or list of API names
        """
        self._deprecated_name = deprecated_name
        self._replacement_module = replacement_module
        self._replacement_name = replacement_name
        self._version = version
        self._removal_version = removal_version
        self._warned_attrs = {}
        self._scoped_apis = scoped_apis or {}

        # Set module-like attributes
        self.__name__ = deprecated_name
        self.__doc__ = f"DEPRECATED: {deprecated_name} is deprecated. Use {replacement_name} instead."

        # Handle scoped APIs
        if isinstance(scoped_apis, dict):
            # Dict mapping API names to their import locations
            self._api_mapping = scoped_apis
            self.__all__ = list(scoped_apis.keys())
        elif isinstance(scoped_apis, (list, tuple)):
            # List of API names to expose from replacement module
            self._api_mapping = {name: replacement_module for name in scoped_apis}
            self.__all__ = list(scoped_apis)
        else:
            # No scoping - use entire replacement module
            self._api_mapping = {}
            if hasattr(replacement_module, '__all__'):
                self.__all__ = replacement_module.__all__

    @property
    def replacement_module(self):
        if isinstance(self._replacement_module, str):
            # Lazy import of replacement module
            import importlib
            self._replacement_module = importlib.import_module(self._replacement_module)
        return self._replacement_module

    def _warn_deprecation(self, attr_name=None):
        """Show deprecation warning for module or attribute access."""
        # Only warn once per attribute to avoid spam
        warn_key = attr_name or '__module__'
        if warn_key in self._warned_attrs:
            return
        self._warned_attrs[warn_key] = True

        if attr_name:
            message = (
                f"Accessing '{attr_name}' from '{self._deprecated_name}' is deprecated "
                f"and will be removed in a future version. "
                f"Use '{self._replacement_name}.{attr_name}' instead."
            )
        else:
            message = (
                f"The '{self._deprecated_name}' module is deprecated "
                f"and will be removed in a future version. "
                f"Use '{self._replacement_name}' instead."
            )

        if self._removal_version:
            message += f" It will be removed in version {self._removal_version}."

        warnings.warn(message, DeprecationWarning, stacklevel=3)

    def __getattr__(self, name):
        """Forward attribute access to replacement module with deprecation warning."""
        self._warn_deprecation(name)

        # Check if we have scoped APIs
        if self._api_mapping:
            if name in self._api_mapping:
                # Get from specific location
                source = self._api_mapping[name]

                if isinstance(source, str):
                    # Import from module path
                    try:
                        # Handle relative imports within brainstate
                        if source.startswith('brainstate.'):
                            # Import the module dynamically
                            import importlib
                            module = importlib.import_module(source)
                            return getattr(module, name)
                        else:
                            # For other module paths, use standard import
                            module_parts = source.split('.')
                            module = __import__(source, fromlist=[name])
                            return getattr(module, name)
                    except (ImportError, AttributeError) as e:
                        # Fallback to replacement module
                        try:
                            return getattr(self._replacement_module, name)
                        except AttributeError:
                            raise AttributeError(
                                f"Module '{self._deprecated_name}' has no attribute '{name}'. "
                                f"Failed to import from '{source}': {e}. "
                                f"Check '{self._replacement_name}' for available attributes."
                            )
                else:
                    # Source is a module object
                    try:
                        return getattr(source, name)
                    except AttributeError:
                        # Fallback to replacement module
                        return getattr(self._replacement_module, name)
            else:
                # Attribute not in scoped APIs
                available_apis = ', '.join(self.__all__ or [])
                raise AttributeError(
                    f"Module '{self._deprecated_name}' has no attribute '{name}'. "
                    f"Available attributes: {available_apis}. "
                    f"Check '{self._replacement_name}' for more attributes."
                )

        # Fallback to replacement module for non-scoped access
        try:
            return getattr(self.replacement_module, name)
        except AttributeError:
            raise AttributeError(
                f"Module '{self._deprecated_name}' has no attribute '{name}'. "
                f"Check '{self._replacement_name}' for available attributes."
            )

    def __dir__(self):
        """Return attributes from replacement module."""
        self._warn_deprecation()

        if self._api_mapping:
            # Return only scoped APIs plus standard module attributes
            base_attrs = ['__name__', '__doc__', '__all__']
            return base_attrs + list(self._api_mapping.keys())
        else:
            return dir(self.replacement_module)

    def __repr__(self):
        """Return a deprecation-aware repr."""
        return f"<DeprecatedModule '{self._deprecated_name}' -> '{self._replacement_name}'>"


def create_deprecated_module_proxy(
    deprecated_name,
    replacement_module,
    replacement_name,
    **kwargs
):
    """
    Create a deprecated module proxy.

    Args:
        deprecated_name: Name of the deprecated module
        replacement_module: The module to forward calls to
        replacement_name: Name of the replacement module
        **kwargs: Additional arguments for DeprecatedModule

    Returns:
        DeprecatedModule proxy instance
    """
    return DeprecatedModule(
        deprecated_name=deprecated_name,
        replacement_module=replacement_module,
        replacement_name=replacement_name,
        **kwargs
    )
