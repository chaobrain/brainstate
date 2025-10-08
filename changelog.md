# Release Notes


## Version 0.2.0

This is a major release with significant refactoring, new features, and comprehensive documentation improvements.

### Breaking Changes

- **Module Deprecations**: Deprecated `brainstate.augment`, `brainstate.compile`, and `brainstate.functional` modules in favor of `brainstate.transform` and `brainstate.nn`
  - Added deprecation proxies to guide users towards replacement modules
  - Updated all documentation and examples to use new module paths

- **State Management**: Replaced `write_back_state_values` with `assign_state_vals_v2` for improved state management

- **Import Path Changes**: Major refactoring of import paths across the codebase
  - Moved initialization references to use `brainstate.nn`
  - Updated random functions to use `brainstate.random`
  - Standardized imports across all modules

- **Type System**: Implemented `JointTypes` and `OneOfTypes` generic aliases to enhance type checking and avoid metaclass conflicts
  - Support for subscript syntax
  - Improved type hints across modules

- **Copyright**: Updated copyright notices to reflect new ownership by BrainX Ecosystem Limited

### New Features

#### Neural Network Components

- **Transposed Convolution Layers**: Complete implementations for upsampling operations
  - `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`
  - Support for both channels-first and channels-last data formats via `channel_first` parameter
  - Configurable stride for controllable upsampling factors
  - Grouped transposed convolution support
  - Automatic padding computation for 'SAME' and 'VALID' modes

- **Convolution Enhancements**: Added support for both channels-first and channels-last data formats
  - New `channel_first` boolean parameter (default: `False`)
  - PyTorch-compatible format (e.g., `[B, C, H, W]`) when `channel_first=True`
  - Default JAX-style format (e.g., `[B, H, W, C]`) when `channel_first=False`

- **Padding Layers**: Added padding layers for 1D, 2D, and 3D tensors with various modes

- **Unpooling Layers**: Added `MaxUnpool1d`, `MaxUnpool2d`, and `MaxUnpool3d` with `return_indices` support

- **Gradient Utilities**: Implemented `clip_grad_norm` function for gradient clipping in PyTree structures

- **Embedding Enhancements**:
  - Added `padding_idx`, `max_norm`, and `norm_type` parameters
  - Improved gradient management with new `_contains_tracer` function
  - Optimized max_norm application with accessed mask for scaling

- **BatchNorm Improvements**: Added `feature_axis` and `track_running_stats` parameters

- **LoRA Layer**: Added `in_size` parameter for improved size handling

- **Activation Functions**: Added new activation functions and improved signatures

#### Transform & Compilation

- **StatefulMapping**: Introduced for enhanced state management in vmap transformations

- **Mixin Classes**: Added `Mode`, `JointMode`, `Batching`, and `Training` classes for computation behavior control

- **Bounded Cache**: Implemented thread-safe bounded cache for JAX Jaxpr with:
  - Comprehensive validation
  - Statistics tracking
  - Enhanced error handling

- **Input Validation**: Enhanced input size handling to support numpy integer types

- **Context Parameters**: Update method now accepts additional context parameters for improved environment settings

#### Random & Initialization

- **Dependencies**: Integrated `braintools` for initialization and surrogate gradient functions
  - Updated all initialization references
  - Refactored to use `braintools.surrogate` for spike functions

- **Random Functions**: Replaced `uniform_for_unit` with `jr.uniform` for consistency and performance

#### Utilities & Infrastructure

- **Filter Utilities**: Added comprehensive filter utilities for nested structures

- **Pretty Representation**: Enhanced pretty_pytree module with:
  - Comprehensive documentation
  - Mapping functions
  - JAX integration

- **Error Handling**: Improved state length validation by replacing assertions with `ValueError` exceptions

- **Collective Operations**: Updated function signatures to return target in collective operations

### Documentation

- **Comprehensive Docstrings**: Added detailed NumPy-style docstrings across all modules
  - Full parameter descriptions with types and default values
  - Multiple practical examples in code blocks
  - Comparison sections highlighting differences from PyTorch
  - Mathematical formulas where applicable
  - References to original papers
  - Best practices and use cases

- **New Documentation Pages**:
  - `brainstate.environ` module documentation
  - `brainstate.transform` (renamed from compile.rst)
  - Random number generation module
  - Pretty representation module
  - State management tutorial notebook

- **Enhanced Examples**: Updated documentation examples to use interactive prompts for clarity

- **Module Descriptions**: Enhanced documentation with detailed descriptions, key features, and usage examples

### Testing

- **Comprehensive Test Coverage**: Added extensive test suites for:
  - `_BoundedCache` and `StatefulFunction`
  - `brainstate.mixin` module
  - `brainstate.environ` module (context management, precision settings, callbacks)
  - DeprecatedModule and proxy creation functionality
  - Compatible import module
  - Metrics module
  - Node class and helper functions
  - Activation functions with shape and gradient checks
  - Dropout layers
  - Surrogate gradient functions
  - Filter utilities
  - Struct module
  - Pretty representation

- **Test Framework Updates**: Refactored tests to use `absltest` for better JAX compatibility

### Refactoring

- **File Reorganization**:
  - Renamed `metrics.py` to `_metrics.py`
  - Renamed `_rate_rnns.py` to `_rnns.py`
  - Renamed `_init.py` to `init.py`
  - Reorganized graph module files
  - Cleaned up unused imports and classes

- **Code Quality**:
  - Streamlined imports across all modules
  - Enhanced code formatting and whitespace consistency
  - Removed unnecessary inheritance and unused elements
  - Simplified type annotations
  - Improved method signatures for clarity

- **Neuron & Synapse Classes**: Refactored to use brainpy module and updated initialization methods

- **Base Classes**: Changed base class of `EINet` and `Net` from `DynamicsGroup` to `Module` for consistency

- **Evaluation Functions**: Refactored and updated method names for consistency

### Infrastructure

- **Version Bump**: Updated version to 0.2.0

- **Development Dependencies**: Added `braintools` to development requirements

- **Issue Templates**: Added bug report and feature request templates for improved issue tracking

- **CI/CD**: Refactored CI configurations to update pip installation commands

- **Git Ignore**: Updated to exclude example figures directory and build artifacts

### Bug Fixes

- Enhanced delay handling for multi-dimensional inputs
- Fixed gradient function references
- Improved deprecation handling in tests
- Fixed precision checks in complex number handling


## Version 0.1.0

The first version of the project.


