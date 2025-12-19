# Release Notes


## Version 0.2.8

This release ensures compatibility with JAX 0.8.2+ and removes the experimental module that was superseded by upstream changes.

### Compatibility

- **JAX 0.8.2+ Support**: Added compatibility with JAX version 0.8.2 and later. The library now uses `jax.make_jaxpr` directly for JAX >= 0.8.2 while maintaining backward compatibility with earlier versions.

### Breaking Changes

- **Removed `abstracted_axes` parameter**: The `abstracted_axes` parameter has been removed from:
  - `StatefulFunction.__init__`
  - `StatefulMapping.__init__`
  - `make_jaxpr` function
  - `_make_jaxpr` internal function

### Improvements

- **Debug mode support**: Added `debug_call` method to `StatefulFunction` for proper execution when `jax.config.jax_disable_jit` is enabled. This improves debugging workflows by allowing stateful functions to execute without JIT compilation.

- **Lazy loading optimization**: `RandomState` import in the `_mapping` module is now lazily loaded via `_import_rand_state()`, improving initial import performance and reducing circular dependency issues.

### Internal Changes

- Removed unused imports (`annotate`, `api_boundary` from `jax._src`) at module level; now imported only where needed
- Removed internal helper functions `_broadcast_prefix` and `_flat_axes_specs`
- Simplified `_abstractify` function by removing abstracted axes handling
- Updated example files to reflect API changes

## Version 0.2.7

BrainState 0.2.7 modernizes the experimental compilation stack, deepens the transformation APIs, and tightens runtime infrastructure across the project.

### Experimental Compiler and Visualization

- Introduced the experimental `neuroir` compiler built on dataclass-based graph IR elements and an explicit `CompilationContext`, improving dependency tracking, hidden-state mapping, and ClosedJaxpr fidelity even for self-connections and delay buffers.
- Added GraphDisplayer and TextDisplayer backends with hierarchical and force-directed layouts, plus richer diagnostics and tests that cover large sample networks and neuro-graph visualizations.

### Transformations and Autodiff

- Added the `jit_named_scope` decorator and supporting utilities so nested transformations emit meaningful names inside traced functions, together with `_make_jaxpr` refinements that separate dynamic/static arguments and improve caching semantics for `StatefulFunction`.
- Expanded the gradient toolkit by exporting the new Jacobian (forward and reverse), Hessian, and SOFO transforms, unifying gradient handling for classes, auxiliary returns, and state-aware updates through the transform module.

### State and Runtime Enhancements

- Replaced the experimental `ArrayParam` with a dedicated `DelayState`, propagating the new state through the compiler, delay modules, and neuro-IR so historical buffers participate in tracing and optimization just like other states.
- Environment helpers can now run against injected `EnvironmentState` instances, enabling sandboxed or per-thread configurations while DelayState-aware unit tests extend coverage of the updated modules.

### Experimental and Infrastructure Updates

- Completed the neuron IR → neuroir rename, aligned the GDiist BPU codebase with the new terminology, and added new sample networks plus placeholder skips to keep the growing compiler/displayer test surface manageable.
- Added `braincell` to the development requirements, refreshed documentation wording, and kept CI dependencies current for the GitHub Actions runners.

### Bug Fixes

- Hardened caching, randomness, and initialization logic by fixing `get_arg_cache_key`, removing stale decorator parameters, validating truncated normal draws, and correcting the exported version metadata.
- Declared Python 3.14 support and cleaned up compiler import ordering to keep linting noise low.


## Version 0.2.6

This release focuses on the experimental export pipeline and device-aware execution adapters.

### Device-Aware Wrappers

- Added registry-driven `ForLoop` and `JIT` adapters that expose decorator-style ergonomics, call counters, and validation, with CPU/GPU/TPU implementations wired through `register_*_impl` so experiments can swap device backends without touching user code.

### GDiist BPU Export

- Replaced the monolithic exporter with `gdiist_bpu.main`, refreshed parser/component/utils modules, and renamed `BpuParser` to `GdiistBpuParser`, yielding clearer analysis output, text display helpers, and far more granular unit tests.
- Introduced the thread-safe `BoundedCache` utility and integrated it with compiler wrappers to safely reuse traced graphs, alongside `_make_jaxpr` updates that enforce argument checks and improve cache key generation.
- Updated tutorials and examples to the streamlined naming scheme and refreshed device implementation docs for the new wrapper entry points.


## Version 0.2.5

Version 0.2.5 concentrates on intermediate-representation (IR) optimization quality.

### IR Optimization

- Added `_ir_optim_v2`, a comprehensive optimizer that ships constant folding, dead-code elimination, common subexpression elimination, copy propagation, and algebraic simplification passes backed by identity-aware set semantics.
- Updated the transform exports and accompanying tests to exercise the new optimizer while pruning unused configuration knobs from the earlier implementation.


## Version 0.2.4

This release introduces the new `ArrayParam` state type for parameter arrays with custom transformations, experimental BPU backend export support, enhanced JAXPR optimization capabilities, and improved module organization.

### New Features

#### ArrayParam State Type

- **ArrayParam Class**: New state type for managing parameter arrays with advanced transformation control
  - Supports custom transformations (e.g., quantization, normalization) that preserve array identity
  - Enables `vmap`, `pmap`, and other JAX transformations to correctly handle stateful parameters
  - Provides `identity()` method that returns the raw array without applying custom transformations
  - Integrates seamlessly with existing State management infrastructure
  - Useful for implementing quantization-aware training and other advanced parameter manipulations
  - Comprehensive documentation with usage examples and best practices

#### Experimental BPU Backend Export (`brainstate.experimental.gdiist_bpu`)

- **BPU Backend Export Support**: Complete infrastructure for exporting models to GDiist BPU hardware backend (727 lines)
  - `export.py`: Main export API with `to_bpu()` function for model conversion
  - `parser.py`: Operation parser that analyzes JAXPR to identify operations and connections (305 lines)
  - `data.py`: Data structures and analysis utilities for operation representation (215 lines)

- **Operation Parser Features**:
  - Automatic detection of operations from JAXPR equations using brainevent primitives
  - Data flow analysis to identify connections between operations
  - Support for various operation types: slice, add, multiply, and more
  - Detailed analysis output showing equations, inputs, outputs, and connections

- **Analysis and Debugging Tools**:
  - `display_analysis_results()`: Comprehensive visualization of parsed operations
  - Shows operation details including equation count, variable mappings, and connections
  - Displays connection information with producer/consumer operations and variable details
  - Example implementation in `examples/400_CUBA_2005_bpu.py`

### Enhancements

#### JAXPR Optimization Improvements

- **Enhanced Constant Folding**:
  - Better handling of literal values in constant folding optimization
  - Improved detection and elimination of redundant literal operations
  - More efficient constant propagation through computation graphs

- **Identity Equation Optimization**:
  - Optimized handling of `Literal` outputs to avoid unnecessary bridging equations
  - Improved identity equation creation for interface preservation
  - Better handling of edge cases in optimization passes

- **Error Handling**:
  - Added fallback source info utility for better error messages
  - Fixed potential NoneType errors in equation handling
  - Improved validation of optimization results

#### State Management

- **Enhanced State Tests**: Comprehensive test refactoring with improved coverage (454 tests)
  - Better organization of state type tests
  - More thorough validation of state behavior
  - Enhanced test readability and maintainability



## Version 0.2.3

This release introduces powerful IR (Intermediate Representation) optimization capabilities for JAX computation graphs, comprehensive state management refactoring for vectorized mapping operations, and extensive testing infrastructure improvements.

### New Features

#### IR Optimization (`brainstate.transform._ir_optim`)

- **Intermediate Representation Optimization Module** (876 lines): Complete suite of compiler-level optimizations for JAX computation graphs
  - `constant_fold`: Evaluates constant expressions at compile time, reducing runtime computation
  - `dead_code_elimination`: Removes equations whose outputs are unused, reducing computation overhead
  - `common_subexpression_elimination`: Identifies and reuses results of identical computations
  - `copy_propagation`: Eliminates unnecessary copy operations by propagating original variables
  - `algebraic_simplification`: Applies algebraic identities (x+0=x, x*1=x, x-x=0, etc.)
  - `optimize_jaxpr`: Orchestrates multiple optimization passes with configurable iteration and verbose mode

- **IdentitySet Class**: Custom set implementation using object identity (`id()`) instead of equality
  - Enables proper handling of JAX variables and Literals in optimization passes
  - Implements `MutableSet` interface with full collection protocol support
  - Essential for tracking variable usage without relying on equality comparisons

#### Optimization Features

- **Interface Preservation**: All optimizations preserve function input/output variables (invars/outvars)
  - Identity equations automatically added when needed to maintain correct interfaces
  - Uses `convert_element_type` primitive with matching dtypes as identity operation
  - Ensures optimized functions remain drop-in replacements

- **Optimization Pipeline**: Configurable multi-pass optimization with convergence detection
  - Customizable optimization sequence via `optimizations` parameter
  - Automatic convergence detection when no more reductions possible
  - Maximum iteration control with `max_iterations` parameter
  - Verbose mode with detailed statistics and progress tracking

- **JAX Integration**: Full support for JAX primitives and special cases
  - Blacklist for primitives that shouldn't be folded (broadcast_in_dim, broadcast)
  - Proper handling of `closed_call` and `scan` primitives
  - Support for both Jaxpr and ClosedJaxpr inputs

#### State Management Refactoring (`brainstate.transform._mapping`)

- **Renamed vmap to vmap2**: Major refactoring of vectorized mapping implementation (647 lines)
  - Enhanced state management with improved axis tracking
  - Better error messages and validation
  - Streamlined state value restoration logic

- **Old vmap Implementation Preserved** (`_mapping_old.py`, 579 lines): Legacy vmap with explicit state management
  - Exports original `vmap` and `vmap_new_states` functions
  - Maintains backward compatibility for existing code
  - Specialized for stateful functions with explicit state parameters

### Documentation

#### API Documentation

- **transform.rst**: Added comprehensive IR Optimization section (24 lines)
  - Detailed module description explaining compiler optimizations
  - All 6 optimization functions documented with autosummary
  - Clear explanation of benefits: reduced computation overhead, improved runtime performance
  - Positioned between Compilation Tools and Gradient Computations sections

- **NumPy-style Docstrings**: All optimization functions include:
  - Comprehensive parameter descriptions with types and defaults
  - Detailed return value documentation
  - Notes sections explaining preservation of function interfaces
  - Multiple practical examples demonstrating usage
  - Algorithm descriptions for complex optimizations
  - Cross-references between related functions

### Enhancements

#### Optimization Pipeline

- **Progress Tracking**: Verbose mode shows equation count changes after each optimization
  - Displays initial, intermediate, and final equation counts
  - Shows reduction statistics with percentages
  - Indicates convergence detection
  - Reports iteration counts

- **Validation**: Runtime checks ensure optimization correctness
  - Verifies input variables unchanged after optimization
  - Validates output variables preserved
  - Raises clear errors if interface violated
  - Checks for valid optimization names

- **Flexibility**: Customizable optimization sequences
  - Apply all optimizations in recommended order (default)
  - Select specific optimizations only
  - Control iteration limits
  - Toggle verbose output

#### JAX Integration

- **JaxprEqn Construction**: Proper handling of required `ctx` parameter
  - Uses `JaxprEqnContext(None, True)` for identity equations
  - Ensures compatibility with JAX internal API
  - Maintains proper equation structure

- **Primitive Handling**: Special cases for JAX primitives
  - Blacklist for primitives that shouldn't be optimized
  - Proper parameter extraction and validation
  - Support for effects and source_info fields

### Bug Fixes

- Fixed JaxprEqn constructor calls to include required `ctx` parameter (7th positional argument)
- Corrected import paths for `vmap2` in test files and tutorials
- Fixed `RandomState.uniform()` calls to use `size` parameter instead of `shape`
- Enhanced test assertions for proper state axis handling
- Improved error messages for batch axis mismatches

### Refactoring

#### Transform Module

- **Renamed Files**:
  - `vmap` → `vmap2` in `_mapping.py`
  - Preserved original `vmap` in `_mapping_old.py` for compatibility

- **Module Exports**: Updated `__init__.py` to export both old and new vmap implementations
  - `vmap` from `_mapping_old.py` (legacy)
  - `vmap2` from `_mapping.py` (new)
  - `vmap_new_states` from both modules

## Version 0.2.2

This release focuses on enhancing hidden state management for recurrent neural networks and eligibility trace-based learning, along with comprehensive testing and documentation improvements.

### New Features

#### Hidden State Classes

- **HiddenGroupState**: New class for managing multiple hidden states within a single array
  - Stores multiple states in the last dimension of a single array
  - Provides `get_value()` and `set_value()` methods for accessing individual states by index or name
  - Optimized for LSTM-style architectures with multiple hidden components (h, c)
  - Includes `name2index` mapping for convenient state access

- **HiddenTreeState**: New class for managing multiple hidden states with different physical units
  - Supports PyTree structure (dict or sequence) of hidden states
  - Preserves physical units (e.g., voltage, current, conductance) via `brainunit` integration
  - Provides `name2unit` and `index2unit` mappings for unit tracking
  - Ideal for neuroscience models with heterogeneous state variables
  - Maintains compatibility with BrainScale online learning

#### State Utilities

- **maybe_state**: New utility function for flexible value extraction
  - Extracts values from State objects automatically
  - Returns non-State values unchanged
  - Simplifies writing functions that accept both states and raw values

### Enhancements

#### State Classes

- **HiddenState**: Enhanced documentation and type checking
  - Restricted to `numpy.ndarray`, `jax.Array`, and `brainunit.Quantity` types only
  - Added comprehensive docstrings with examples
  - Clarified equivalence to `brainstate.HiddenState` for online learning
  - Improved error messages for invalid input types

- **BatchState**: Now properly exported in the public API
  - Available via `brainstate.BatchState`
  - Enhanced documentation for batch data management

#### Documentation

- **API Reference**: Completely reorganized `brainstate.rst` documentation
  - Organized into 6 major sections: Core State Classes, State Management, State Utilities, Error Handling, and Submodules
  - Added detailed descriptions for each section and subsection
  - Included comprehensive bullet-point summaries for all APIs
  - Enhanced deprecation warnings with clear migration paths
  - Added module-level descriptions for all submodules

- **State Classes**: Enhanced documentation for all state types
  - Added detailed use case descriptions
  - Included practical examples for each state type
  - Clarified semantic distinctions between state types
  - Documented integration with JAX transformations

- **JAX Transformations**: Improved documentation for stateful transforms
  - Enhanced docstrings for `jit`, `grad`, `vmap`, `scan`, and other transforms
  - Added examples showing state management patterns
  - Documented state tracing behavior
  - Clarified interaction with `StateTraceStack`

#### Transform System

- **Enhanced State Finding**: New `_find_state.py` module for automatic state discovery
  - Improved state detection in nested structures
  - Better handling of state dependencies
  - Enhanced error messages for state-related issues

- **StatefulFunction**: Major enhancements to `make_jaxpr` functionality
  - Improved Jaxpr generation for stateful computations
  - Better handling of state read/write tracking
  - Enhanced debugging support

- **Mapping Transformations**: Significant refactoring of `vmap` and `pmap`
  - Improved state management across vectorized operations
  - Better handling of state broadcasting
  - Enhanced error reporting for mapping operations

#### Random Number Generation

- **Module Reorganization**: Complete refactoring of random module structure
  - Renamed `_rand_funs.py` to `_fun.py`
  - Renamed `_rand_seed.py` to `_seed.py`
  - Renamed `_rand_state.py` to `_state.py`
  - Extracted distribution implementations to new `_impl.py` module (691 lines)

- **Improved Random State**: Enhanced `RandomState` class with better state management
  - Simplified implementation (reduced from 534 to ~300 lines)
  - Better integration with JAX's random number generation
  - Improved thread safety and state isolation

### Testing

- **Comprehensive Test Suite**: Added 102 tests covering all state functionality
  - **TestBasicState** (13 tests): Core State class operations
  - **TestShortTermState** (2 tests): Short-term state behavior
  - **TestLongTermState** (2 tests): Long-term state behavior
  - **TestParamState** (2 tests): Parameter state usage patterns
  - **TestBatchState** (2 tests): Batch state functionality
  - **TestHiddenState** (7 tests): Hidden state with different array types
  - **TestHiddenGroupState** (9 tests): Multiple hidden state management
  - **TestHiddenTreeState** (12 tests): PyTree hidden states with units
  - **TestFakeState** (4 tests): Lightweight state alternative
  - **TestStateDictManager** (6 tests): State collection management
  - **TestStateTraceStack** (11 tests): State tracing and recovery
  - **TestTreefyState** (6 tests): PyTree state references
  - **TestContextManagers** (6 tests): State context managers
  - **TestStateCatcher** (8 tests): State catching utilities
  - **TestIntegrationScenarios** (5 tests): Real-world use cases

### Bug Fixes

- Fixed `HiddenGroupState.set_value()` to work correctly with JAX arrays
- Improved error handling in hidden state value validation
- Enhanced type checking for hidden state initialization


### Documentation

#### Tutorial Reorganization

- **Basics Tutorials**: Complete rewrite and expansion
  - `01_getting_started.ipynb`: Enhanced introduction with practical examples
  - `02_state_management.ipynb`: Comprehensive state management guide
  - `03_random_numbers.ipynb`: In-depth random number generation tutorial

- **Neural Networks Tutorials**: Restructured and expanded
  - `01_module_basics.ipynb`: New comprehensive module system guide
  - `02_basic_layers.ipynb`: Enhanced layer documentation with examples
  - `03_activations_normalization.ipynb`: Detailed activation and normalization guide
  - `04_recurrent_networks.ipynb`: New RNN tutorial with practical examples
  - `05_dynamics_systems.ipynb`: New dynamical systems tutorial

- **Examples**: Reorganized and enhanced
  - Renamed `10_image_classification.ipynb` to `01_image_classification.ipynb`
  - Renamed `11_sequence_modeling.ipynb` to `02_sequence_modeling.ipynb`
  - Added `03_brain_inspired_computing.ipynb`: New brain-inspired computing examples
  - Renamed `18_optimization_tricks.ipynb` to `04_optimization_tricks.ipynb`
  - Renamed `19_model_deployment.ipynb` to `05_model_deployment.ipynb`

- **Transforms Tutorials**: Reorganized for better flow
  - `01_jit_compilation.ipynb`: New comprehensive JIT guide
  - `02_automatic_differentiation.ipynb`: Enhanced autodiff tutorial
  - `03_vectorization.ipynb`: Improved vmap/pmap guide
  - `04_loops_conditions.ipynb`: Enhanced control flow guide
  - `05_other_transforms.ipynb`: Other transformation utilities

- **Advanced Tutorials**: Renumbered for clarity
  - `01_graph_operations.ipynb` (formerly `14_graph_operations.ipynb`)
  - `02_mixin_system.ipynb` (formerly `15_mixin_system.ipynb`)
  - `03_typing_system.ipynb` (formerly `16_typing_system.ipynb`)
  - `04_utilities.ipynb` (formerly `17_utilities.ipynb`)

- **Migration Guides**: Updated and simplified
  - `01_migration_from_pytorch.ipynb`: Enhanced PyTorch migration guide
  - Removed outdated BrainPy integration notebook

- **Supplementary**: Reorganized
  - `01_performance_optimization.ipynb`
  - `02_debugging_tips.ipynb`
  - `03_faq.ipynb`: Updated FAQ with new content

#### API Documentation

- Enhanced module documentation in `nn.rst` with 306 line improvements
- Updated `transform.rst` with new transform APIs
- Improved `environ.rst` and `graph.rst` documentation

### Refactoring

- Removed deprecated `eval_shape` module and tests
- Removed deprecated `_random.py` transform module
- Cleaned up unused imports across all modules
- Improved code organization in neural network layers
- Enhanced type hints and docstrings throughout

### Infrastructure

- Added development dependency for tutorial generation
- Updated benchmark scripts for performance testing
- Improved test coverage across transformation modules




## Version 0.2.0

This is a major release with significant refactoring, new features, and comprehensive documentation improvements.

### Breaking Changes

- **Module Deprecations**: Deprecated `brainstate.transform`, `brainstate.transform`, and `brainstate.functional` modules in favor of `brainstate.transform` and `brainstate.nn`
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


