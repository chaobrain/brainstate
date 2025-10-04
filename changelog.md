# Release Notes



## Version 0.2.0

### New Features

- **Convolution Enhancements**: Added support for both channels-first and channels-last data formats in all convolution layers (`Conv1d`, `Conv2d`, `Conv3d`, and their weight-standardized variants).
  - New `channel_first` boolean parameter (default: `False`) allows PyTorch-style channels-first format (e.g., `[B, C, H, W]`) in addition to default JAX-style channels-last format (e.g., `[B, H, W, C]`).
  - Set `channel_first=True` for PyTorch-compatible data format.
  - Enables easier migration of PyTorch models to BrainState.

- **Transposed Convolution Layers**: Added complete implementations of transposed convolutions for upsampling operations.
  - New layers: `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`
  - Support for both channels-first and channels-last data formats via `channel_first` parameter
  - Configurable stride for controllable upsampling factors
  - Grouped transposed convolution support for efficient parameter usage
  - Automatic padding computation for 'SAME' and 'VALID' modes with stride > 1
  - Comprehensive NumPy-style docstrings with detailed examples and use cases
  - Commonly used in encoder-decoder architectures, GANs, semantic segmentation, and super-resolution networks

### Documentation

- **Comprehensive Docstrings**: Added detailed NumPy-style docstrings to all convolution APIs with:
  - Full parameter descriptions with types and default values
  - Multiple practical examples wrapped in `.. code-block:: python` directives
  - Comparison sections highlighting differences from PyTorch
  - Mathematical formulas for weight standardization
  - References to original papers
  - Detailed notes on use cases and best practices



## Version 0.1.0

The first version of the project.


