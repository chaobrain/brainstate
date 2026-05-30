# Installation

BrainState runs anywhere JAX runs. Install the variant that matches your hardware; the only
difference is which JAX backend is pulled in.

## Install

::::{tab-set}

:::{tab-item} CPU
```bash
pip install -U brainstate[cpu]
```
:::

:::{tab-item} GPU (CUDA)
```bash
pip install -U brainstate[cuda12]   # CUDA 12
pip install -U brainstate[cuda13]   # CUDA 13
```
:::

:::{tab-item} TPU
```bash
pip install -U brainstate[tpu]
```
:::

::::

The extras (`[cpu]`, `[cuda12]`, `[cuda13]`, `[tpu]`) select the appropriate `jax`/`jaxlib` build.
Plain `pip install -U brainstate` installs the library against whatever JAX is already present,
which is convenient if you manage JAX yourself.

BrainState needs **Python 3.10 or newer**.

## The wider ecosystem

BrainState is one component of a brain-modeling ecosystem. To install it together with the
compatible companion packages — `brainunit` for physical units, `braintools` for optimizers,
metrics and initializers, `brainpy` for dynamical-systems modeling — install the bundle:

```bash
pip install -U BrainX
```

## From source

To track the development version or contribute, install from the repository in editable mode:

```bash
git clone https://github.com/chaobrain/brainstate.git
cd brainstate
pip install -e .
```

## Verify

Confirm the installation and check which backend JAX selected:

```bash
python -c "import brainstate; print(brainstate.__version__)"
```

```python
import jax
print(jax.devices())   # e.g. [CpuDevice(id=0)] or [CudaDevice(id=0)]
```

If `jax.devices()` reports a CPU device on a GPU machine, the CPU build of `jaxlib` is installed;
reinstall with the matching CUDA extra above.

## Next steps

- [Quickstart](quickstart.ipynb) — train a model end to end in a few minutes.
- [Thinking in BrainState](thinking_in_brainstate.md) — the mental model for the rest of the docs.
