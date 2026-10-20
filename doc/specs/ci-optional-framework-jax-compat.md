# Skip interop tests when an optional framework's jax requirement is unmet

## Problem

The `test_linux (3.13, 0.9.0)` CI leg fails 31 tests in
`brainstate/interop/_conversion_test.py` with

    AttributeError: module 'jax.extend.core' has no attribute 'get_opaque_trace_state'

CI installs `requirements-dev.txt` first. That pulls in the latest flax (0.12.10, which
declares `Requires-Dist: jax>=0.11.1`). CI then downgrades with `pip install jax==0.9.0`,
which leaves an incompatible flax installed. `import flax.nnx` still succeeds, so the
existing `_importorskip` guard doesn't skip, and flax fails at runtime instead.

## Decision

Keep the CI install as it is. Before importing an optional framework, the test helper reads
the framework's installed package metadata (the same `Requires-Dist` that PyPI publishes,
read offline through `importlib.metadata`). It walks the framework and its runtime
dependencies transitively. If any of them declares a `jax`/`jaxlib` requirement that the
installed version doesn't satisfy, the tests are skipped with a reason that names the
offending requirement.

## Design

All changes are in `brainstate/interop/_conversion_test.py`:

- `_unmet_jax_requirements(requires, installed)` is a pure function. It parses
  `Requires-Dist` strings with `packaging`, keeps `jax`/`jaxlib` requirements that apply to a
  plain install (no extras), and returns the specs the installed versions don't satisfy.
  Pre-releases are allowed, so a jax dev build can still satisfy a spec.
- `_jax_incompatibility(dist_name)` walks the dependency graph transitively, with a visited
  set to handle cycles. It skips distributions that aren't installed. It returns a reason
  string, or `None` if everything is satisfied.
- `_importorskip(modname)` maps the module to its distribution, runs the check, and skips if
  needed. The existing broad-exception import fallback stays as a second line of defence.

## Consequence

With flax 0.12.10, the nnx/linen interop tests skip on the jax 0.9.0, 0.10.0 and 0.11.0 CI
legs and run only on the latest-jax leg. The equinox tests keep running on every leg.

## Tests

Unit tests for the pure function cover:

- the CI reproduction case (flax 0.12.10 requirements with jax 0.9.0)
- a satisfied requirement
- a bare `jax` requirement
- requirements that apply only with extras
- a `jaxlib` mismatch
- an exclusion spec (`!=`)
- non-jax requirements

There are also tests for an unknown distribution and for a transitive mismatch with a cycle,
using a monkeypatched metadata graph.
