# Contributing to BrainState

Thanks for taking the time to contribute. The goal of this guide is to make it easy for you to get started, whether you are reporting a bug, improving the documentation, or proposing new features.

## Code of Conduct
Participation in the BrainState community is governed by our `CODE_OF_CONDUCT.md`. By contributing you agree to uphold those standards.

## Ways to Contribute
- Report bugs or request features through https://github.com/chaobrain/brainstate/issues.
- Improve documentation, tutorials, and examples.
- Share benchmarks or reproducible research artefacts that highlight how BrainState is used.
- Review pull requests and help triage incoming issues.

If you are unsure about the best way to help, feel free to start a discussion in an issue outlining your idea.

## Development Environment
1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-user>/brainstate.git
   cd brainstate
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
3. Upgrade pip tooling and install the project in editable mode with the test dependencies:
   ```bash
   python -m pip install -U pip setuptools wheel
   pip install -e ".[testing]"
   ```
4. (Optional) Install documentation requirements if you plan to build the docs:
   ```bash
   pip install -r requirements-doc.txt
   ```

## Tooling and Style
- Install pre-commit to run the project linters automatically:
  ```bash
  pip install pre-commit
  pre-commit install
  ```
  Run `pre-commit run --all-files` before pushing to ensure your changes satisfy the configured checks.
- Python code should follow the standards enforced by `flake8` and the rest of the tooling in `.pre-commit-config.yaml`.
- Keep imports tidy and prefer explicit typing when it clarifies intent.

## Typing

`brainstate` ships inline type information (PEP 561 `py.typed`) verified by a
blocking `mypy` gate. When adding or modifying code, follow these rules:

1. **Future annotations.** Start every module with `from __future__ import
   annotations`. Annotations become lazy strings — zero runtime cost, faster
   imports, and no circular-import headaches.
2. **One source of truth for shared types.** Import shared aliases/protocols
   (`ArrayLike`, `Shape`, `Size`, `Axes`, `DTypeLike`, `PyTree`, `SeedOrKey`,
   `Filter`, `Key`, ...) from `brainstate.typing`. Do not redefine them locally;
   add new shared concepts there with a docstring.
3. **Type-only imports go under `TYPE_CHECKING`.** Use `if TYPE_CHECKING:` for
   imports needed only for annotations so runtime imports stay lean.
4. **Run mypy before pushing:** `mypy brainstate/` (or `pre-commit run mypy`).

### Advancing the ratchet (typing a new module)

The mypy config in `pyproject.toml` suppresses errors for not-yet-typed modules
via a `brainstate.*` wildcard and enforces finished modules via concrete
overrides (a concrete module pattern takes precedence over the wildcard). To
"finish" a module:

1. Add `from __future__ import annotations`; type its public `__all__` exports,
   then its internals; move type-only imports under `TYPE_CHECKING`.
2. Add the module to the sorted `module = [...]` list in the **RATCHET** override
   block in `pyproject.toml`.
3. Run `mypy brainstate/` until it reports `Success`.
4. Where useful, add `assert_type` checks for the module's public APIs to
   `brainstate/_typing_static_check.py`.

Prefer readable public aliases over raw `Union[...]`; keep `Any` rare and
intentional. Keep annotations consistent with the NumPy-doc `name : type` field.

## Testing
Run the full test suite with:
```bash
pytest
```
Add or update tests whenever you fix a bug or add functionality. Tests should pass before you submit a pull request.

## Pull Request Checklist
- Include a clear description of the change and the motivation behind it.
- Reference the related issue when applicable (e.g. "Closes #123").
- Update or add documentation and examples when behaviour changes.
- Update `changelog.md` when the change impacts users.
- Ensure `pre-commit run --all-files` and `pytest` succeed locally.
- Keep each pull request focused; large refactors should be discussed in advance.

## Documentation Contributions
Documentation lives in the `docs/` directory and is built with Sphinx. After installing the documentation requirements you can build the HTML site locally with:
```bash
sphinx-build -b html docs docs/_build/html
```
Open `docs/_build/html/index.html` in your browser to review the result.

## Need Help?
If you have questions or want early feedback on substantial changes, open a draft pull request or start a discussion on the issue tracker. We are happy to help you land your contribution.

## Testing conventions

All tests live next to the source as `<module>_test.py` and run via `pytest brainstate/`.

### Framework
- Use `unittest.TestCase`, with `absl.testing.parameterized` for parametric cases.
- Every test class/method has a one-line NumPy-doc summary describing the scenario.
- Naming: `class Test<Thing>`, `def test_<behavior>`.

### Shared helpers (`brainstate/_testing.py`)
- `assert_allclose(actual, expected, *, rtol, atol, check_dtype)` — unit-aware (brainunit) value+shape check.
- `assert_jit_equal(fn, *args)` — `jit(fn)` matches eager.
- `assert_grad_finite(fn, *args, argnums=0)` — gradients of a scalar fn are finite.
- `assert_vmap_equal(fn, *batched_args)` — `vmap(fn)` over axis 0 matches a Python loop.
- `assert_transform_compatible(fn, *args, transforms=("jit","grad","vmap"))` — umbrella.
- `assert_pytree_roundtrip(obj)` — flatten/unflatten roundtrip.
- Size constants `SMALL_BATCH=4`, `SMALL_DIM=16`, `SMALL_SEQ=5`.
- `seeded(seed)` — context manager (`with seeded(0): ...`) that seeds and restores RNG.

### Random numbers
- Always use `brainstate.random` (e.g. `brainstate.random.rand`, `.randn`, `.seed`).
- **Never** import or call `jax.random` directly in tests.

### Performance
- Keep inputs tiny: batch ≤ 4, dims ≤ 16 (use the `SMALL_*` constants).
- Reuse compiled functions / built modules across cases; do not re-`jit` per case.
- Mark expensive tests `@pytest.mark.slow` (skipped by default; run with `pytest --run-slow`).

### The per-API checklist
Each public API should be tested for: (1) happy path; (2) shapes/dtypes/units;
(3) argument variations; (4) edge cases (empty/zero/single/large, boundary, NaN/inf);
(5) failure paths (assert exception type AND message); (6) JAX-transform compatibility
(jit/grad/vmap/scan); (7) State semantics (read/write, no leaks, pytree roundtrip);
(8) determinism under seeded RNG; (9) serialization (treefy/state_dict) where applicable.

### Coverage bar
Each subpackage targets **≥90% line coverage** (branch coverage measured/reported). Measure
with the coverage CLI (the `pytest-cov` plugin aborts under JAX/XLA on some platforms; the CLI
starts the tracer cleanly):
```bash
coverage run -m pytest brainstate/<subpackage>/
coverage report -m
```
Coverage settings (source, branch, omit) live in `[tool.coverage.*]` in `pyproject.toml`.
