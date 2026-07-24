# brainstate

## Working agreement

1. Before writing any code, describe approach, wait for approval.
2. Requirements ambiguous? Ask clarifying questions before writing code.
3. After writing code, list edge cases + suggest test cases.
4. Bug? Write a test that reproduces it, then fix until the test passes.
5. Every correction: reflect on the mistake, plan to avoid repeating it.
6. All updates must be happened on the worktree branch, not main. 
7. Use `brainstate.random` instead of `jax.random` directly for all random number generation. 
8. Write spec and plan under `<PROJECT_ROOT>/doc/specs` before implementation, so they're available for reference during implementation.
9. Tests should >90% coverage, but focus on meaningful tests that cover edge cases and critical paths, not just trivial lines. 
10. Maintain compatibility with JAX versions >= 0.8.0. Guard any version-specific behavior in `brainstate/_compatible_import.py`, preferring feature/shape detection over hard version checks.


## Docstring style (NumPy-doc)

All public classes, methods, functions must use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). Canonical section order:

1. **Short summary** – one-line imperative description (no blank line before).
2. **Extended summary** – optional, follow blank line after short summary.
3. **Parameters** – each entry: `name : type` on own line, description indented below.
4. **Returns** / **Yields** – same format as Parameters.
5. **Raises** – exception type and when raised.
6. **See Also** – related functions / classes.
7. **Notes** – implementation details, math, references.
8. **References** – numbered bibliography entries (`.. [1]`).
9. **Examples** – runnable, doctestable code snippets.

### Rules for the Examples section

- Wrap example code in `.. code-block:: python` directive so Sphinx render with syntax highlighting.
- Prefix every input line with `>>>` (continuation lines with `...`) for `doctest` compatibility.
- Show expected output on line immediately after statement, **without** prompt prefix.
- Separate distinct scenarios with blank `>>>` line.
- Always include necessary imports (`import brainunit as u`, etc.) at top of example block so self-contained.
