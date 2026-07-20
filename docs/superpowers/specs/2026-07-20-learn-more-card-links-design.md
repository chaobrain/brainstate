# Learn More Card Links Design

## Goal

Make every card in the `Learn more` section of `docs/index.rst` lead readers to
the correct documentation entry point while preserving the existing card layout,
titles, icons, ordering, column widths, and visual classes.

## Link Strategy

Use Sphinx document links instead of output-file URLs. Each card will specify a
source document with `:link:` and add `:link-type: doc`, allowing Sphinx to resolve
and validate the target during documentation builds.

| Card | Document target |
| --- | --- |
| Get Started | `getting_started/index` |
| Tutorials | `tutorials/core/index` |
| How-to guides | `how_to/index` |
| Concepts | `concepts/index` |
| Examples | `examples/index` |
| API Reference | `apis/brainstate` |

The Tutorials card intentionally opens the Core tutorial track because the project
does not currently provide a top-level `tutorials/index` document.

## Scope

Only the six card link values and their link type options will change. No other
content or styling in `docs/index.rst` will be modified.

## Verification

1. Confirm all six source documents exist.
2. Build the Sphinx HTML documentation with the local preview configuration.
3. Confirm the build exits successfully and produces `docs/_build/html/index.html`.
4. Inspect the generated homepage links and verify that each resolves to its
   intended generated HTML page.
