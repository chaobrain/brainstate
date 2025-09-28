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
