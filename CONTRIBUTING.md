# Contributing

## Getting started

1. Fork the repo and create a feature branch from `main`.
2. Install dependencies: `pip install -r requirements.txt`
3. Make your changes, keeping each commit focused on one concern.
4. Add or update tests in `tests/` for any new behaviour.
5. Run `pytest tests/ -v` — all tests must pass before opening a PR.
6. Open a pull request with a clear description of what changed and why.

## Code style

- Python 3.10+, type hints where practical.
- Keep modules focused: one concern per file.
- No bare `except` clauses.

## Reporting bugs

Open a GitHub Issue with: Python version, OS, exact command run, and the full error output.
