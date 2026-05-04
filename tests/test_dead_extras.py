"""Regression guard against the dead [integrations] extra coming back.

The audit confirmed that no first-party module imports `gspread` or
`googleapiclient`. If a future change introduces such an import, the
[integrations] optional-dependency must be re-declared in pyproject.toml.
This test fails loudly so the dependency declaration cannot drift again.
"""

from __future__ import annotations

import re
from pathlib import Path

PACKAGE_ROOTS = ["keyword_clustering", "tests", "app"]
FORBIDDEN_TOP_LEVEL_IMPORTS = ("gspread", "googleapiclient", "google.oauth2")
IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+([\w\.]+)", re.MULTILINE)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _python_files() -> list[Path]:
    root = _project_root()
    files: list[Path] = []
    for r in PACKAGE_ROOTS:
        sub = root / r
        if sub.exists():
            files.extend(sub.rglob("*.py"))
    return files


def test_no_first_party_imports_of_dead_integrations_extras():
    offenders: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for match in IMPORT_PATTERN.finditer(text):
            module = match.group(1)
            if any(module == bad or module.startswith(bad + ".") for bad in FORBIDDEN_TOP_LEVEL_IMPORTS):
                offenders.append(f"{path.relative_to(_project_root())} imports {module}")
    assert not offenders, (
        "First-party code imports a dead [integrations] dependency. Re-declare the "
        "extra in pyproject.toml or remove the import. Offenders:\n  " + "\n  ".join(offenders)
    )


def test_pyproject_does_not_declare_dead_integrations_extra():
    root = _project_root()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    # Allow [integrations] only if a real first-party import exists. Since the
    # previous test guards against such imports, the extra must stay removed.
    assert "\nintegrations =" not in pyproject, (
        "pyproject.toml still declares an [integrations] extra. If you've added a "
        "real GSC/GA4/Sheets connector, update test_no_first_party_imports_of_dead_integrations_extras "
        "to allow the new dependency before re-introducing the extra."
    )


def test_requirements_txt_is_not_present():
    """pyproject.toml [project.dependencies] is the single source of truth."""
    root = _project_root()
    assert not (root / "requirements.txt").exists(), (
        "requirements.txt has reappeared. Use `pip install .[app,semantic,advanced,dev]` "
        "from pyproject.toml — do not maintain two dependency lists."
    )
