"""Smoke tests for keyword_clustering.cli — argparse wiring + per-subcommand dispatch."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from keyword_clustering import cli


def _run(argv: Iterable[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["keyword-cluster", *argv])
    cli.main()


def test_build_parser_includes_every_subcommand():
    parser = cli.build_parser()
    actions = parser._subparsers._actions  # noqa: SLF001 — argparse internals
    sub_action = next(a for a in actions if hasattr(a, "choices") and "run" in (a.choices or {}))
    expected = {
        "run",
        "compare",
        "tune",
        "import-gsc",
        "import-ga4",
        "import-ahrefs",
        "import-semrush",
        "import-screamingfrog",
        "import-sitebulb",
        "crawl",
        "enrich-pages",
    }
    assert expected.issubset(set(sub_action.choices))


def test_run_help_does_not_explode(capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["keyword-cluster", "run", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    captured = capsys.readouterr().out
    assert "--keywords" in captured
    assert "--method" in captured


def test_run_subcommand_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out_dir = tmp_path / "out"
    _run(
        [
            "run",
            "--keywords",
            "examples/sample_keywords.csv",
            "--pages",
            "examples/sample_pages.csv",
            "--topics",
            "examples/sample_topics.csv",
            "--output",
            str(out_dir),
            "--method",
            "kmeans",
            "--clusters",
            "4",
            "--embedding-model",
            "tfidf",
        ],
        monkeypatch,
    )
    assert (out_dir / "clustered_keywords.csv").exists()
    assert (out_dir / "recommendations.md").exists()
    assert (out_dir / "cluster_quality_report.csv").exists()


def test_import_gsc_normalises_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    src = tmp_path / "raw_gsc.csv"
    pd.DataFrame(
        [
            {"Query": "blue widget", "Clicks": 12, "Impressions": 200, "CTR": "6%", "Position": 7.2},
            {"Query": "red gadget", "Clicks": 3, "Impressions": 88, "CTR": "3.4%", "Position": 11.5},
        ]
    ).to_csv(src, index=False)
    out = tmp_path / "gsc_normalized.csv"
    _run(["import-gsc", "--file", str(src), "--output", str(out)], monkeypatch)
    df = pd.read_csv(out)
    assert "keyword" in df.columns
    assert len(df) == 2


def test_compare_subcommand_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    a.mkdir()
    b.mkdir()
    pd.DataFrame(
        [
            {"keyword": "blue widget", "cluster_id": 0, "opportunity_score": 0.5, "recommended_url": "/a"},
            {"keyword": "red gadget", "cluster_id": 1, "opportunity_score": 0.7, "recommended_url": "/a"},
        ]
    ).to_csv(a / "clustered_keywords.csv", index=False)
    pd.DataFrame(
        [
            {"keyword": "blue widget", "cluster_id": 0, "opportunity_score": 0.55, "recommended_url": "/a"},
            {"keyword": "red gadget", "cluster_id": 2, "opportunity_score": 0.65, "recommended_url": "/b"},
        ]
    ).to_csv(b / "clustered_keywords.csv", index=False)

    out = tmp_path / "compare"
    _run(["compare", "--run-a", str(a), "--run-b", str(b), "--output", str(out)], monkeypatch)
    assert out.exists()
    files = list(out.iterdir())
    assert any(f.suffix in {".csv", ".json"} for f in files)


def test_main_with_unknown_command_prints_help(capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch):
    # No subcommand at all — should print help, not blow up.
    monkeypatch.setattr("sys.argv", ["keyword-cluster"])
    cli.main()
    captured = capsys.readouterr().out
    assert "usage:" in captured.lower() or "keyword-cluster" in captured


def test_main_surfaces_value_error_concisely(
    capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"oops": [1, 2]}).to_csv(bad, index=False)  # missing 'keyword' column
    monkeypatch.setattr("sys.argv", ["keyword-cluster", "run", "--keywords", str(bad), "--output", str(tmp_path / "o")])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    # ValueError surfaces as a concise "invalid input" message without dumping a traceback.
    assert "Error" in err
    assert "invalid input" in err.lower()
    assert "Traceback" not in err


def test_main_verbose_flag_re_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"oops": [1, 2]}).to_csv(bad, index=False)
    monkeypatch.setattr(
        "sys.argv", ["keyword-cluster", "--verbose", "run", "--keywords", str(bad), "--output", str(tmp_path / "o")]
    )
    with pytest.raises(ValueError):
        cli.main()
