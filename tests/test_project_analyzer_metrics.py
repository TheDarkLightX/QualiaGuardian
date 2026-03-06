from pathlib import Path

from guardian.cli.analyzer import ProjectAnalyzer


def test_analyze_project_reports_nonzero_average_complexity(tmp_path: Path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "module.py").write_text(
        "\n".join(
            [
                "def branchy(x):",
                "    if x > 10:",
                "        return 1",
                "    if x > 5:",
                "        return 2",
                "    return 3",
            ]
        ),
        encoding="utf-8",
    )

    analyzer = ProjectAnalyzer({})
    result = analyzer.analyze_project(str(project_dir))

    assert result["status"] in {"success", "analysis_complete", "analysis_partial"}
    assert result["metrics"]["python_files_analyzed"] >= 1
    assert result["metrics"]["average_cyclomatic_complexity"] > 0
