from guardian.cli import _root_cli_module as guardian_cli


def test_main_routes_quality_alias_to_analysis(monkeypatch):
    captured = {}

    def fake_run_analysis(args):
        captured["project_path"] = args.project_path
        return 17

    monkeypatch.setattr(guardian_cli, "run_analysis", fake_run_analysis)

    result = guardian_cli.main(["quality", "/tmp/example-project", "--output-format", "json"])

    assert result == 17
    assert captured["project_path"] == "/tmp/example-project"


def test_main_routes_typer_subcommands(monkeypatch):
    captured = {}

    def fake_app(*, prog_name, args):
        captured["prog_name"] = prog_name
        captured["args"] = args
        return 23

    monkeypatch.setattr(guardian_cli, "app", fake_app)

    result = guardian_cli.main(["gamify", "--help"])

    assert result == 23
    assert captured["prog_name"] == "guardian"
    assert captured["args"] == ["gamify", "--help"]
