import json
import sys


def test_sessions_retention_is_dry_run_and_read_only_by_default(monkeypatch, capsys, tmp_path):
    import hermes_cli.main as main_mod
    import hermes_state

    opened = []

    class FakeDB:
        db_path = tmp_path / "state.db"

        def __init__(self, read_only=False):
            opened.append(read_only)

        def list_prune_candidates(self, **kwargs):
            return []

        def estimate_sessions_logical_bytes(self, session_ids):
            return 0, 0

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", FakeDB)
    monkeypatch.setattr(
        sys, "argv", ["hermes", "sessions", "retention", "--profile-label", "test"]
    )

    main_mod.main()

    report = json.loads(capsys.readouterr().out)
    assert opened == [True]
    assert report["dry_run"] is True
    assert report["profile"] == "test"
    assert report["totals"]["candidate_count"] == 0


def test_sessions_retention_rejects_execute_without_exact_authorization(monkeypatch, capsys):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "sessions",
            "retention",
            "--execute-cron-archive-prune",
            "--archive-host",
            "MM1",
        ],
    )

    main_mod.main()

    assert "requires --authorization ARCHIVE_AND_PRUNE_EXACT_IDS" in capsys.readouterr().out
