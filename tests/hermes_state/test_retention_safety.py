import json
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _ended(db: SessionDB, session_id: str, *, source="cron", parent=None, days=100):
    db.create_session(session_id, source=source, parent_session_id=parent)
    started = time.time() - days * 86400
    db._conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=?, end_reason=? WHERE id=?",
        (started, started + 1, "cron_complete" if source == "cron" else "agent_close", session_id),
    )


def test_retention_candidate_selection_fails_closed_for_live_lineage_handoff_and_goal(db):
    _ended(db, "eligible")
    _ended(db, "archived")
    db.set_session_archived("archived", True)

    _ended(db, "pending-handoff")
    db._conn.execute(
        "UPDATE sessions SET handoff_state='pending' WHERE id='pending-handoff'"
    )

    _ended(db, "live-parent")
    db.create_session("live-child", source="cron", parent_session_id="live-parent")

    _ended(db, "active-goal")
    db.set_meta("goal:active-goal", json.dumps({"status": "active"}))

    rows = db.list_prune_candidates(
        source="cron",
        started_before=time.time() - 7 * 86400,
        archived=False,
        retention_safe=True,
    )

    assert [row["id"] for row in rows] == ["eligible"]


def test_exact_prune_is_all_or_nothing_and_revalidates_retention_safety(db):
    _ended(db, "one")
    _ended(db, "two")

    with pytest.raises(ValueError, match="exact retention set changed"):
        db.prune_sessions(
            older_than_days=None,
            session_ids=["one", "missing"],
            require_all=True,
            archived=False,
            retention_safe=True,
        )

    assert db.get_session("one") is not None
    assert db.get_session("two") is not None

    deleted = db.prune_sessions(
        older_than_days=None,
        session_ids=["one", "two"],
        require_all=True,
        archived=False,
        retention_safe=True,
    )

    assert deleted == 2
    assert db.get_session("one") is None
    assert db.get_session("two") is None
