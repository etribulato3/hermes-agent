import gzip
import io
import json
import time

import pytest

from hermes_cli.session_retention import (
    DiskPressureError,
    RetentionLockError,
    build_archive_manifest,
    build_retention_plan,
    check_disk_guard,
    execute_cron_archive_prune,
    execute_disposable_cron_prune,
    exclusive_retention_lock,
    stream_compressed_jsonl,
)
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _ended(db, session_id, source, days, end_reason):
    db.create_session(session_id, source=source)
    started = time.time() - days * 86400
    db._conn.execute(
        "UPDATE sessions SET started_at=?, ended_at=?, end_reason=? WHERE id=?",
        (started, started + 10, end_reason, session_id),
    )
    db.append_message(session_id, "user", f"payload-{session_id}")


def test_plan_applies_source_ages_settlement_and_unknown_fail_closed(db):
    _ended(db, "cron-old", "cron", 8, "cron_complete")
    _ended(db, "cron-young", "cron", 6, "cron_complete")
    _ended(db, "agent-settled", "subagent", 31, "agent_close")
    _ended(db, "agent-unsettled", "subagent", 31, "agent_close")
    _ended(db, "human-old", "telegram", 91, "session_reset")
    _ended(db, "unknown-old", "mystery", 1000, "done")

    plan = build_retention_plan(
        db,
        now=time.time(),
        settled_session_ids={"agent-settled"},
        protected_session_ids=set(),
    )

    assert plan.session_ids("cron") == ("cron-old",)
    assert plan.session_ids("agent") == ("agent-settled",)
    assert plan.session_ids("human") == ("human-old",)
    report = plan.public_report()
    assert report["categories"]["agent"]["awaiting_exact_settlement"] == 1
    assert report["excluded"]["unknown_source"] == 1
    assert "session_ids" not in str(report)
    assert report["totals"]["estimated_logical_bytes"] > 0


def test_plan_defaults_agent_candidates_to_zero_without_settlement_evidence(db):
    _ended(db, "agent-old", "task", 60, "agent_close")

    plan = build_retention_plan(db, now=time.time())

    assert plan.session_ids("agent") == ()
    assert plan.public_report()["categories"]["agent"] == {
        "candidate_count": 0,
        "message_count": 0,
        "estimated_logical_bytes": 0,
        "awaiting_exact_settlement": 1,
    }


def test_protected_ids_are_removed_from_every_category(db):
    _ended(db, "cron-protected", "cron", 20, "cron_complete")

    plan = build_retention_plan(
        db,
        now=time.time(),
        protected_session_ids={"cron-protected"},
    )

    assert plan.session_ids("cron") == ()
    assert plan.public_report()["excluded"]["protected"] == 1


def test_stream_archive_is_compressed_jsonl_and_manifest_contains_no_content(db):
    _ended(db, "cron-archive", "cron", 20, "cron_complete")
    sink = io.BytesIO()

    receipt = stream_compressed_jsonl(db, ["cron-archive"], sink)
    compressed = sink.getvalue()
    records = [json.loads(line) for line in gzip.decompress(compressed).splitlines()]
    manifest = build_archive_manifest(
        source_profile="main",
        category="cron",
        rows=db.list_prune_candidates(session_ids=["cron-archive"]),
        receipt=receipt,
    )

    assert len(records) == 1
    assert records[0]["id"] == "cron-archive"
    assert records[0]["messages"][0]["content"] == "payload-cron-archive"
    assert receipt.session_count == 1
    assert receipt.message_count == 1
    assert receipt.compressed_size == len(compressed)
    assert len(receipt.sha256) == 64
    assert manifest["session_ids"] == ["cron-archive"]
    assert set(manifest) == {
        "format_version",
        "source_profile",
        "category",
        "session_ids",
        "started_at_min",
        "started_at_max",
        "ended_at_min",
        "ended_at_max",
        "session_count",
        "message_count",
        "compressed_size",
        "sha256",
    }
    assert "payload-cron-archive" not in json.dumps(manifest)


def test_single_instance_lock_fails_closed(tmp_path):
    lock_path = tmp_path / "retention.lock"
    with exclusive_retention_lock(lock_path):
        with pytest.raises(RetentionLockError):
            with exclusive_retention_lock(lock_path):
                pass


def test_disk_guard_stops_before_archive_when_free_space_is_critical(tmp_path):
    with pytest.raises(DiskPressureError):
        check_disk_guard(tmp_path, min_free_bytes=10**30)


class _FakeArchiveTarget:
    def __init__(self, *, valid=True, before_prune=None, available_bytes=10**12):
        self.valid = valid
        self.before_prune = before_prune
        self._available_bytes = available_bytes
        self.events = []
        self.payload = io.BytesIO()
        self.manifest = None

    def available_bytes(self):
        self.events.append("capacity")
        return self._available_bytes

    def upload_archive(self, archive_name, producer):
        self.events.append("upload_archive")
        receipt = producer(self.payload)
        return receipt

    def upload_manifest(self, manifest_name, manifest):
        self.events.append("upload_manifest")
        self.manifest = manifest

    def verify(self, archive_name, manifest_name, manifest, *, temporary):
        self.events.append("verify_temp" if temporary else "verify_final")
        if self.before_prune and not temporary:
            callback, self.before_prune = self.before_prune, None
            callback()
        return {
            "verified": self.valid,
            "session_count": manifest["session_count"],
            "message_count": manifest["message_count"],
            "sha256": manifest["sha256"],
            "sample_session_id": manifest["session_ids"][0],
            "sample_message_count": 1,
            "available_bytes": 123456,
        }

    def finalize(self, archive_name, manifest_name, manifest):
        self.events.append("finalize")


def test_verified_archive_is_pruned_by_exact_native_id_set(db, tmp_path):
    _ended(db, "cron-batch", "cron", 20, "cron_complete")
    plan = build_retention_plan(
        db, now=time.time(), batch_limits={"cron": 1, "agent": 0, "human": 0}
    )
    target = _FakeArchiveTarget()

    result = execute_cron_archive_prune(
        db,
        plan=plan,
        target=target,
        source_profile="test",
        sessions_dir=tmp_path / "sessions",
        audit_path=tmp_path / "audit.jsonl",
        now=time.time(),
        min_free_bytes=0,
    )

    assert result["candidate_count"] == 1
    assert result["archived_count"] == 1
    assert result["verified_count"] == 1
    assert result["pruned_count"] == 1
    assert db.get_session("cron-batch") is None
    assert target.events == [
        "capacity",
        "upload_archive",
        "upload_manifest",
        "verify_temp",
        "finalize",
        "verify_final",
    ]
    audit = json.loads((tmp_path / "audit.jsonl").read_text())
    assert "path" not in json.dumps(audit).lower()
    assert "payload" not in json.dumps(audit).lower()


def test_remote_capacity_is_checked_before_archive(db, tmp_path):
    _ended(db, "cron-batch", "cron", 20, "cron_complete")
    plan = build_retention_plan(db, now=time.time(), batch_limits={"cron": 1})
    target = _FakeArchiveTarget(available_bytes=0)

    with pytest.raises(DiskPressureError, match="archive target"):
        execute_cron_archive_prune(
            db,
            plan=plan,
            target=target,
            source_profile="test",
            sessions_dir=tmp_path / "sessions",
            audit_path=tmp_path / "audit.jsonl",
            now=time.time(),
            min_free_bytes=0,
        )

    assert target.events == ["capacity"]
    assert db.get_session("cron-batch") is not None


def test_verification_mismatch_prunes_nothing(db, tmp_path):
    _ended(db, "cron-batch", "cron", 20, "cron_complete")
    plan = build_retention_plan(db, now=time.time(), batch_limits={"cron": 1})

    with pytest.raises(RuntimeError, match="verification failed"):
        execute_cron_archive_prune(
            db,
            plan=plan,
            target=_FakeArchiveTarget(valid=False),
            source_profile="test",
            sessions_dir=tmp_path / "sessions",
            audit_path=tmp_path / "audit.jsonl",
            now=time.time(),
            min_free_bytes=0,
        )

    assert db.get_session("cron-batch") is not None


def test_session_becoming_live_after_archive_prunes_nothing(db, tmp_path):
    _ended(db, "cron-batch", "cron", 20, "cron_complete")
    plan = build_retention_plan(db, now=time.time(), batch_limits={"cron": 1})

    def make_lineage_live():
        db.create_session("live-child", source="cron", parent_session_id="cron-batch")

    with pytest.raises(ValueError, match="exact retention set changed"):
        execute_cron_archive_prune(
            db,
            plan=plan,
            target=_FakeArchiveTarget(before_prune=make_lineage_live),
            source_profile="test",
            sessions_dir=tmp_path / "sessions",
            audit_path=tmp_path / "audit.jsonl",
            now=time.time(),
            min_free_bytes=0,
        )

    assert db.get_session("cron-batch") is not None


def test_exact_disposable_cron_prune_never_archives_and_requires_all_ids(db, tmp_path):
    _ended(db, "disposable", "cron", 1, "cron_complete")
    _ended(db, "protected-live-parent", "cron", 20, "cron_complete")
    db.create_session(
        "protected-live-child", source="cron", parent_session_id="protected-live-parent"
    )

    with pytest.raises(ValueError, match="exact retention set changed"):
        execute_disposable_cron_prune(
            db,
            session_ids=["disposable", "protected-live-parent"],
            sessions_dir=tmp_path / "sessions",
            audit_path=tmp_path / "audit.jsonl",
            min_free_bytes=0,
        )

    assert db.get_session("disposable") is not None
    result = execute_disposable_cron_prune(
        db,
        session_ids=["disposable"],
        sessions_dir=tmp_path / "sessions",
        audit_path=tmp_path / "audit.jsonl",
        min_free_bytes=0,
    )
    assert result["pruned_count"] == 1
    assert result["archived_count"] == 0
    assert db.get_session("disposable") is None


def test_exact_disposable_cron_prune_accepts_same_ids_in_any_order(db, tmp_path):
    _ended(db, "z-older", "cron", 20, "cron_complete")
    _ended(db, "a-newer", "cron", 10, "cron_complete")

    result = execute_disposable_cron_prune(
        db,
        session_ids=["a-newer", "z-older"],
        sessions_dir=tmp_path / "sessions",
        audit_path=tmp_path / "audit.jsonl",
        min_free_bytes=0,
    )

    assert result["pruned_count"] == 2
    assert db.get_session("a-newer") is None
    assert db.get_session("z-older") is None
