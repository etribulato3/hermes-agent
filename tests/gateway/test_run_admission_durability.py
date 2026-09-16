"""Native run-admission regression tests; no model or live gateway execution."""


import pytest


@pytest.mark.parametrize("terminal_status", ["interrupted", "completed", "failed", "cancelled"])
@pytest.mark.parametrize("explicit_deadline", [False, True])
def test_expiry_does_not_release_unacknowledged_run(
    tmp_path, monkeypatch, terminal_status, explicit_deadline,
):
    from gateway.platforms import api_server_run_idempotency as module

    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    store = module.RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        outcome, _ = store.reserve(
            "tenant", "operation", "fingerprint", "original-run",
            {"status": terminal_status},
            retention_until=1001.0 if explicit_deadline else 0,
        )
        assert outcome == "created"
        now[0] += 2 * store.RETENTION_SECONDS
        outcome, record = store.reserve(
            "tenant", "operation", "fingerprint", "duplicate-run",
            {"status": "running"},
        )
        assert outcome == "reused"
        assert record["run_id"] == "original-run"
        assert record["status"] == {"status": terminal_status}
    finally:
        store.close()


def test_scope_conflict_and_reopen_preserve_original_reservation(tmp_path):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    path = str(tmp_path / "runs.db")
    store = RunIdempotencyStore(path)
    assert store.durable
    try:
        outcome, original = store.reserve(
            "tenant", "operation", "fingerprint", "original-run",
            {"status": "running"}, owner_pid=123, owner_started=456,
        )
        assert outcome == "created"
    finally:
        store.close()
    reopened = RunIdempotencyStore(path)
    try:
        outcome, replay = reopened.lookup("tenant", "operation", "fingerprint")
        assert (outcome, replay) == ("reused", original)
        outcome, conflict = reopened.reserve(
            "tenant", "operation", "different-fingerprint", "duplicate-run",
            {"status": "running"},
        )
        assert (outcome, conflict) == ("conflict", original)
        assert reopened.lookup("other-tenant", "operation", "fingerprint") == ("missing", None)
        assert reopened.status_for_run("other-tenant", "original-run") is None
        assert not reopened.owns_run("other-tenant", "original-run")
        status = reopened.status_for_run("tenant", "original-run")
        assert status is not None
        assert status["status"] == {"status": "running"}
    finally:
        reopened.close()


def test_two_processes_contend_for_one_reservation(tmp_path):
    import json
    import os
    import subprocess
    import sys
    import time
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    database = str(tmp_path / "runs.db")
    RunIdempotencyStore(database).close()
    script = r"""
import json, os, sys, time
from pathlib import Path
sys.path[:] = json.loads(sys.argv[1])
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
root, run_id = Path(sys.argv[2]), sys.argv[3]
store = RunIdempotencyStore(str(root / 'runs.db'))
(root / (run_id + '.ready')).write_text('ready')
deadline = time.monotonic() + 10
while not (root / 'go').exists():
    assert time.monotonic() < deadline, 'admission barrier timed out'
    time.sleep(0.01)
try:
    outcome, record = store.reserve('tenant', 'operation', 'fingerprint', run_id,
                                    {'status': 'running'}, owner_pid=os.getpid())
    print(json.dumps({'pid': os.getpid(), 'outcome': outcome, 'record': record}), flush=True)
finally:
    store.close()
"""
    children = []
    results = []
    try:
        for run_id in ("run-a", "run-b"):
            children.append(subprocess.Popen(
                [sys.executable, "-I", "-B", "-c", script, json.dumps(sys.path), str(tmp_path), run_id],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, close_fds=True,
            ))
        deadline = time.monotonic() + 15
        while not all((tmp_path / (run_id + ".ready")).exists() for run_id in ("run-a", "run-b")):
            assert all(child.poll() is None for child in children), "admission child exited before barrier"
            assert time.monotonic() < deadline, "children did not reach admission barrier"
            time.sleep(0.01)
        (tmp_path / "go").write_text("go")
        for child in children:
            stdout, stderr = child.communicate(timeout=15)
            assert child.returncode == 0, stderr
            results.append(json.loads(stdout))
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            stdout, stderr = child.communicate(timeout=5)
            (tmp_path / f"child-{child.pid}.stdout").write_text(stdout)
            (tmp_path / f"child-{child.pid}.stderr").write_text(stderr)
    assert len({item["pid"] for item in results} | {os.getpid()}) == 3
    assert sorted(item["outcome"] for item in results) == ["created", "reused"]
    created = next(item for item in results if item["outcome"] == "created")
    assert all(item["record"] == created["record"] for item in results)
    assert created["record"]["owner_pid"] == created["pid"]
    reopened = RunIdempotencyStore(database)
    try:
        assert reopened.lookup("tenant", "operation", "fingerprint") == ("reused", created["record"])
    finally:
        reopened.close()
    (tmp_path / "process-admission-receipt.json").write_text(json.dumps(results, indent=2))


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_failure", [False, True])
async def test_keyed_admission_rejects_nondurable_store_before_reading_body(tmp_path, monkeypatch, storage_failure):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-token", "model_name": "fixture-model"}))
    initial = getattr(adapter, "_run_idempotency_store", None)
    if initial is not None:
        initial.close()
    path = str(tmp_path / "absent-directory" / "runs.db") if storage_failure else ":memory:"
    store = RunIdempotencyStore(path)
    assert not store.durable
    adapter._run_idempotency_store = store
    request = SimpleNamespace(
        headers={"Authorization": "Bearer fixture-token", "Idempotency-Key": "operation"},
        json=AsyncMock(return_value={}),
    )
    try:
        response = await adapter._handle_runs(request)
        assert response.status == 503
        request.json.assert_not_awaited()
        assert not adapter._active_run_tasks
        assert not adapter._run_streams
    finally:
        store.close()
        adapter._response_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("conflicting_payload", [False, True])
async def test_keyed_replay_or_conflict_precedes_capacity_limit(tmp_path, monkeypatch, conflicting_payload):
    import hashlib
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-token", "model_name": "fixture-model"}))
    original_body = {"input": "fixture"}
    fingerprint = hashlib.sha256(json.dumps(
        {"body": original_body, "gateway_session_key": ""},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    adapter._run_idempotency_store.reserve(
        "api", "operation", fingerprint, "existing-run",
        {"status": "interrupted", "run_id": "existing-run"},
    )
    adapter._max_concurrent_runs = 1
    adapter._inflight_agent_runs = 1
    create = Mock(side_effect=AssertionError("replay must never create an agent"))
    monkeypatch.setattr(adapter, "_create_agent", create)
    body = {"input": "different"} if conflicting_payload else original_body
    request = SimpleNamespace(
        headers={"Authorization": "Bearer fixture-token", "Idempotency-Key": "operation"},
        json=AsyncMock(return_value=body),
    )
    try:
        response = await adapter._handle_runs(request)
        assert response.status == (409 if conflicting_payload else 202)
        data = json.loads(response.text)
        if conflicting_payload:
            assert data["error"]["code"] == "idempotency_key_conflict"
        else:
            assert data["run_id"] == "existing-run"
            assert data["status"] == "interrupted"
            assert response.headers["Idempotency-Replayed"] == "true"
        create.assert_not_called()
        assert not adapter._active_run_tasks
        assert not adapter._run_streams
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_reservation_is_committed_before_task_scheduling(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-token", "model_name": "fixture-model"}))
    create = Mock(side_effect=AssertionError("unit test must not invoke a real agent"))
    monkeypatch.setattr(adapter, "_create_agent", create)
    scheduled = []
    original_schedule = asyncio.create_task

    def require_committed_reservation(coroutine, *args, **kwargs):
        try:
            row = adapter._run_idempotency_store._conn.execute(
                "SELECT COUNT(*) AS reservations FROM run_idempotency WHERE scope=? AND idempotency_key=?",
                ("api", "operation"),
            ).fetchone()
            assert row["reservations"] == 1, "task scheduled before durable reservation"
        except BaseException:
            coroutine.close()
            raise
        task = original_schedule(coroutine, *args, **kwargs)
        scheduled.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", require_committed_reservation)
    request = SimpleNamespace(
        headers={"Authorization": "Bearer fixture-token", "Idempotency-Key": "operation"},
        json=AsyncMock(return_value={"input": "fixture"}),
    )
    try:
        first = await adapter._handle_runs(request)
        second = await adapter._handle_runs(request)
        assert first.status == second.status == 202
        assert json.loads(first.text)["run_id"] == json.loads(second.text)["run_id"]
        assert second.headers["Idempotency-Replayed"] == "true"
        assert len(scheduled) == 1
        create.assert_not_called()
    finally:
        for task in scheduled:
            task.cancel()
        await asyncio.gather(*scheduled, return_exceptions=True)
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled", "interrupted"])
async def test_terminal_status_survives_cold_adapter_reopen(tmp_path, monkeypatch, terminal):
    import hashlib
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = PlatformConfig(enabled=True, extra={"key": "fixture-token", "model_name": "fixture-model"})
    original = APIServerAdapter(config)
    body = {"input": "fixture"}
    fingerprint = hashlib.sha256(json.dumps(
        {"body": body, "gateway_session_key": ""},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    try:
        original._run_idempotency_store.reserve(
            "api", "operation", fingerprint, "existing-run",
            original._set_run_status("existing-run", "queued"),
        )
        original._run_idempotency_ids.add("existing-run")
        original._set_run_status("existing-run", terminal, final_response="fixture-result")
    finally:
        await original.disconnect()
    reopened = APIServerAdapter(config)
    create = Mock(side_effect=AssertionError("cold replay must not launch an agent"))
    monkeypatch.setattr(reopened, "_create_agent", create)
    request = SimpleNamespace(
        headers={"Authorization": "Bearer fixture-token", "Idempotency-Key": "operation"},
        match_info={"run_id": "existing-run"}, json=AsyncMock(return_value=body),
    )
    try:
        assert not reopened._run_statuses
        response = await reopened._handle_get_run(request)
        assert response.status == 200, "durable status vanished after adapter replacement"
        status = json.loads(response.text)
        assert status["status"] == terminal
        assert status["final_response"] == "fixture-result"
        replay = await reopened._handle_runs(request)
        assert replay.status == 202
        assert json.loads(replay.text)["status"] == terminal
        assert json.loads(replay.text)["run_id"] == "existing-run"
        assert replay.headers["Idempotency-Replayed"] == "true"
        create.assert_not_called()
        assert not reopened._active_run_tasks
    finally:
        await reopened.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_read", ["status", "replay"])
async def test_killed_admission_owner_replays_interrupted_without_new_task(tmp_path, monkeypatch, first_read):
    import asyncio
    import json
    import os
    import signal
    import subprocess
    import sys
    import time
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    script = r"""
import asyncio, json, os, signal, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
sys.path[:] = json.loads(sys.argv[1])
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
root = Path(sys.argv[2])
async def once():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-token", "model_name": "fixture-model"}))
    def forbid_agent(**kwargs):
        (root / 'unexpected-agent').write_text('created')
        raise AssertionError('no model invocation allowed')
    adapter._create_agent = forbid_agent
    response = await adapter._handle_runs(SimpleNamespace(
        headers={"Authorization": "Bearer fixture-token", "Idempotency-Key": "operation"},
        json=AsyncMock(return_value={"input": "fixture"})))
    assert response.status == 202
    result = json.loads(response.text)
    result.update(pid=os.getpid(), boundary="accepted-before-task-execution")
    (root / 'owner-ready.json').write_text(json.dumps(result))
    command = sys.stdin.buffer.read(1)  # Park before yielding to the task.
    if command == b"K":
        os.kill(os.getpid(), signal.SIGKILL)  # Only the child itself is targeted.
    os._exit(2)  # EOF must not yield and accidentally execute the queued task.
asyncio.run(once())
"""
    child = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", script, json.dumps(sys.path), str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, close_fds=True,
    )
    try:
        deadline = time.monotonic() + 15
        ready = tmp_path / "owner-ready.json"
        while not ready.exists():
            assert child.poll() is None, "owner exited before admission boundary"
            assert time.monotonic() < deadline, "owner failed to reach admission boundary"
            await asyncio.sleep(0.01)
        admitted = json.loads(ready.read_text())
        assert admitted["pid"] == child.pid != os.getpid()
        stdout, stderr = child.communicate(input="K", timeout=5)
        assert child.returncode == -signal.SIGKILL
    finally:
        if child.poll() is None and child.stdin is not None and not child.stdin.closed:
            child.stdin.close()
            child.stdin = None
        stdout, stderr = child.communicate(timeout=5)
        (tmp_path / "owner.stdout").write_text(stdout)
        (tmp_path / "owner.stderr").write_text(stderr)
    assert not (tmp_path / "unexpected-agent").exists()
    replacement = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-token", "model_name": "fixture-model"}))
    create = Mock(side_effect=AssertionError("replacement must not create an agent"))
    monkeypatch.setattr(replacement, "_create_agent", create)
    request = SimpleNamespace(
        headers={"Authorization": "Bearer fixture-token", "Idempotency-Key": "operation"},
        match_info={"run_id": admitted["run_id"]}, json=AsyncMock(return_value={"input": "fixture"}),
    )
    try:
        handler = replacement._handle_get_run if first_read == "status" else replacement._handle_runs
        response = await handler(request)
        assert response.status == (200 if first_read == "status" else 202)
        assert json.loads(response.text)["status"] == "interrupted"
        replay = await replacement._handle_runs(request)
        assert json.loads(replay.text)["status"] == "interrupted"
        assert json.loads(replay.text)["run_id"] == admitted["run_id"]
        assert replay.headers["Idempotency-Replayed"] == "true"
        create.assert_not_called()
        assert not replacement._active_run_tasks
        stored = replacement._run_idempotency_store.status_for_run("api", admitted["run_id"])
        assert stored["status"]["status"] == "interrupted"
        (tmp_path / "owner-replacement-receipt.json").write_text(json.dumps({
            "original": admitted, "exit_code": child.returncode,
            "replacement_pid": os.getpid(), "status": stored,
            "scope": "admission process death only; no descendant-effect cessation claim",
        }, sort_keys=True))
    finally:
        await replacement.disconnect()


@pytest.mark.asyncio
async def test_surviving_owner_progress_refreshes_replacement_cache(tmp_path, monkeypatch):
    import json
    import os
    from types import SimpleNamespace
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = PlatformConfig(enabled=True, extra={"key": "fixture-token", "model_name": "fixture-model"})
    owner = APIServerAdapter(config)
    observer = APIServerAdapter(config)
    # Cache-only unit seam: confinement intentionally denies procfs identity
    # reads. Keep the known current-process oracle local, not a live-owner proof.
    from gateway.platforms import api_server_runs
    monkeypatch.setattr(api_server_runs, "_owner_alive", lambda pid, birth:
                        pid == os.getpid() and birth == owner._run_owner_started)
    request = SimpleNamespace(headers={"Authorization": "Bearer fixture-token"}, match_info={"run_id": "existing-run"})
    try:
        owner._run_idempotency_store.reserve(
            "api", "operation", "fingerprint", "existing-run", {"status": "running"},
            owner_pid=os.getpid(), owner_started=owner._run_owner_started,
        )
        first = await observer._handle_get_run(request)
        assert json.loads(first.text)["status"] == "running"
        owner._run_idempotency_store.update_status("existing-run", {
            "run_id": "existing-run", "status": "completed", "output": "fixture-result",
        })
        second = await observer._handle_get_run(request)
        assert json.loads(second.text)["status"] == "completed", "replacement cached stale owner status"
        assert json.loads(second.text)["output"] == "fixture-result"
        assert not observer._active_run_tasks
    finally:
        await observer.disconnect()
        await owner.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [False, True])
async def test_lost_reservation_race_discards_only_provisional_transport(tmp_path, monkeypatch, conflict):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-token", "model_name": "fixture-model"}))
    store = adapter._run_idempotency_store
    competitor = RunIdempotencyStore(store._db_path)
    lookup = store.lookup
    def race_after_lookup(scope, key, fingerprint):
        result = lookup(scope, key, fingerprint)
        assert result == ("missing", None)
        outcome, _ = competitor.reserve(scope, key, "different" if conflict else fingerprint,
                                         "winning-run", {"run_id": "winning-run", "status": "completed"})
        assert outcome == "created"
        return result
    monkeypatch.setattr(store, "lookup", race_after_lookup)
    create = Mock(side_effect=AssertionError("losing caller must not launch"))
    monkeypatch.setattr(adapter, "_create_agent", create)
    request = SimpleNamespace(
        headers={"Authorization": "Bearer fixture-token", "Idempotency-Key": "operation"},
        json=AsyncMock(return_value={"input": "fixture"}),
    )
    try:
        response = await adapter._handle_runs(request)
        assert response.status == (409 if conflict else 202)
        if not conflict:
            assert json.loads(response.text)["run_id"] == "winning-run"
            assert response.headers["Idempotency-Replayed"] == "true"
        assert not adapter._run_streams
        assert not adapter._run_streams_created
        assert not adapter._run_approval_sessions
        assert not adapter._active_run_tasks
        assert set(adapter._run_statuses) <= {"winning-run"}
        create.assert_not_called()
        assert competitor.status_for_run("api", "winning-run")["status"]["status"] == "completed"
    finally:
        competitor.close()
        await adapter.disconnect()
