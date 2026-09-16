"""Narrow native run-admission helpers extracted from upstream API runs.

The installed gateway has one bearer principal per profile. Room grants and
upstream multi-profile/session-routing features are deliberately not imported.
"""

import os
import time
from gateway.status import get_process_start_time
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore, TERMINAL_STATUSES

try:
    from aiohttp import web
except ImportError:  # The API adapter remains an optional gateway dependency.
    web = None

RUN_SCOPE = "api"


def _initialize_run_state(self):
    self._run_idempotency_store = RunIdempotencyStore()
    self._run_idempotency_ids = set()
    self._run_owner_pid = os.getpid()
    try:
        self._run_owner_started = int(get_process_start_time(self._run_owner_pid) or 0)
    except Exception:
        self._run_owner_started = 0


def _owner_alive(owner_pid, owner_started):
    """Provider liveness hint only; never proof that descendant effects ceased."""
    try:
        from gateway.status import _pid_exists
        return owner_pid > 0 and bool(_pid_exists(owner_pid)) and (
            not owner_started or int(get_process_start_time(owner_pid) or 0) == owner_started
        )
    except Exception:
        return False


def _durable_run_status(self, request, run_id):
    status = self._run_statuses.get(run_id)
    # Only locally executing or legacy unkeyed runs have an authoritative
    # memory view. Hydrated records must observe progress made by their owner.
    if status is not None and (
        run_id not in self._run_idempotency_ids or run_id in self._active_run_tasks
    ):
        return status
    record = self._run_idempotency_store.status_for_run(RUN_SCOPE, run_id)
    if record is None:
        return None
    status = dict(record["status"])
    if status.get("status") not in TERMINAL_STATUSES and not _owner_alive(
        int(record.get("owner_pid") or 0), int(record.get("owner_started") or 0),
    ):
        status.update(
            status="interrupted",
            error="Native owner unavailable or identity unverifiable; effects remain unresolved.",
            last_event="run.interrupted", updated_at=time.time(),
        )
        self._run_idempotency_store.update_status(run_id, status)
    self._run_statuses[run_id] = status
    self._run_idempotency_ids.add(run_id)
    return status


def _forget_unadmitted_run(self, run_id):
    """Release only provisional transport state, never the durable reservation."""
    self._run_streams.pop(run_id, None)
    self._run_streams_created.pop(run_id, None)
    self._run_approval_sessions.pop(run_id, None)
    self._run_statuses.pop(run_id, None)


def _accepted_response(run_id, status, gateway_session_key, *, replayed):
    """Return the upstream 202 replay envelope, without scheduling execution."""
    assert web is not None
    headers = {"Idempotency-Replayed": "true"} if replayed else {}
    if gateway_session_key:
        headers["X-Hermes-Session-Key"] = gateway_session_key
    return web.json_response(
        {"run_id": run_id, "status": status, "replayed": replayed},
        status=202, headers=headers,
    )


def _replay_or_conflict(self, request, outcome, record, gateway_session_key, _openai_error):
    """Conflict or replay the existing reservation; never create a successor."""
    assert web is not None
    if outcome == "conflict":
        return web.json_response(
            _openai_error(
                "Idempotency-Key was already used with a different request payload",
                code="idempotency_key_conflict",
            ),
            status=409,
        )
    status = self._durable_run_status(request, record["run_id"]) or record["status"]
    return _accepted_response(
        record["run_id"], status.get("status", "queued"),
        gateway_session_key, replayed=True,
    )
