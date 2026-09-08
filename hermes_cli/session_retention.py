"""Fail-closed, source-aware session retention planning.

The planner is read-only.  Archival and exact-ID pruning are separate explicit
operations so installing this module cannot activate deletion or scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import gzip
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Mapping, Optional, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses msvcrt below
    fcntl = None
    import msvcrt


POLICY_VERSION = "source-aware-v1"
RETENTION_DAYS = {"cron": 7, "agent": 30, "human": 90}


class RetentionLockError(RuntimeError):
    pass


class DiskPressureError(RuntimeError):
    pass


@contextmanager
def exclusive_retention_lock(path: Path):
    """Acquire the process-wide retention lock without waiting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    try:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - exercised on Windows CI
                handle.seek(0)
                handle.write(b"\0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except (BlockingIOError, OSError) as exc:
            raise RetentionLockError("another retention run holds the lock") from exc
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        handle.close()


def check_disk_guard(path: Path, *, min_free_bytes: int) -> int:
    """Return free bytes or stop before archive/VACUUM work."""
    free = shutil.disk_usage(path).free
    if free < min_free_bytes:
        raise DiskPressureError("critical disk pressure; archive and prune disabled")
    return free


SOURCE_CLASSES = {
    "cron": frozenset({"cron"}),
    "agent": frozenset({"agent", "subagent", "task", "kanban"}),
    "human": frozenset(
        {
            "cli",
            "tui",
            "desktop",
            "api",
            "telegram",
            "discord",
            "slack",
            "whatsapp",
            "signal",
            "matrix",
            "mattermost",
            "teams",
            "email",
            "sms",
            "imessage",
            "line",
            "simplex",
            "google_chat",
            "dingtalk",
            "feishu",
            "wecom",
            "weixin",
            "yuanbao",
        }
    ),
}


@dataclass(frozen=True)
class RetentionCategory:
    session_ids: tuple[str, ...]
    message_count: int
    estimated_logical_bytes: int
    awaiting_exact_settlement: int = 0


@dataclass(frozen=True)
class RetentionPlan:
    categories: Mapping[str, RetentionCategory]
    unknown_source_count: int
    protected_count: int

    def session_ids(self, category: str) -> tuple[str, ...]:
        return self.categories[category].session_ids

    def public_report(self) -> Dict[str, Any]:
        categories: Dict[str, Dict[str, int]] = {}
        total_count = total_messages = total_bytes = 0
        for name in ("cron", "agent", "human"):
            category = self.categories[name]
            row = {
                "candidate_count": len(category.session_ids),
                "message_count": category.message_count,
                "estimated_logical_bytes": category.estimated_logical_bytes,
            }
            if name == "agent":
                row["awaiting_exact_settlement"] = category.awaiting_exact_settlement
            categories[name] = row
            total_count += len(category.session_ids)
            total_messages += category.message_count
            total_bytes += category.estimated_logical_bytes
        return {
            "policy_version": POLICY_VERSION,
            "dry_run": True,
            "retention_days": dict(RETENTION_DAYS),
            "categories": categories,
            "excluded": {
                "unknown_source": self.unknown_source_count,
                "protected": self.protected_count,
            },
            "totals": {
                "candidate_count": total_count,
                "message_count": total_messages,
                "estimated_logical_bytes": total_bytes,
            },
            "physical_reclaim_bytes": None,
        }


def _ids(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(row["id"]) for row in rows]


def build_retention_plan(
    db,
    *,
    now: float,
    settled_session_ids: Optional[Iterable[str]] = None,
    protected_session_ids: Optional[Iterable[str]] = None,
    batch_limits: Optional[Mapping[str, int]] = None,
) -> RetentionPlan:
    """Build a native-selection retention plan without mutating the store.

    Agent/task rows require an exact external settlement set.  An absent set is
    deliberately equivalent to an empty set.  Unknown source values are only
    counted as excluded and can never become candidates.
    """
    settled = {str(value) for value in (settled_session_ids or ())}
    protected = {str(value) for value in (protected_session_ids or ())}
    limits = dict(batch_limits or {})
    known_sources = frozenset().union(*SOURCE_CLASSES.values())
    protected_count = 0
    categories: Dict[str, RetentionCategory] = {}

    for name in ("cron", "agent", "human"):
        rows = []
        cutoff = now - RETENTION_DAYS[name] * 86400
        for source in sorted(SOURCE_CLASSES[name]):
            filters: Dict[str, Any] = {
                "source": source,
                "started_before": cutoff,
                "archived": False,
                "retention_safe": True,
            }
            if name == "cron":
                filters["end_reason"] = "cron_complete"
            rows.extend(db.list_prune_candidates(**filters))
        rows.sort(key=lambda row: (row.get("started_at") or 0, str(row["id"])))

        before_protection = len(rows)
        rows = [row for row in rows if str(row["id"]) not in protected]
        protected_count += before_protection - len(rows)

        awaiting_settlement = 0
        if name == "agent":
            awaiting_settlement = sum(
                1 for row in rows if str(row["id"]) not in settled
            )
            rows = [row for row in rows if str(row["id"]) in settled]

        limit = limits.get(name)
        if limit is not None:
            if limit < 0:
                raise ValueError("batch limits must be non-negative")
            rows = rows[:limit]

        selected_ids = _ids(rows)
        logical_bytes, message_count = db.estimate_sessions_logical_bytes(selected_ids)
        categories[name] = RetentionCategory(
            tuple(selected_ids), message_count, logical_bytes, awaiting_settlement
        )

    unknown_rows = db.list_prune_candidates(
        started_before=now - RETENTION_DAYS["cron"] * 86400,
        archived=False,
        retention_safe=True,
    )
    unknown_count = sum(
        1 for row in unknown_rows if str(row.get("source") or "") not in known_sources
    )
    return RetentionPlan(categories, unknown_count, protected_count)


def load_exact_session_ids(path) -> set[str]:
    """Load a newline-delimited exact-session allow/protection set."""
    if path is None:
        return set()
    with open(path, "r", encoding="utf-8") as handle:
        return {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


@dataclass(frozen=True)
class ArchiveReceipt:
    session_count: int
    message_count: int
    compressed_size: int
    sha256: str


class _DigestWriter:
    def __init__(self, raw: BinaryIO):
        self.raw = raw
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> int:
        written = self.raw.write(data)
        if written is None:
            written = len(data)
        if written != len(data):
            raise OSError("short archive write")
        self.digest.update(data)
        self.size += written
        return written

    def flush(self) -> None:
        self.raw.flush()


def stream_compressed_jsonl(
    db, session_ids: Sequence[str], sink: BinaryIO
) -> ArchiveReceipt:
    """Stream exact native session exports as gzip JSONL into *sink*."""
    exact_ids = list(dict.fromkeys(str(sid) for sid in session_ids))
    writer = _DigestWriter(sink)
    message_count = 0
    with gzip.GzipFile(fileobj=writer, mode="wb", mtime=0) as archive:
        for session_id in exact_ids:
            record = db.export_session(session_id)
            if record is None:
                raise ValueError("exact archive set changed")
            messages = record.get("messages") or []
            message_count += len(messages)
            payload = json.dumps(
                record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            archive.write(payload + b"\n")
    return ArchiveReceipt(
        session_count=len(exact_ids),
        message_count=message_count,
        compressed_size=writer.size,
        sha256=writer.digest.hexdigest(),
    )


def build_archive_manifest(
    *,
    source_profile: str,
    category: str,
    rows: Sequence[Mapping[str, Any]],
    receipt: ArchiveReceipt,
) -> Dict[str, Any]:
    """Build the content-free integrity manifest for one archive batch."""
    session_ids = [str(row["id"]) for row in rows]
    if len(session_ids) != receipt.session_count:
        raise ValueError("manifest rows do not match archive receipt")

    def _bound(field: str, fn):
        values = [row.get(field) for row in rows if row.get(field) is not None]
        return fn(values) if values else None

    return {
        "format_version": 1,
        "source_profile": source_profile,
        "category": category,
        "session_ids": session_ids,
        "started_at_min": _bound("started_at", min),
        "started_at_max": _bound("started_at", max),
        "ended_at_min": _bound("ended_at", min),
        "ended_at_max": _bound("ended_at", max),
        "session_count": receipt.session_count,
        "message_count": receipt.message_count,
        "compressed_size": receipt.compressed_size,
        "sha256": receipt.sha256,
    }


def _write_audit(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n")


class SSHArchiveTarget:
    """MM1-style SSH target with temporary upload and atomic publication."""

    def __init__(self, host: str, root: str = "HermesArchive"):
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host):
            raise ValueError("invalid archive host")
        root_path = Path(root)
        if root_path.is_absolute() or ".." in root_path.parts or not root_path.parts:
            raise ValueError("archive root must be a safe home-relative directory")
        if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in root_path.parts):
            raise ValueError("invalid archive root")
        self.host = host
        self.root = root_path.as_posix()

    @staticmethod
    def _safe_name(name: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid archive filename")
        return name

    def _command(self, script: str, *args: str) -> list[str]:
        remote = "python3 -c " + shlex.quote(script)
        if args:
            remote += " " + " ".join(shlex.quote(str(arg)) for arg in args)
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            self.host,
            remote,
        ]

    def _run(self, script: str, *args: str, input_bytes: Optional[bytes] = None):
        result = subprocess.run(
            self._command(script, *args),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("remote archive operation failed")
        return result.stdout

    def available_bytes(self) -> int:
        script = """import pathlib,shutil,sys
root=pathlib.Path.home()/sys.argv[1]
root.mkdir(parents=True,exist_ok=True)
print(shutil.disk_usage(root).free)
"""
        try:
            return int(self._run(script, self.root).strip())
        except (TypeError, ValueError) as exc:
            raise RuntimeError("remote capacity check returned invalid data") from exc

    def upload_archive(self, archive_name: str, producer) -> ArchiveReceipt:
        name = self._safe_name(archive_name)
        script = """import pathlib,sys
root=pathlib.Path.home()/sys.argv[1]
root.mkdir(parents=True,exist_ok=True)
target=root/('.'+sys.argv[2]+'.tmp')
with target.open('wb') as handle:
    while True:
        block=sys.stdin.buffer.read(1024*1024)
        if not block: break
        handle.write(block)
"""
        process = subprocess.Popen(
            self._command(script, self.root, name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        try:
            receipt = producer(process.stdin)
            process.stdin.close()
            returncode = process.wait()
        except Exception:
            process.kill()
            process.wait()
            raise
        if returncode != 0:
            raise RuntimeError("remote archive upload failed")
        return receipt

    def upload_manifest(
        self, manifest_name: str, manifest: Mapping[str, Any]
    ) -> None:
        name = self._safe_name(manifest_name)
        script = """import pathlib,sys
root=pathlib.Path.home()/sys.argv[1]
root.mkdir(parents=True,exist_ok=True)
(root/('.'+sys.argv[2]+'.tmp')).write_bytes(sys.stdin.buffer.read())
"""
        payload = (json.dumps(dict(manifest), sort_keys=True) + "\n").encode("utf-8")
        self._run(script, self.root, name, input_bytes=payload)

    def verify(
        self,
        archive_name: str,
        manifest_name: str,
        manifest: Mapping[str, Any],
        *,
        temporary: bool,
    ) -> Dict[str, Any]:
        archive = self._safe_name(archive_name)
        manifest_file = self._safe_name(manifest_name)
        script = """import gzip,hashlib,json,pathlib,shutil,sys
root=pathlib.Path.home()/sys.argv[1]
temporary=sys.argv[4]=='1'
def p(name): return root/(('.'+name+'.tmp') if temporary else name)
a=p(sys.argv[2]); m=p(sys.argv[3])
data=json.loads(m.read_text(encoding='utf-8'))
h=hashlib.sha256(); size=0; sessions=messages=0; ids=[]; sample_id=None; sample_messages=None
with a.open('rb') as raw:
    while True:
        block=raw.read(1024*1024)
        if not block: break
        h.update(block); size+=len(block)
with gzip.open(a,'rt',encoding='utf-8') as handle:
    for line in handle:
        record=json.loads(line); sid=str(record['id']); rows=record.get('messages') or []
        if sample_id is None: sample_id=sid; sample_messages=len(rows)
        ids.append(sid); sessions+=1; messages+=len(rows)
verified=(h.hexdigest()==data['sha256'] and size==data['compressed_size'] and sessions==data['session_count'] and messages==data['message_count'] and ids==data['session_ids'])
print(json.dumps({'verified':verified,'session_count':sessions,'message_count':messages,'sha256':h.hexdigest(),'sample_session_id':sample_id,'sample_message_count':sample_messages,'available_bytes':shutil.disk_usage(root).free},sort_keys=True))
"""
        output = self._run(
            script,
            self.root,
            archive,
            manifest_file,
            "1" if temporary else "0",
        )
        try:
            return json.loads(output)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("remote verification returned invalid data") from exc

    def finalize(
        self, archive_name: str, manifest_name: str, manifest: Mapping[str, Any]
    ) -> None:
        archive = self._safe_name(archive_name)
        manifest_file = self._safe_name(manifest_name)
        script = """import json,os,pathlib,sys
root=pathlib.Path.home()/sys.argv[1]
a=sys.argv[2]; m=sys.argv[3]
os.replace(root/('.'+a+'.tmp'),root/a)
os.replace(root/('.'+m+'.tmp'),root/m)
manifest=json.loads((root/m).read_text(encoding='utf-8'))
entry={'archive_filename':a,'manifest_filename':m,'source_profile':manifest['source_profile'],'category':manifest['category'],'session_ids':manifest['session_ids'],'started_at_min':manifest['started_at_min'],'started_at_max':manifest['started_at_max'],'sha256':manifest['sha256']}
catalogue=root/'catalogue.jsonl'; rows=[]
if catalogue.exists():
    rows=[json.loads(line) for line in catalogue.read_text(encoding='utf-8').splitlines() if line.strip()]
rows=[row for row in rows if row.get('sha256')!=entry['sha256']]; rows.append(entry)
tmp=root/'.catalogue.jsonl.tmp'; tmp.write_text(''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\\n' for row in rows),encoding='utf-8'); os.replace(tmp,catalogue)
"""
        self._run(script, self.root, archive, manifest_file)

    @property
    def display_root(self) -> str:
        return f"~/{self.root}"


def execute_cron_archive_prune(
    db,
    *,
    plan: RetentionPlan,
    target,
    source_profile: str,
    sessions_dir: Path,
    audit_path: Path,
    now: float,
    min_free_bytes: int,
) -> Dict[str, Any]:
    """Archive, verify, then atomically prune one exact cron batch."""
    check_disk_guard(db.db_path.parent, min_free_bytes=min_free_bytes)
    session_ids = plan.session_ids("cron")
    if not session_ids:
        return {
            "candidate_count": 0,
            "archived_count": 0,
            "verified_count": 0,
            "pruned_count": 0,
        }

    remote_free = target.available_bytes()
    remote_required = max(
        1024 * 1024 * 1024,
        plan.categories["cron"].estimated_logical_bytes * 2,
    )
    if remote_free < remote_required:
        raise DiskPressureError("archive target has insufficient free space; pruned nothing")

    fingerprint = hashlib.sha256("\n".join(session_ids).encode("utf-8")).hexdigest()[:16]
    archive_name = f"hermes-sessions-{source_profile}-cron-{fingerprint}.jsonl.gz"
    manifest_name = f"{archive_name}.manifest.json"
    cutoff = now - RETENTION_DAYS["cron"] * 86400
    rows = db.list_prune_candidates(
        source="cron",
        end_reason="cron_complete",
        started_before=cutoff,
        archived=False,
        retention_safe=True,
        session_ids=session_ids,
    )
    if [str(row["id"]) for row in rows] != list(session_ids):
        raise ValueError("exact retention set changed; archived and pruned nothing")

    receipt = target.upload_archive(
        archive_name,
        lambda sink: stream_compressed_jsonl(db, session_ids, sink),
    )
    manifest = build_archive_manifest(
        source_profile=source_profile,
        category="cron",
        rows=rows,
        receipt=receipt,
    )
    target.upload_manifest(manifest_name, manifest)

    def _verified(result: Mapping[str, Any]) -> bool:
        return bool(result.get("verified")) and all(
            result.get(key) == manifest[key]
            for key in ("session_count", "message_count", "sha256")
        )

    temporary_verification = target.verify(
        archive_name, manifest_name, manifest, temporary=True
    )
    if not _verified(temporary_verification):
        raise RuntimeError("archive verification failed; pruned nothing")
    target.finalize(archive_name, manifest_name, manifest)
    final_verification = target.verify(
        archive_name, manifest_name, manifest, temporary=False
    )
    if not _verified(final_verification):
        raise RuntimeError("final archive verification failed; pruned nothing")

    pruned = db.prune_sessions(
        older_than_days=None,
        source="cron",
        end_reason="cron_complete",
        started_before=cutoff,
        archived=False,
        retention_safe=True,
        session_ids=session_ids,
        require_all=True,
        sessions_dir=sessions_dir,
    )
    if pruned != len(session_ids):  # defensive: require_all should make this impossible
        raise RuntimeError("native exact prune count mismatch")

    result = {
        "candidate_count": len(session_ids),
        "archived_count": receipt.session_count,
        "verified_count": int(final_verification["session_count"]),
        "pruned_count": pruned,
        "message_count": receipt.message_count,
        "compressed_size": receipt.compressed_size,
        "sha256": receipt.sha256,
        "archive_filename": archive_name,
        "manifest_filename": manifest_name,
        "sample_session_id": final_verification.get("sample_session_id"),
        "sample_message_count": final_verification.get("sample_message_count"),
        "remote_available_bytes": final_verification.get("available_bytes"),
    }
    _write_audit(
        audit_path,
        {
            "schema_version": 1,
            "event": "cron_archive_prune",
            "source_profile": source_profile,
            "category": "cron",
            "batch_fingerprint": fingerprint,
            "candidate_count": len(session_ids),
            "archived_count": receipt.session_count,
            "verified_count": int(final_verification["session_count"]),
            "pruned_count": pruned,
            "message_count": receipt.message_count,
            "estimated_logical_bytes": plan.categories["cron"].estimated_logical_bytes,
            "compressed_size": receipt.compressed_size,
            "sha256": receipt.sha256,
            "outcome": "verified_pruned",
        },
    )
    return result


def execute_disposable_cron_prune(
    db,
    *,
    session_ids: Sequence[str],
    sessions_dir: Path,
    audit_path: Path,
    min_free_bytes: int,
) -> Dict[str, Any]:
    """Directly prune an externally settled exact cron set, without archive."""
    check_disk_guard(db.db_path.parent, min_free_bytes=min_free_bytes)
    exact_ids = tuple(dict.fromkeys(str(sid) for sid in session_ids))
    if not exact_ids:
        raise ValueError("an exact disposable session set is required")
    rows = db.list_prune_candidates(
        source="cron",
        end_reason="cron_complete",
        archived=False,
        retention_safe=True,
        session_ids=exact_ids,
    )
    if [str(row["id"]) for row in rows] != list(exact_ids):
        raise ValueError("exact retention set changed; pruned nothing")
    estimated_bytes, message_count = db.estimate_sessions_logical_bytes(exact_ids)
    pruned = db.prune_sessions(
        older_than_days=None,
        source="cron",
        end_reason="cron_complete",
        archived=False,
        retention_safe=True,
        session_ids=exact_ids,
        require_all=True,
        sessions_dir=sessions_dir,
    )
    fingerprint = hashlib.sha256("\n".join(exact_ids).encode("utf-8")).hexdigest()[:16]
    _write_audit(
        audit_path,
        {
            "schema_version": 1,
            "event": "disposable_cron_prune",
            "category": "cron",
            "batch_fingerprint": fingerprint,
            "candidate_count": len(exact_ids),
            "archived_count": 0,
            "verified_count": len(exact_ids),
            "pruned_count": pruned,
            "message_count": message_count,
            "estimated_logical_bytes": estimated_bytes,
            "outcome": "exact_disposable_pruned",
        },
    )
    return {
        "candidate_count": len(exact_ids),
        "archived_count": 0,
        "verified_count": len(exact_ids),
        "pruned_count": pruned,
        "message_count": message_count,
        "estimated_logical_bytes": estimated_bytes,
        "batch_fingerprint": fingerprint,
    }


def run_retention_cli(args) -> None:
    """Run the inert-by-default native sessions retention command."""
    import time

    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    archive_execute = bool(args.execute_cron_archive_prune)
    disposable_execute = bool(args.execute_disposable_cron_prune)
    if archive_execute and disposable_execute:
        print("Choose one execution mode; nothing archived or pruned.")
        return
    if archive_execute and args.authorization != "ARCHIVE_AND_PRUNE_EXACT_IDS":
        print(
            "Execution requires --authorization ARCHIVE_AND_PRUNE_EXACT_IDS; "
            "nothing archived or pruned."
        )
        return
    if disposable_execute and args.authorization != "PRUNE_EXACT_DISPOSABLE_IDS":
        print(
            "Disposable prune requires --authorization PRUNE_EXACT_DISPOSABLE_IDS; "
            "nothing pruned."
        )
        return
    if archive_execute and not args.archive_host:
        print("Execution requires --archive-host; nothing archived or pruned.")
        return
    if disposable_execute and not args.disposable_session_ids:
        print("Disposable prune requires --disposable-session-ids; nothing pruned.")
        return
    if args.batch_size <= 0:
        print("--batch-size must be positive; nothing archived or pruned.")
        return

    home = get_hermes_home()
    if not (archive_execute or disposable_execute):
        db = SessionDB(read_only=True)
        try:
            plan = build_retention_plan(
                db,
                now=time.time(),
                settled_session_ids=load_exact_session_ids(args.settled_session_ids),
                protected_session_ids=load_exact_session_ids(args.protected_session_ids),
            )
            report = plan.public_report()
            report["profile"] = args.profile_label
            print(json.dumps(report, sort_keys=True))
        finally:
            db.close()
        return

    with exclusive_retention_lock(home / ".session-retention.lock"):
        db = SessionDB()
        try:
            if archive_execute:
                now = time.time()
                plan = build_retention_plan(
                    db,
                    now=now,
                    settled_session_ids=load_exact_session_ids(args.settled_session_ids),
                    protected_session_ids=load_exact_session_ids(args.protected_session_ids),
                    batch_limits={"cron": args.batch_size, "agent": 0, "human": 0},
                )
                target = SSHArchiveTarget(args.archive_host, args.archive_root)
                result = execute_cron_archive_prune(
                    db,
                    plan=plan,
                    target=target,
                    source_profile=args.profile_label,
                    sessions_dir=home / "sessions",
                    audit_path=home / "session-retention-audit.jsonl",
                    now=now,
                    min_free_bytes=args.min_free_bytes,
                )
                if result.get("archive_filename"):
                    result["archive_root"] = target.display_root
            else:
                exact_ids = sorted(load_exact_session_ids(args.disposable_session_ids))[
                    : args.batch_size
                ]
                result = execute_disposable_cron_prune(
                    db,
                    session_ids=exact_ids,
                    sessions_dir=home / "sessions",
                    audit_path=home / "session-retention-audit.jsonl",
                    min_free_bytes=args.min_free_bytes,
                )
            result["profile"] = args.profile_label
            print(json.dumps(result, sort_keys=True))
        finally:
            db.close()
