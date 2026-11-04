"""
Feature 3 (Part 2): Call Telemetry Logger, Token Extractor, and Cost/Performance Analyzer.

Persists call logs to a JSONL file and aggregates metrics to answer:
1. How much money did I spend this week?
2. Which feature (and config) costs the most?
3. Which calls fail the most?
4. Which calls are slow?
"""

from __future__ import annotations

import collections.abc
import copy
import inspect
import json
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

_LOCK_EX = fcntl.LOCK_EX if fcntl is not None else 0
_LOCK_SH = fcntl.LOCK_SH if fcntl is not None else 0
_LOCK_UN = fcntl.LOCK_UN if fcntl is not None else 0


@dataclass
class TokenUsage:
    """Extracted or estimated token counts for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    estimated: bool = False
    detected_model: Optional[str] = None
    input_excludes_cached: Optional[bool] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CallRecord:
    """A single logged LLM call event persisted as one line in the JSONL log file."""

    id: str
    timestamp: str  # ISO-8601 UTC timestamp
    feature: str
    function_name: str
    model: str
    config: Dict[str, Any]
    status: str  # "success", "cached", "failed", "incomplete"
    cache_hit: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float
    retries: int
    error_message: Optional[str] = None
    attempt_errors: List[Dict[str, Any]] = field(default_factory=list)
    input_preview: str = ""
    output_preview: str = ""
    cache_creation_input_tokens: int = 0
    # True when token counts (and therefore cost) were guessed from text length
    estimated_tokens: bool = False
    # Free-form per-call tags such as `user_id`, `tenant`, or `prompt_version`
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CallRecord":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            timestamp=str(data.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            feature=str(data.get("feature") or "default"),
            function_name=str(data.get("function_name") or "unknown"),
            model=str(data.get("model") or "gpt-4o-mini"),
            config=dict(data.get("config") or {}),
            status=str(data.get("status") or "success"),
            cache_hit=bool(data.get("cache_hit", False)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cached_input_tokens=int(data.get("cached_input_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            retries=int(data.get("retries", 0)),
            error_message=data.get("error_message"),
            attempt_errors=list(data.get("attempt_errors") or []),
            input_preview=str(data.get("input_preview") or ""),
            output_preview=str(data.get("output_preview") or ""),
            cache_creation_input_tokens=int(data.get("cache_creation_input_tokens", 0)),
            estimated_tokens=bool(data.get("estimated_tokens", False)),
            metadata=dict(data.get("metadata") or {}),
        )


def _estimate_tokens_from_text(value: Any) -> int:
    """Rough heuristic token estimator (~4 characters per token) when SDK usage is absent."""
    if value is None:
        return 0
    if inspect.isgenerator(value) or inspect.isasyncgen(value) or isinstance(
        value, (collections.abc.Iterator, collections.abc.AsyncIterator)
    ):
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str)
        except Exception:
            text = str(value)
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_from_usage_obj(
    usage: Any,
    detected_model: Optional[str] = None,
) -> Optional[TokenUsage]:
    """Extract `TokenUsage` from a `usage` object or dict if token fields are present."""
    if usage is None:
        return None

    in_tok = (
        _get_attr_or_key(usage, "input_tokens")
        if _get_attr_or_key(usage, "input_tokens") is not None
        else _get_attr_or_key(usage, "prompt_tokens")
    )
    out_tok = (
        _get_attr_or_key(usage, "output_tokens")
        if _get_attr_or_key(usage, "output_tokens") is not None
        else _get_attr_or_key(usage, "completion_tokens")
    )

    has_anthropic_cache_fields = (
        _get_attr_or_key(usage, "cache_read_input_tokens") is not None
        or _get_attr_or_key(usage, "cache_creation_input_tokens") is not None
    )

    cached_tok = _get_attr_or_key(usage, "cache_read_input_tokens")
    if cached_tok is None:
        cached_tok = _get_attr_or_key(usage, "cached_tokens")
    if cached_tok is None:
        details = (
            _get_attr_or_key(usage, "prompt_tokens_details")
            or _get_attr_or_key(usage, "input_tokens_details")
            or _get_attr_or_key(usage, "input_token_details")
        )
        if details is not None:
            cached_tok = _get_attr_or_key(details, "cached_tokens", 0)

    cache_creation_tok = _get_attr_or_key(usage, "cache_creation_input_tokens", 0)

    if in_tok is not None or out_tok is not None or has_anthropic_cache_fields:
        return TokenUsage(
            input_tokens=int(in_tok or 0),
            output_tokens=int(out_tok or 0),
            cached_input_tokens=int(cached_tok or 0),
            cache_creation_input_tokens=int(cache_creation_tok or 0),
            estimated=False,
            detected_model=detected_model,
            input_excludes_cached=True if has_anthropic_cache_fields else None,
        )
    return None


def extract_token_usage(
    response: Any,
    args: Sequence[Any] = (),
    kwargs: Optional[Dict[str, Any]] = None,
) -> TokenUsage:
    """
    Extract input/output token counts from OpenAI, Anthropic (including `message_start`
    stream events), Google Gemini, or dict responses.
    Falls back to a lightweight character-based estimate if no structured usage metadata is present.
    """
    detected_model: Optional[str] = None
    model_val = _get_attr_or_key(response, "model") or _get_attr_or_key(response, "model_version")
    if isinstance(model_val, str) and model_val.strip():
        detected_model = model_val.strip()

    # Check nested `response.message` (Anthropic `message_start` stream event carries `chunk.message.usage` and `chunk.message.model`)
    nested_msg = _get_attr_or_key(response, "message")
    if nested_msg is not None:
        if detected_model is None:
            nested_model = _get_attr_or_key(nested_msg, "model")
            if isinstance(nested_model, str) and nested_model.strip():
                detected_model = nested_model.strip()

    # 1. Check `usage` attribute/key on `response` AND on `response.message`
    top_usage = _extract_from_usage_obj(_get_attr_or_key(response, "usage"), detected_model)
    nested_usage = (
        _extract_from_usage_obj(_get_attr_or_key(nested_msg, "usage"), detected_model)
        if nested_msg is not None
        else None
    )
    if top_usage is not None and nested_usage is not None:
        return TokenUsage(
            input_tokens=top_usage.input_tokens or nested_usage.input_tokens,
            output_tokens=top_usage.output_tokens or nested_usage.output_tokens,
            cached_input_tokens=top_usage.cached_input_tokens or nested_usage.cached_input_tokens,
            cache_creation_input_tokens=(
                top_usage.cache_creation_input_tokens or nested_usage.cache_creation_input_tokens
            ),
            estimated=False,
            detected_model=detected_model,
            input_excludes_cached=(
                top_usage.input_excludes_cached
                if top_usage.input_excludes_cached is not None
                else nested_usage.input_excludes_cached
            ),
        )
    if top_usage is not None:
        return top_usage
    if nested_usage is not None:
        return nested_usage

    # 2. Check Google Gemini `usage_metadata` attribute or dict key
    usage_meta = _get_attr_or_key(response, "usage_metadata")
    if usage_meta is not None:
        in_tok = _get_attr_or_key(usage_meta, "prompt_token_count")
        if in_tok is None:
            in_tok = _get_attr_or_key(usage_meta, "input_tokens")
        out_tok = _get_attr_or_key(usage_meta, "candidates_token_count")
        if out_tok is None:
            out_tok = _get_attr_or_key(usage_meta, "output_tokens")
        cached_tok = _get_attr_or_key(usage_meta, "cached_content_token_count", 0)
        if in_tok is not None or out_tok is not None:
            return TokenUsage(
                input_tokens=int(in_tok or 0),
                output_tokens=int(out_tok or 0),
                cached_input_tokens=int(cached_tok or 0),
                cache_creation_input_tokens=0,
                estimated=False,
                detected_model=detected_model,
                input_excludes_cached=False,
            )

    # 3. Check top-level dict keys (`input_tokens` / `output_tokens` or `prompt_tokens` / `completion_tokens`)
    if isinstance(response, dict):
        in_tok = response.get("input_tokens", response.get("prompt_tokens"))
        out_tok = response.get("output_tokens", response.get("completion_tokens"))
        has_anth_fields = (
            "cache_read_input_tokens" in response or "cache_creation_input_tokens" in response
        )
        if in_tok is not None or out_tok is not None or has_anth_fields:
            return TokenUsage(
                input_tokens=int(in_tok or 0),
                output_tokens=int(out_tok or 0),
                cached_input_tokens=int(
                    response.get("cached_input_tokens", response.get("cache_read_input_tokens", 0))
                ),
                cache_creation_input_tokens=int(response.get("cache_creation_input_tokens", 0)),
                estimated=False,
                detected_model=detected_model,
                input_excludes_cached=True if has_anth_fields else None,
            )

    # 4. Heuristic estimation from input arguments & output response
    est_in = _estimate_tokens_from_text({"args": args, "kwargs": kwargs or {}})
    est_out = _estimate_tokens_from_text(response)
    return TokenUsage(
        input_tokens=est_in,
        output_tokens=est_out,
        cached_input_tokens=0,
        cache_creation_input_tokens=0,
        estimated=True,
        detected_model=detected_model,
        input_excludes_cached=None,
    )


def format_preview(
    value: Any,
    max_len: int = 140,
    redact: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Create a short human-readable preview string of an input or output for search & inspection.
    `redact` (if given) is applied to the full text before truncation.
    """
    if value is None:
        return ""
    if inspect.isgenerator(value) or inspect.isasyncgen(value) or isinstance(
        value, (collections.abc.Iterator, collections.abc.AsyncIterator)
    ):
        return f"<stream:{type(value).__name__}>"
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            text = repr(value)
    if redact is not None:
        text = str(redact(text))
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _safe_json_dumps(obj: Any, sort_keys: bool = False) -> str:
    """Serialize `obj` to JSON safely even if it contains non-serializable objects or mixed key types."""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, default=str)
    except TypeError:
        return json.dumps(str(obj), ensure_ascii=False)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        written += os.write(fd, view[written:])


def _clone_record(rec: CallRecord) -> CallRecord:
    """Copy a record (and its mutable fields) so callers can't corrupt the read cache."""
    clone = copy.copy(rec)
    clone.config = dict(rec.config)
    clone.metadata = dict(rec.metadata)
    clone.attempt_errors = list(rec.attempt_errors)
    return clone


class JSONLLogger:
    """
    Thread-safe and multiprocess-safe, append-only JSONL reader and writer for `CallRecord` events.

    - Every write is a single appended line under an exclusive `fcntl.flock`, so concurrent
      threads and worker processes never lose records. Writers re-check the file's inode after
      locking so they never append to a file another process just rotated away.
    - `update()` appends the new version of a record; readers keep the **last** line for each
      `id` (stream calls therefore occupy two lines: a placeholder and the final record).
      `compact()` rewrites the file with one line per record.
    - `read_all()` is incremental: it remembers what it already parsed and only reads bytes
      appended since the previous call.
    - Optional rotation: when the file would exceed `max_bytes`, it is renamed to
      `<name>.1` (older backups shift to `.2`, ... up to `backups`) and a new file is started.
    """

    def __init__(
        self,
        log_path: str | Path = ".llm_tally/logs.jsonl",
        max_bytes: Optional[int] = None,
        backups: int = 3,
    ) -> None:
        self.log_path = Path(log_path)
        self.max_bytes = int(max_bytes) if max_bytes else None
        self.backups = max(0, int(backups))
        self._lock = threading.RLock()
        self._reset_cache()

    # ---- locking helpers ---------------------------------------------------------
    @staticmethod
    def _flock(fd: int, mode: int) -> None:
        if fcntl is not None:
            try:
                fcntl.flock(fd, mode)
            except OSError:
                pass

    def _open_locked_for_append(self) -> int:
        """Open the current log file for appending and hold an exclusive lock on it."""
        while True:
            fd = os.open(str(self.log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            self._flock(fd, _LOCK_EX)
            try:
                on_disk = os.stat(str(self.log_path))
                opened = os.fstat(fd)
                if (on_disk.st_dev, on_disk.st_ino) == (opened.st_dev, opened.st_ino):
                    return fd
            except FileNotFoundError:
                pass
            # Rotated or replaced between open() and flock(): retry on the new file
            self._flock(fd, _LOCK_UN)
            os.close(fd)

    def _rotate_locked(self) -> None:
        """Shift `log.N` -> `log.N+1` and move the current log to `log.1` (caller holds the lock)."""
        base = str(self.log_path)
        if self.backups == 0:
            os.unlink(base)
            return
        for idx in range(self.backups - 1, 0, -1):
            src = f"{base}.{idx}"
            if os.path.exists(src):
                os.replace(src, f"{base}.{idx + 1}")
        os.replace(base, f"{base}.1")

    # ---- writing -------------------------------------------------------------------
    def _append_line(self, line: str) -> None:
        payload = (line + "\n").encode("utf-8")
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            fd = self._open_locked_for_append()
            try:
                if self.max_bytes is not None:
                    size = os.fstat(fd).st_size
                    if size > 0 and size + len(payload) > self.max_bytes:
                        self._rotate_locked()
                        self._flock(fd, _LOCK_UN)
                        os.close(fd)
                        fd = self._open_locked_for_append()
                _write_all(fd, payload)
            finally:
                self._flock(fd, _LOCK_UN)
                os.close(fd)

    def append(self, record: CallRecord) -> None:
        """Append a single `CallRecord` line to the JSONL log file."""
        self._append_line(_safe_json_dumps(record.to_dict(), sort_keys=False))

    def update(self, record: CallRecord) -> None:
        """
        Record a new version of an existing `CallRecord` (matched by `record.id`).
        Appends a line instead of rewriting the file, so the cost is O(1) regardless of
        log size; readers keep the last line per `id`.
        """
        self.append(record)

    def compact(self) -> int:
        """
        Rewrite the log with exactly one line (the latest version) per record id, atomically.
        Returns the number of records kept.
        """
        with self._lock:
            if not self.log_path.exists():
                return 0
            fd = self._open_locked_for_append()
            try:
                records = self._parse_bytes(_read_all_bytes_from_path(self.log_path))
                tmp_path = self.log_path.with_name(f".{self.log_path.name}.compact.tmp")
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    for rec in records.values():
                        fh.write(_safe_json_dumps(rec.to_dict(), sort_keys=False) + "\n")
                os.replace(tmp_path, self.log_path)
            finally:
                self._flock(fd, _LOCK_UN)
                os.close(fd)
            self._reset_cache()
            return len(records)

    # ---- reading -------------------------------------------------------------------
    def _reset_cache(self) -> None:
        self._cache_file_key: Optional[Tuple[int, int]] = None
        self._cache_offset = 0
        self._cache_records: Dict[str, CallRecord] = {}

    @staticmethod
    def _parse_bytes(raw: bytes, into: Optional[Dict[str, CallRecord]] = None) -> Dict[str, CallRecord]:
        records = {} if into is None else into
        for raw_line in raw.decode("utf-8", errors="replace").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                rec = CallRecord.from_dict(json.loads(stripped))
            except Exception:
                continue
            records[rec.id] = rec
        return records

    def read_all(self, copy_records: bool = True) -> List[CallRecord]:
        """
        Return all stored `CallRecord` entries (latest version per id, in first-seen order).
        Only bytes appended since the previous call are read and parsed. Pass
        `copy_records=False` for read-only internal use to skip copying.
        """
        with self._lock:
            try:
                fd = os.open(str(self.log_path), os.O_RDONLY)
            except FileNotFoundError:
                self._reset_cache()
                return []
            try:
                self._flock(fd, _LOCK_SH)
                st = os.fstat(fd)
                file_key = (st.st_dev, st.st_ino)
                if file_key != self._cache_file_key or st.st_size < self._cache_offset:
                    self._reset_cache()
                    self._cache_file_key = file_key
                new_bytes = b""
                if st.st_size > self._cache_offset:
                    os.lseek(fd, self._cache_offset, os.SEEK_SET)
                    new_bytes = _read_to_end(fd)
            finally:
                self._flock(fd, _LOCK_UN)
                os.close(fd)

            # Only consume complete lines; keep a trailing partial line for the next read
            # unless it is valid JSON on its own (e.g. a file written without a final newline).
            last_nl = new_bytes.rfind(b"\n")
            complete = new_bytes[: last_nl + 1] if last_nl >= 0 else b""
            tail = new_bytes[last_nl + 1 :]
            if tail.strip():
                try:
                    json.loads(tail.decode("utf-8"))
                    complete = new_bytes
                except Exception:
                    pass
            self._cache_offset += len(complete)
            self._parse_bytes(complete, into=self._cache_records)

            records = list(self._cache_records.values())
            return [_clone_record(r) for r in records] if copy_records else records

    def clear(self) -> None:
        """Remove all records from the JSONL log file."""
        with self._lock:
            if self.log_path.exists():
                self.log_path.unlink()
            self._reset_cache()


def _read_to_end(fd: int) -> bytes:
    chunks: List[bytes] = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _read_all_bytes_from_path(path: Path) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _parse_iso_utc(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def filter_records_since(
    records: Sequence[CallRecord],
    days: Optional[float],
    now: Optional[datetime] = None,
) -> List[CallRecord]:
    """Return only the records from the last `days` days (all records if `days` is None)."""
    if days is None:
        return list(records)
    cutoff = (now or datetime.now(timezone.utc)).astimezone(timezone.utc) - timedelta(
        days=float(days)
    )
    return [rec for rec in records if _parse_iso_utc(rec.timestamp) >= cutoff]


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = (len(sorted_vals) - 1) * (pct / 100.0)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return float(sorted_vals[low])
    weight = idx - low
    return float(sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight)


def analyze_logs(
    records: Sequence[CallRecord],
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive cost, failure, and latency analytics across `records`.

    Answers:
    1. How much money did I spend this week? (`spend_summary` & `weekly_daily_spend`)
    2. Which feature costs the most? (`feature_costs`)
    3. Which calls fail the most? (`failure_stats`)
    4. Which calls are slow? (`latency_stats` & `slowest_calls`)
    """
    reference_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    today_start = reference_now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today_start - timedelta(days=6)  # 7-day window including today
    month_ago = today_start - timedelta(days=29)

    spend_today = 0.0
    spend_this_week = 0.0
    spend_this_month = 0.0
    spend_all_time = 0.0
    saved_by_cache_usd = 0.0

    # Prepare 7-day daily buckets for "How much money did I spend this week?"
    daily_buckets: Dict[str, Dict[str, Any]] = {}
    for offset in range(6, -1, -1):
        day_dt = today_start - timedelta(days=offset)
        day_key = day_dt.strftime("%Y-%m-%d")
        daily_buckets[day_key] = {
            "date": day_key,
            "label": day_dt.strftime("%a %b %d"),
            "cost_usd": 0.0,
            "calls": 0,
            "cached_calls": 0,
            "failed_calls": 0,
        }

    # Group by (feature, config_json) for "Which feature costs the most?"
    feature_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    # Group by (feature, function_name) for failures & latency
    call_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    total_calls = len(records)
    total_cached = 0
    total_failed = 0
    total_retries = 0
    estimated_calls = 0
    estimated_cost_usd = 0.0
    # metadata key -> metadata value -> {"cost": float, "calls": int}
    metadata_groups: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for rec in records:
        dt = _parse_iso_utc(rec.timestamp)
        cost = float(rec.cost_usd)
        spend_all_time += cost
        if rec.estimated_tokens:
            estimated_calls += 1
            estimated_cost_usd += cost
        for meta_key, meta_val in (rec.metadata or {}).items():
            val_label = meta_val if isinstance(meta_val, str) else _safe_json_dumps(meta_val)
            bucket = metadata_groups.setdefault(str(meta_key), {}).setdefault(
                val_label, {"cost": 0.0, "calls": 0}
            )
            bucket["cost"] += cost
            bucket["calls"] += 1

        if dt >= today_start:
            spend_today += cost
        if dt >= week_ago:
            spend_this_week += cost
        if dt >= month_ago:
            spend_this_month += cost

        day_key = dt.strftime("%Y-%m-%d")
        if day_key in daily_buckets:
            daily_buckets[day_key]["cost_usd"] += cost
            daily_buckets[day_key]["calls"] += 1
            if rec.cache_hit:
                daily_buckets[day_key]["cached_calls"] += 1
            if rec.status == "failed":
                daily_buckets[day_key]["failed_calls"] += 1

        if rec.cache_hit:
            total_cached += 1
        if rec.status == "failed":
            total_failed += 1
        total_retries += int(rec.retries)

        config_str = _safe_json_dumps(rec.config, sort_keys=True) if rec.config else "{}"
        f_key = (rec.feature, config_str)
        if f_key not in feature_groups:
            feature_groups[f_key] = {
                "feature": rec.feature,
                "config": rec.config,
                "config_label": config_str if config_str != "{}" else "default",
                "models": set(),
                "total_cost_usd": 0.0,
                "week_cost_usd": 0.0,
                "total_calls": 0,
                "cached_calls": 0,
                "failed_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latencies_ms": [],
            }
        fg = feature_groups[f_key]
        fg["models"].add(rec.model)
        fg["total_cost_usd"] += cost
        if dt >= week_ago:
            fg["week_cost_usd"] += cost
        fg["total_calls"] += 1
        if rec.cache_hit:
            fg["cached_calls"] += 1
        if rec.status == "failed":
            fg["failed_calls"] += 1
        fg["input_tokens"] += rec.input_tokens
        fg["output_tokens"] += rec.output_tokens
        fg["latencies_ms"].append(rec.latency_ms)

        c_key = (rec.feature, rec.function_name)
        if c_key not in call_groups:
            call_groups[c_key] = {
                "feature": rec.feature,
                "function_name": rec.function_name,
                "models": set(),
                "total_calls": 0,
                "failed_calls": 0,
                "total_retries": 0,
                "latencies_ms": [],
                "non_cached_latencies_ms": [],
                "last_error": None,
                "attempt_errors": [],
            }
        cg = call_groups[c_key]
        cg["models"].add(rec.model)
        cg["total_calls"] += 1
        cg["total_retries"] += int(rec.retries)
        cg["latencies_ms"].append(rec.latency_ms)
        if not rec.cache_hit:
            cg["non_cached_latencies_ms"].append(rec.latency_ms)
        if rec.status == "failed":
            cg["failed_calls"] += 1
            if rec.error_message:
                cg["last_error"] = rec.error_message
        if rec.attempt_errors:
            cg["attempt_errors"].extend(rec.attempt_errors)

    # Finalize Feature Cost Breakdown ("Which feature costs the most?")
    feature_costs: List[Dict[str, Any]] = []
    for fg in feature_groups.values():
        non_cached_calls = max(0, fg["total_calls"] - fg["cached_calls"] - fg["failed_calls"])
        avg_cost_non_cached = (
            fg["total_cost_usd"] / non_cached_calls if non_cached_calls > 0 else 0.0
        )
        saved_by_cache_usd += fg["cached_calls"] * avg_cost_non_cached
        share_pct = (
            (fg["total_cost_usd"] / spend_all_time * 100.0) if spend_all_time > 0 else 0.0
        )
        lats = sorted(fg["latencies_ms"])
        feature_costs.append(
            {
                "feature": fg["feature"],
                "config": fg["config"],
                "config_label": fg["config_label"],
                "models": sorted(fg["models"]),
                "total_cost_usd": round(fg["total_cost_usd"], 8),
                "week_cost_usd": round(fg["week_cost_usd"], 8),
                "share_pct": round(share_pct, 2),
                "total_calls": fg["total_calls"],
                "cached_calls": fg["cached_calls"],
                "failed_calls": fg["failed_calls"],
                "avg_cost_usd": round(fg["total_cost_usd"] / max(1, fg["total_calls"]), 8),
                "input_tokens": fg["input_tokens"],
                "output_tokens": fg["output_tokens"],
                "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0.0,
            }
        )
    feature_costs.sort(key=lambda x: (x["total_cost_usd"], x["total_calls"]), reverse=True)

    # Finalize Failure Breakdown ("Which calls fail the most?")
    failure_stats: List[Dict[str, Any]] = []
    latency_stats: List[Dict[str, Any]] = []
    for cg in call_groups.values():
        fail_rate = (cg["failed_calls"] / cg["total_calls"] * 100.0) if cg["total_calls"] > 0 else 0.0
        failure_stats.append(
            {
                "feature": cg["feature"],
                "function_name": cg["function_name"],
                "models": sorted(cg["models"]),
                "total_calls": cg["total_calls"],
                "failed_calls": cg["failed_calls"],
                "failure_rate_pct": round(fail_rate, 2),
                "total_retries": cg["total_retries"],
                "last_error": cg["last_error"] or "",
            }
        )

        lats = sorted(cg["latencies_ms"])
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        p95_lat = _percentile(lats, 95.0)
        max_lat = lats[-1] if lats else 0.0
        latency_stats.append(
            {
                "feature": cg["feature"],
                "function_name": cg["function_name"],
                "models": sorted(cg["models"]),
                "total_calls": cg["total_calls"],
                "avg_latency_ms": round(avg_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "max_latency_ms": round(max_lat, 2),
            }
        )

    failure_stats.sort(
        key=lambda x: (x["failed_calls"], x["total_retries"], x["failure_rate_pct"]),
        reverse=True,
    )
    latency_stats.sort(
        key=lambda x: (x["avg_latency_ms"], x["max_latency_ms"]),
        reverse=True,
    )

    # Top individual slowest calls
    slowest_calls = [
        rec.to_dict()
        for rec in sorted(records, key=lambda r: r.latency_ms, reverse=True)[:15]
    ]

    top_cost_feature = feature_costs[0]["feature"] if feature_costs else "N/A"
    most_failing = (
        f"{failure_stats[0]['feature']} ({failure_stats[0]['failed_calls']} failed)"
        if failure_stats and failure_stats[0]["failed_calls"] > 0
        else "None (0 failures)"
    )
    slowest_feature = (
        f"{latency_stats[0]['feature']} ({latency_stats[0]['avg_latency_ms']:.1f} ms avg)"
        if latency_stats
        else "N/A"
    )

    # "Which user / tenant / prompt version costs the most?" (top 25 values per metadata key)
    metadata_costs: Dict[str, List[Dict[str, Any]]] = {}
    for meta_key, values in metadata_groups.items():
        rows = [
            {"value": val, "total_cost_usd": round(b["cost"], 8), "total_calls": b["calls"]}
            for val, b in values.items()
        ]
        rows.sort(key=lambda r: (r["total_cost_usd"], r["total_calls"]), reverse=True)
        metadata_costs[meta_key] = rows[:25]

    return {
        "generated_at": reference_now.isoformat(),
        "metadata_costs": metadata_costs,
        "spend_summary": {
            "today_usd": round(spend_today, 8),
            "this_week_usd": round(spend_this_week, 8),
            "this_month_usd": round(spend_this_month, 8),
            "all_time_usd": round(spend_all_time, 8),
            "saved_by_cache_usd": round(saved_by_cache_usd, 8),
        },
        "overview": {
            "total_calls": total_calls,
            "cached_calls": total_cached,
            "failed_calls": total_failed,
            "total_retries": total_retries,
            "cache_hit_rate_pct": round(
                (total_cached / total_calls * 100.0) if total_calls > 0 else 0.0, 2
            ),
            "top_cost_feature": top_cost_feature,
            "most_failing_call": most_failing,
            "slowest_call": slowest_feature,
            "estimated_calls": estimated_calls,
            "estimated_cost_usd": round(estimated_cost_usd, 8),
        },
        "weekly_daily_spend": list(daily_buckets.values()),
        "feature_costs": feature_costs,
        "failure_stats": failure_stats,
        "latency_stats": latency_stats,
        "slowest_calls": slowest_calls,
        "recent_calls": [
            rec.to_dict() for rec in sorted(records, key=lambda r: r.timestamp, reverse=True)[:250]
        ],
    }
