"""
Feature 1: Self-Evicting In-Memory Cache for LLM Inputs, Settings, and Outputs.

How this cache works:
1. It stores a mapping of `(function_identity + input_args + input_kwargs + config_settings) -> output`
   in local memory alongside the input settings snapshot.
2. If the user calls the function with the exact same input and settings, a deep copy of the
   cached output is returned immediately without re-invoking the underlying LLM function.
3. Lifetime (TTL) eviction: Each entry has a configurable `ttl` (lifetime in seconds).
   Once that lifetime passes, the entry vanishes from memory and executes the function
   to generate and store a fresh response.
4. Request-count (`max_requests`) self-destruction: Each entry can be configured with
   `max_requests=X`. After serving `X` cached requests for that input, the entry kills
   (vanishes) itself from memory so the subsequent request generates a fresh response.
5. Bounded memory (`max_size`): Automatically purges expired entries and evicts the oldest
   entries when `max_size` (default 1,000) is reached, preventing memory leaks in long-running
   processes.
"""

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import hashlib
import inspect
import io
import json
import re
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Set, Tuple, TypeVar

T = TypeVar("T")

# Matches default Python memory address suffixes like " at 0x104a8b2c0"
_MEM_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")


@dataclass
class CacheConfig:
    """Configuration settings for the self-evicting in-memory cache."""

    enabled: bool = True
    ttl: Optional[float] = 300.0  # Lifetime in seconds (None = no time expiration)
    max_requests: Optional[int] = None  # Kill entry after X cached requests (None = unlimited)
    max_size: Optional[int] = 1000  # Maximum entries in memory (None = unlimited)


@dataclass
class CacheEntry:
    """Represents a single stored input+settings -> output entry in memory."""

    key: str
    input_settings: Dict[str, Any]
    output: Any
    created_at: float = field(default_factory=time.monotonic)
    ttl: Optional[float] = 300.0
    max_requests: Optional[int] = None
    hits: int = 0

    def is_expired_by_ttl(self, now: Optional[float] = None) -> bool:
        """Return True if the entry's lifetime (TTL) has elapsed."""
        if self.ttl is None:
            return False
        current = time.monotonic() if now is None else now
        return (current - self.created_at) >= self.ttl

    def is_expired_by_requests(self) -> bool:
        """Return True if the entry has reached or exceeded its max_requests limit."""
        if self.max_requests is None:
            return False
        return self.hits >= self.max_requests


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _safe_clone(value: Any, _memo: Optional[Dict[int, Any]] = None) -> Any:
    """
    Return a deep copy of `value` so callers mutating a result cannot corrupt cached state.
    If `value` (such as an SDK response object) holds an uncopyable attribute like a
    `threading.Lock` or socket, clones the object shell and all copyable attributes/containers
    recursively while preserving the uncopyable attribute reference.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes, frozenset, type)):
        return value

    if _memo is None:
        _memo = {}

    obj_id = id(value)
    if obj_id in _memo:
        return _memo[obj_id]

    try:
        return copy.deepcopy(value, _memo)
    except Exception:
        pass

    # Fallback recursive clone when `copy.deepcopy` fails due to an embedded lock/socket
    if isinstance(value, dict):
        cloned_dict: Dict[Any, Any] = {}
        _memo[obj_id] = cloned_dict
        for k, v in value.items():
            cloned_dict[_safe_clone(k, _memo)] = _safe_clone(v, _memo)
        return cloned_dict

    if isinstance(value, list):
        cloned_list: list[Any] = []
        _memo[obj_id] = cloned_list
        cloned_list.extend(_safe_clone(v, _memo) for v in value)
        return cloned_list

    if isinstance(value, tuple):
        cloned_tuple = tuple(_safe_clone(v, _memo) for v in value)
        _memo[obj_id] = cloned_tuple
        return cloned_tuple

    if isinstance(value, set):
        cloned_set = {_safe_clone(v, _memo) for v in value}
        _memo[obj_id] = cloned_set
        return cloned_set

    # Custom object or SDK response instance holding an uncopyable field (e.g. a Lock)
    try:
        cloned_obj = copy.copy(value)
        _memo[obj_id] = cloned_obj
        if hasattr(value, "__dict__"):
            for attr_k, attr_v in list(vars(value).items()):
                try:
                    setattr(cloned_obj, attr_k, _safe_clone(attr_v, _memo))
                except Exception:
                    pass
        cls = type(value)
        for base in cls.__mro__:
            slots = getattr(base, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot_name in slots:
                if slot_name in ("__dict__", "__weakref__"):
                    continue
                if hasattr(value, slot_name):
                    try:
                        setattr(
                            cloned_obj,
                            slot_name,
                            _safe_clone(getattr(value, slot_name), _memo),
                        )
                    except Exception:
                        pass
        return cloned_obj
    except Exception:
        return value


def is_stream_or_generator(value: Any) -> bool:
    """Return True if `value` is a single-use generator, iterator, or async iterator/stream."""
    if inspect.isgenerator(value) or inspect.isasyncgen(value):
        return True
    if isinstance(value, (collections.abc.Iterator, collections.abc.AsyncIterator)):
        return True
    return False


def is_llm_stream(value: Any) -> bool:
    """
    Return True if `value` is a generator or an SDK stream iterator (not a file handle
    or built-in collection iterator like `map`, `zip`, `filter`, `enumerate`).
    """
    if isinstance(value, io.IOBase):
        return False
    if isinstance(value, (map, filter, zip, enumerate)):
        return False
    if type(value).__name__ in (
        "list_iterator",
        "tuple_iterator",
        "dict_keyiterator",
        "dict_valueiterator",
        "dict_itemiterator",
        "set_iterator",
        "str_iterator",
        "bytes_iterator",
        "range_iterator",
        "callable_iterator",
    ):
        return False
    return is_stream_or_generator(value)


def _extract_object_state(value: Any) -> Dict[str, Any]:
    """Extract all instance attributes from `__dict__` and `__slots__`."""
    state: Dict[str, Any] = {}
    if hasattr(value, "__dict__"):
        state.update(vars(value))
    for base in type(value).__mro__:
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in ("__dict__", "__weakref__"):
                continue
            if hasattr(value, slot):
                try:
                    state[slot] = getattr(value, slot)
                except Exception:
                    pass
    return state


def _normalize_for_key(value: Any, _seen: Optional[Set[int]] = None) -> Any:
    """
    Recursively convert arbitrary Python objects into a type-tagged, cycle-safe,
    deterministic JSON-serializable structure.

    - Distinguishes `{1: "a"}` from `{"1": "a"}` and `(1, 2)` from `[1, 2]` and `True` from `1`.
    - Handles mixed-type sets (`{1, "a"}`) and sets of objects without raising `TypeError`.
    - Detects cyclic references via `_seen` to prevent `RecursionError`.
    - Walks attributes (`__dict__` and `__slots__`) of custom input classes (e.g. `Prompt("...")`)
      so distinct object instances never collide.
    - Hashes binary buffer / array objects (like `numpy.ndarray`) by their full bytes/shape/dtype
      so truncated `...` reprs never collide.
    """
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", repr(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ["bytes", bytes(value).hex()]

    if _seen is None:
        _seen = set()

    obj_id = id(value)
    if obj_id in _seen:
        return ["cycle", f"{type(value).__module__}.{type(value).__qualname__}"]

    _seen.add(obj_id)
    try:
        if isinstance(value, tuple):
            return ["tuple", [_normalize_for_key(v, _seen) for v in value]]
        if isinstance(value, list):
            return ["list", [_normalize_for_key(v, _seen) for v in value]]
        if isinstance(value, (set, frozenset)):
            tag = "frozenset" if isinstance(value, frozenset) else "set"
            normalized_items = [_normalize_for_key(v, _seen) for v in value]
            normalized_items.sort(key=_canonical_json)
            return [tag, normalized_items]
        if isinstance(value, dict):
            pairs = [
                [_normalize_for_key(k, _seen), _normalize_for_key(v, _seen)]
                for k, v in value.items()
            ]
            pairs.sort(key=lambda pair: _canonical_json(pair[0]))
            return ["dict", pairs]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields_data = [
                [f.name, _normalize_for_key(getattr(value, f.name), _seen)]
                for f in dataclasses.fields(value)
            ]
            return [
                "dataclass",
                f"{type(value).__module__}.{type(value).__qualname__}",
                fields_data,
            ]
        if isinstance(value, SimpleNamespace):
            ns_pairs = [
                [str(k), _normalize_for_key(v, _seen)]
                for k, v in sorted(vars(value).items(), key=lambda item: str(item[0]))
            ]
            return ["namespace", ns_pairs]

        cls = type(value)
        type_name = f"{cls.__module__}.{cls.__qualname__}"

        # Array / tensor objects (e.g. numpy.ndarray) whose repr truncates with "..."
        if hasattr(value, "tobytes") and hasattr(value, "shape") and hasattr(value, "dtype"):
            try:
                raw_bytes = value.tobytes()
                digest = hashlib.sha256(raw_bytes).hexdigest()
                return ["ndarray", type_name, str(value.dtype), list(value.shape), digest]
            except Exception:
                pass

        # Extract attributes from `__dict__` and `__slots__` (covers custom classes like `Prompt("...")`)
        state = _extract_object_state(value)
        if state:
            state_pairs = [
                [str(k), _normalize_for_key(v, _seen)]
                for k, v in sorted(state.items(), key=lambda item: str(item[0]))
            ]
            return ["object_state", type_name, state_pairs]

        # Built-in / C types without `__dict__` (e.g., datetime, Decimal, UUID, Path, Regex)
        try:
            raw_repr = repr(value)
            cleaned_repr = _MEM_ADDR_RE.sub("", raw_repr)
            return ["repr", type_name, cleaned_repr]
        except Exception:
            return ["instance", type_name]
    finally:
        _seen.discard(obj_id)


def build_cache_key(
    func_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
    func: Optional[Callable[..., Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a deterministic SHA-256 cache key and structured snapshot from the
    function identifier, positional/keyword inputs (including `self` on methods),
    and configuration settings.
    """
    snapshot = {
        "function": func_name,
        "args": _normalize_for_key(args),
        "kwargs": _normalize_for_key(kwargs),
        "settings": _normalize_for_key(settings or {}),
    }
    serialized = _canonical_json(snapshot)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest, snapshot


class InMemoryCache:
    """
    Thread-safe local in-memory cache that stores `(input + settings) -> output`.

    Eviction rules:
    - **TTL (Lifetime)**: If `ttl` seconds have elapsed since the entry was created,
      it vanishes from memory and signals a cache miss so a new response is generated.
    - **Max Requests (`max_requests`)**: After the entry is fetched `max_requests` times,
      it kills itself from memory so the next fetch generates a fresh response.
    - **Max Size (`max_size`)**: Purges expired entries and evicts oldest entries when
      the cache reaches `max_size` (default 1,000) so memory cannot grow unbounded.
    """

    def __init__(
        self,
        default_ttl: Optional[float] = 300.0,
        default_max_requests: Optional[int] = None,
        max_size: Optional[int] = 1000,
    ) -> None:
        self.default_ttl = default_ttl
        self.default_max_requests = default_max_requests
        self.max_size = max_size
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._total_evictions: int = 0

    def _purge_expired_locked(self, now: Optional[float] = None) -> int:
        """Remove all TTL-expired or request-exhausted entries while holding `self._lock`."""
        current = time.monotonic() if now is None else now
        expired_keys = [
            k
            for k, entry in self._store.items()
            if entry.is_expired_by_ttl(current) or entry.is_expired_by_requests()
        ]
        for k in expired_keys:
            del self._store[k]
            self._total_evictions += 1
        return len(expired_keys)

    def fetch(self, key: str) -> Tuple[bool, Any, str]:
        """
        Attempt to fetch a cached output by `key`.

        Returns:
            Tuple of `(hit: bool, output: Any, reason: str)` where `reason` is one of:
            - `"hit"`: Valid cache hit; deep-copied output returned.
            - `"hit_and_vanished"`: Valid cache hit on the X-th (`max_requests`) fetch;
              output is returned and the entry immediately kills itself from memory.
            - `"expired_ttl"`: Entry lifetime expired on fetch; entry vanished itself.
            - `"expired_max_requests"`: Entry request limit reached; entry vanished itself.
            - `"miss"`: No entry found for `key`.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._total_misses += 1
                return False, None, "miss"

            now = time.monotonic()
            if entry.is_expired_by_ttl(now):
                # Lifetime elapsed: vanish itself upon fetch
                del self._store[key]
                self._total_evictions += 1
                self._total_misses += 1
                return False, None, "expired_ttl"

            if entry.is_expired_by_requests():
                # Request limit already reached: vanish itself upon fetch
                del self._store[key]
                self._total_evictions += 1
                self._total_misses += 1
                return False, None, "expired_max_requests"

            # Valid cache hit
            entry.hits += 1
            self._total_hits += 1
            output_copy = _safe_clone(entry.output)

            # Refresh LRU position on hit
            self._store.pop(key, None)

            # If this fetch reached `max_requests`, kill itself immediately after serving
            if entry.max_requests is not None and entry.hits >= entry.max_requests:
                self._total_evictions += 1
                return True, output_copy, "hit_and_vanished"

            self._store[key] = entry
            return True, output_copy, "hit"

    def store(
        self,
        key: str,
        input_settings: Dict[str, Any],
        output: Any,
        ttl: Optional[float] = None,
        max_requests: Optional[int] = None,
        use_defaults: bool = True,
    ) -> Optional[CacheEntry]:
        """Store an input+settings snapshot and its corresponding output in memory."""
        # Never cache single-use generators/iterators/streams
        if is_stream_or_generator(output):
            return None

        effective_ttl = self.default_ttl if (ttl is None and use_defaults) else ttl
        effective_max_req = (
            self.default_max_requests if (max_requests is None and use_defaults) else max_requests
        )

        # If ttl <= 0 or max_requests <= 0, do not retain the entry
        if (effective_ttl is not None and effective_ttl <= 0) or (
            effective_max_req is not None and effective_max_req <= 0
        ):
            with self._lock:
                self._store.pop(key, None)
            return None

        with self._lock:
            self._purge_expired_locked()

            if self.max_size is not None and self.max_size <= 0:
                self._store.clear()
                return None

            # Re-inserting an existing key removes the old position first
            if key in self._store:
                del self._store[key]

            # Evict oldest entries if max_size is reached
            if self.max_size is not None:
                while len(self._store) >= self.max_size:
                    oldest_key = next(iter(self._store))
                    del self._store[oldest_key]
                    self._total_evictions += 1

            entry = CacheEntry(
                key=key,
                input_settings=input_settings,
                output=_safe_clone(output),
                created_at=time.monotonic(),
                ttl=effective_ttl,
                max_requests=effective_max_req,
                hits=0,
            )
            self._store[key] = entry
            return entry

    def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Inspect a raw cache entry without incrementing its hit counter."""
        with self._lock:
            entry = self._store.get(key)
            if entry is not None and (entry.is_expired_by_ttl() or entry.is_expired_by_requests()):
                del self._store[key]
                self._total_evictions += 1
                return None
            return entry

    def invalidate(self, key: str) -> bool:
        """Manually remove a specific key from the cache."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._total_evictions += 1
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the in-memory cache."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        """Return current number of active (non-expired) entries in memory."""
        with self._lock:
            self._purge_expired_locked()
            return len(self._store)

    def stats(self) -> Dict[str, int]:
        """Return cache telemetry statistics."""
        with self._lock:
            self._purge_expired_locked()
            return {
                "active_entries": len(self._store),
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "total_evictions": self._total_evictions,
            }


def cached_llm_call(
    func: Callable[..., T],
    *args: Any,
    cache: Optional[InMemoryCache] = None,
    ttl: Optional[float] = 300.0,
    max_requests: Optional[int] = None,
    settings: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> T:
    """
    Execute `func(*args, **kwargs)` with local in-memory caching of Input Settings + Output.

    IMPORTANT — How this function works:
    - Saves the input arguments (`args`, `kwargs`) and configuration `settings` alongside the
      generated `output` in an in-memory cache.
    - When called again with the exact same input and settings, returns a copy of the stored
      output without calling `func`.
    - **Lifetime (`ttl`)**: After `ttl` seconds have elapsed, the cached entry vanishes itself
      from memory and `func` is called to generate and cache a fresh response.
    - **Self-Destruct after X Requests (`max_requests`)**: If `max_requests` is set to `X`,
      the cached entry will vanish itself after being fetched `X` times, forcing the next
      call to generate a fresh response.
    """
    active_cache = cache or _DEFAULT_GLOBAL_CACHE
    func_name = f"{getattr(func, '__module__', '__main__')}.{getattr(func, '__qualname__', getattr(func, '__name__', 'callable'))}"
    key, snapshot = build_cache_key(func_name, args, kwargs, settings=settings, func=func)

    if inspect.iscoroutinefunction(func):

        async def _async_runner() -> Any:
            hit, cached_output, _ = active_cache.fetch(key)
            if hit:
                return cached_output
            result = await func(*args, **kwargs)
            active_cache.store(
                key=key,
                input_settings=snapshot,
                output=result,
                ttl=ttl,
                max_requests=max_requests,
                use_defaults=False,
            )
            return result

        return _async_runner()  # type: ignore[return-value]

    hit, cached_output, _ = active_cache.fetch(key)
    if hit:
        return cached_output

    result = func(*args, **kwargs)
    active_cache.store(
        key=key,
        input_settings=snapshot,
        output=result,
        ttl=ttl,
        max_requests=max_requests,
        use_defaults=False,
    )
    return result


_DEFAULT_GLOBAL_CACHE = InMemoryCache()
