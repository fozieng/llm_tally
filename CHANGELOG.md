# Changelog

## 0.2.0

### Breaking changes
- The package was renamed from `django-llm-tally` / `django_llm_tally` to `llm-tally` / `llm_tally`, and the Django helpers (`get_django_llm_tally_settings`, `make_django_dashboard_view`) were removed.
- **Caching is now opt-in.** `default_cache_enabled` defaults to `False`. Pass `cache=True` per call, or `default_cache_enabled=True` to restore the old behaviour. `@tally.cache`, `cached_call`, and `cached_llm_call` still always cache.
- **Dashboard auto-updates are throttled** to at most once per `dashboard_update_interval` seconds (default 2.0). Pending updates are written in the background, at exit, or by `tally.flush()`. Set `dashboard_update_interval=0` for the old write-on-every-call behaviour.
- **HTTP 429 / rate-limit errors are now retried** (honoring `Retry-After`). Set `retry_on_rate_limit=False` to disable this.
- **The JSONL log is append-only.** A finished stream appends its final record instead of rewriting the file in place, so a stream call takes two lines and readers keep the last line per `id`. `JSONLLogger.compact()` / `llm-tally compact` rewrites the file with one line per call. If other tools read the file directly, de-duplicate by `id`.
- `log_previews=False` now also hides error message text (exception types and status codes are kept), and `redact` now also applies to error messages, retry-attempt messages, and `config` strings.

### Fixed
- Passing `None` to `LLMTally(...)` arguments now means "use the default". Before, it silently disabled caching, retries, or the dashboard, logged the model as `"None"`, or raised `TypeError`.
- A `model=` keyword passed to the wrapped function is now used to price the call when the response doesn't report the model. Before, it was priced as `default_model`.
- A stream abandoned before it finished was left logged as `success` with 0 output tokens. It is now logged as `incomplete`, with the tokens seen so far.
- Stream context managers whose `__enter__` / `__aenter__` return a different object (e.g. Anthropic `MessageStreamManager`) are now tracked and iterated correctly. Usage is read from the SDK's message snapshot when you iterate `stream.text_stream`.
- A stream manager that fails on `with` entry is logged as `failed` with the real error. It used to be logged as `incomplete` / "abandoned".
- A `redact` function that raises no longer loses the record. The text is stored as `"[redaction failed]"`.
- On large logs, a stream call took about 110 ms because the whole file was rewritten; it now takes about 1 ms (100k-line log). Reads are incremental, and the dashboard throttle stretches to 10× the rebuild time.

### Added
- `CallRecord.estimated_tokens` flags costs that were estimated from text length. The report's `overview` has `estimated_calls` / `estimated_cost_usd`, and the dashboard marks these costs with `~`.
- Per-call `metadata=` on `track` / `run` / `cache` / `cached_call`, instance-wide `metadata=`, and `tally.context(**metadata)` (safe across threads and asyncio). The report's new `metadata_costs` and a new dashboard section show which user, tenant, or prompt version costs the most.
- `log_previews=False` and `redact=callable` to keep prompt and response text out of logs and the dashboard.
- `on_record=` sinks / `add_sink()`, and `persist_logs=False`, for custom storage. `async_sinks=True` (with `sink_queue_size`) delivers records on a background thread.
- `max_log_mb=` / `log_backups=` log rotation, and `llm-tally compact`.
- `tally.wrap_context(fn)` carries `tally.context(...)` metadata into thread-pool workers.
- `cache_backend=` for a shared or custom cache.
- `retry_jitter`, `max_retry_after`, `RetryConfig.compute_wait()`, and `extract_retry_after()`.
- `pricing_file=` / `tally.load_pricing()` / `PricingRegistry.load_file()` for JSON price lists.
- `days=` on `get_records` / `get_report` / `render_dashboard` / `generate_dashboard`, and `dashboard_days=`.
- `llm-tally` command-line tool (`report`, `dashboard`), also runnable as `python -m llm_tally`.
- `py.typed` marker, and a CI job that builds the wheel and checks it includes the templates.
