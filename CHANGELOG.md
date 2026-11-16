# Changelog

## Unreleased

### Breaking changes
- **No default model.** `default_model` now defaults to `None`. A call whose model can't be detected is logged as `"unknown"` with cost `$0` and `unpriced=True`, instead of being labelled and priced as `gpt-4o-mini`. Pass `default_model="..."` to keep the old behaviour.
- **Unknown models are not priced with a guessed rate.** They cost `$0` and are flagged `unpriced=True` (new `CallRecord` field), counted in `overview["unpriced_calls"]` / `overview["unpriced_models"]`, and shown in a dashboard notice. `fallback_pricing=(in, out)` restores a placeholder rate. `PricingRegistry.resolve()` still falls back; the new `PricingRegistry.lookup()` returns `None`.
- `run()`, `cached_call()`, and `retry_call()` pass options that the target function declares (e.g. `model`, `metadata` on SDK methods) through to it instead of consuming them. Use the new `tally_options={...}` for llm-tally options with colliding names.
- With no retries configured (`default_retries=0` / `retries=0`) the original exception is raised instead of `RetryExhaustedError`.
- A returned dict is only treated as an HTTP response if it has `status_code` / `http_status`; a `status` field counts only on objects that also have `headers`.
- Input previews show the arguments the caller passed by name (without `self` and defaults), or the text itself for a single text argument. Output previews show the answer text of SDK responses.
- A stream's placeholder record has `status="streaming"` until it finishes; `close()` before the end logs it as `incomplete`.
- `overview["top_cost_feature"]` now adds up all configs of a feature (new `feature_totals` list). Latency figures leave out cache hits and failed calls.
- Report days use the local time zone (or the time zone of `now=`) instead of UTC.

### Fixed
- The model is read from a positional `model` argument and from the function's `model=` default, not only from a `model=` keyword.
- The model is read from LangChain messages (`response_metadata`).
- Gemini thinking tokens (`thoughts_token_count`) and tool-use prompt tokens are billed.
- OpenAI Responses API streams (`response.output_text.delta`, `response.completed`) record real usage and model.
- GPT-4 Turbo preview snapshots, `gpt-4o-2024-05-13`, audio/transcribe/TTS models, `:free` variants, and Bedrock `global.` / `apac.` / `us-gov.` prefixes no longer resolve to the wrong price.
- `on_record` sinks receive records even when the log file or dashboard can't be written.
- Tracked methods no longer hash `self` on every call when caching is off; a deep object graph no longer raises `RecursionError` before the call runs.
- Secret-looking arguments (`api_key`, `password`, `token`, `Authorization`, ...) are masked in previews.
- Reports and the dashboard include rotated log backups, so totals no longer reset at a rotation.
- Dashboard rebuilds for large logs run on a background thread instead of inside the tracked call.
- The dashboard file is written with mode 0644 (was 0600).
- The unknown-model warning appears once per model and points at the calling line.
- Forked worker processes reset llm-tally's locks and timers.
- `@tally.cache` no longer fails when the wrapped function has a parameter named like a cache option (`ttl`, `model`, ...).

### Added
- `fallback_pricing=`, `raise_original_error=`, `dashboard_recent_calls=` options; `LLM_TALLY_DIR` environment variable.
- Calls logged as unpriced are re-priced in reports once their model has a price (`register_model`, `load_pricing`, or `llm-tally report --pricing-file`).
- `PRICING_AS_OF` (date the built-in prices were last reviewed), shown on the dashboard.
- Dashboard: notices for unpriced and estimated calls, expandable errors with every retry attempt, full input/output/config per call, "latest N of M" in search, a filter for unpriced calls, local times, phone-width layout, and 30-day spend.
- CLI: `--pricing-file`, `--backups`, and a message when the log file is missing.

## 0.2.0

### Breaking changes
- The package was renamed from `django-llm-tally` / `django_llm_tally` to `llm-tally` / `llm_tally`, and the Django helpers (`get_django_llm_tally_settings`, `make_django_dashboard_view`) were removed.
- **Caching is now opt-in.** `default_cache_enabled` defaults to `False`. Pass `cache=True` per call, or `default_cache_enabled=True` to restore the old behaviour. `@tally.cache`, `cached_call`, and `cached_llm_call` still always cache.
- **Dashboard auto-updates are throttled** to at most once per `dashboard_update_interval` seconds (default 2.0). Pending updates are written in the background, at exit, or by `tally.flush()`. Set `dashboard_update_interval=0` for the old write-on-every-call behaviour.
- **HTTP 429 / rate-limit errors are now retried** (honoring `Retry-After`). Set `retry_on_rate_limit=False` to disable this.
- **The JSONL log is append-only.** A finished stream appends its final record instead of rewriting the file in place, so a stream call takes two lines and readers keep the last line per `id`. `JSONLLogger.compact()` / `llm-tally compact` rewrites the file with one line per call. If other tools read the file directly, de-duplicate by `id`.
- `log_previews=False` now also hides error message text (exception types and status codes are kept), and `redact` now also applies to error messages, retry-attempt messages, and `config` strings.

### Fixed
- Functions created by a factory (closures with the same qualified name) no longer share cache entries. Each live closure gets its own cache identity.
- A *returned* response object with HTTP 429 is now retried like a raised rate-limit error (unless `retry_on_rate_limit=False`), and `Retry-After` is read from dict-style responses too.
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
