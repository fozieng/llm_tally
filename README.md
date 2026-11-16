# llm-tally

Zero-dependency Python toolkit for **self-evicting in-memory LLM caching**, **resilient 5xx / 429 / network retries**, and **per-feature cost & reliability analytics** with a self-contained **HTML dashboard**.

Works in any Python 3.9+ application with zero external runtime dependencies. Fully typed (`py.typed`).

---

## Core Features

### 1. Self-Evicting Local In-Memory Cache (opt-in)
- **Input + Settings Snapshot**: Hashes the function's arguments (`args`, `kwargs`) together with its settings (`model`, `feature`, `config`) and stores the generated `output`.
- **Instant Cache Hits**: Calling the function again with the same input and settings returns a copy of the cached output without invoking the LLM.
- **Lifetime (`ttl`) Auto-Vanish**: Entries expire after `ttl` seconds (default `300s`).
- **Kill After `X` Requests (`max_requests`)**: An entry removes itself after being fetched `X` times.
- **Off by default in `@tally.track`.** LLM output with `temperature > 0` is supposed to vary, so caching only happens where you ask for it (`cache=True`, `@tally.cache`, or `default_cache_enabled=True`).
- **Pluggable**: pass `cache_backend=` to share a cache across workers (e.g. a Redis-backed object with the same `fetch`/`store`/`clear` methods as `InMemoryCache`).
- Factory-created closures have separate cache identities. These identities last for the function's lifetime in its process, so closure entries are not shared across workers or restarts.

### 2. Smart Retries on HTTP 5xx, 429 Rate Limits & Network Issues
- Retries **HTTP 500/5xx** errors (raised exceptions or returned response objects), **HTTP 429 / `RateLimitError`**, and **network/timeout** errors (`ConnectionError`, `TimeoutError`, `URLError`, SDK connection/timeout exceptions).
- Retries **3 times** by default with delays of **`0.5s`, `1.0s`, `1.5s`** (configurable).
- Honors the server's **`Retry-After` / `retry-after-ms`** header (capped by `max_retry_after`, default 60s). Optional `retry_jitter` spreads out retries from many workers.
- If all attempts fail, raises `RetryExhaustedError` with every attempt's error (`err.attempts`, `err.errors`).

### 3. Per-Feature Cost Analyzer & Interactive HTML Dashboard
- **Built-in pricing** for OpenAI, Anthropic, and Google Gemini models, plus `register_model(...)` or a JSON `pricing_file` for anything else.
- **No guessed prices**: a model without a known price is logged with cost `$0`, flagged `unpriced=True`, and listed on the dashboard. Register its price later and past calls are re-priced in reports.
- **Model attribution**: uses the decorator's `model=`, then `config["model"]`, then the model reported in the response, then the `model` argument your function was called with (by keyword, by position, or its default). If none of these exist the model is recorded as `unknown`.
- **Estimated vs. real tokens**: when a response carries no token usage, tokens are estimated from text length and the record is flagged `estimated_tokens=True` (shown as `~$` in the dashboard).
- **Per-call metadata**: tag calls with `user_id`, `tenant`, `prompt_version`, ... and see which ones cost the most.
- **Self-Contained Static HTML Dashboard (`.llm_tally/dashboard.html`)** answers:
  1. **How much money did I spend this week?**
  2. **Which feature costs the most?** (per feature and per config)
  3. **Which user / tenant / prompt version costs the most?** (per metadata key)
  4. **Which calls fail the most?**
  5. **Which calls are slow?** (Average, P95, Max)
  6. **Interactive search** across features, models, configs, metadata, inputs/outputs, statuses, and errors.

---

## Installation

```bash
pip install llm-tally
```

---

## Quick Start

```python
from llm_tally import LLMTally

tally = LLMTally()  # logs to .llm_tally/logs.jsonl, dashboard at .llm_tally/dashboard.html

@tally.track(feature="support_chatbot")
def ask_llm(prompt: str, model: str = "gpt-4o") -> dict:
    # Replace with an OpenAI / Anthropic / Gemini SDK call and return its response
    return {
        "content": f"Answer to: {prompt}",
        "usage": {"input_tokens": 120, "output_tokens": 45},
    }

ask_llm("How do I reset my password?")                       # priced as gpt-4o (the default)
ask_llm("How do I reset my password?", model="gpt-4o-mini")  # priced as gpt-4o-mini
```

Open `.llm_tally/dashboard.html` in a browser to see the results. Add `.llm_tally/` to your
`.gitignore`: the log contains prompt previews. Set the `LLM_TALLY_DIR` environment variable to an
absolute path to keep every process (and the CLI) on the same log no matter where it starts.

### Exact costs need the SDK response

llm-tally reads token counts and the model from the object your function returns. **Return the SDK
response object** (or a dict with a `usage` key) for exact costs. If your function returns only the
text, tokens are estimated (about 4 characters per token, marked `~` on the dashboard), and the
model must come from `model=` on the decorator or a `model` argument; otherwise the call is logged
as model `unknown` with no price.

Supported response shapes: OpenAI Chat Completions and Responses API (including streams with
`stream_options={"include_usage": True}` and `response.completed` events), Anthropic Messages (and
`messages.stream()`), Google Gemini (`usage_metadata`, including thinking tokens), LangChain
messages (`usage_metadata` / `response_metadata`), and dicts with `usage` / `input_tokens` /
`prompt_tokens`.

Every constructor argument is optional, and passing `None` always means "use the default", so you can forward optional settings directly:

```python
tally = LLMTally(
    default_model=os.getenv("LLM_MODEL"),    # None -> no assumed model ("unknown" if undetectable)
    log_path=os.getenv("LLM_TALLY_LOG"),     # None -> ".llm_tally/logs.jsonl"
    default_cache_enabled=True,              # opt in to caching for every tracked call
    default_ttl=300.0,
    default_retries=3,
    retry_delays=(0.5, 1.0, 1.5),
)
```

---

## Which function should I use?

| You want to...                                         | Use |
|--------------------------------------------------------|-----|
| Track cost + retry (+ optional cache) on a function    | `@tally.track(...)` |
| Same thing, but for one call without a decorator       | `tally.run(func, *args, **options)` |
| Cache a function (with cost tracking)                  | `@tally.cache(...)` / `tally.cached_call(func, ...)` |
| Cache only, no logging, no tally instance              | `cached_llm_call(func, ...)` |
| Retry only, no logging                                 | `tally.retry_call(func, ...)` |

`run`, `cached_call`, and `retry_call` take their own options (`feature`, `model`, `config`, `metadata`, `ttl`, ...) as keyword arguments. **An option that `func` itself declares is passed to `func`**, so SDK methods work directly:

```python
tally.run(
    client.chat.completions.create,
    model="gpt-4o",                       # sent to the SDK *and* used for pricing
    messages=[{"role": "user", "content": "hi"}],
    metadata={"trace": "abc"},            # the SDK declares `metadata`, so the SDK gets it
    tally_options={"feature": "chat", "metadata": {"user_id": "u1"}},  # llm-tally's own options
)
```

`model` is also passed to functions that take `**kwargs`. Use `tally_options={...}` for any
llm-tally option whose name collides with one of `func`'s parameters.

---

## Caching (Feature 1)

```python
@tally.track(feature="faq", cache=True, ttl=120.0, max_requests=5, config={"temperature": 0})
def answer_faq(question: str) -> str:
    ...

from llm_tally import cached_llm_call

out = cached_llm_call(expensive_prompt, "Quantum Computing", ttl=60.0, max_requests=2,
                      settings={"temperature": 0})
```

After `ttl` seconds the entry vanishes on its next fetch; after `max_requests` hits it kills itself, and the next call generates a fresh response.

---

## Retries (Feature 2)

```python
from llm_tally import LLMTally, RetryExhaustedError

tally = LLMTally(
    retry_on_rate_limit=True,   # retry HTTP 429 / RateLimitError (default)
    retry_jitter=0.2,           # add up to +20% random wait per retry
    max_retry_after=30.0,       # never wait longer than 30s for a Retry-After header
)

@tally.track(feature="payment_summary", model="claude-3-7-sonnet")
def flaky_call():
    raise ConnectionError("Upstream LLM gateway reset connection")

try:
    flaky_call()
except RetryExhaustedError as err:
    for attempt in err.attempts:
        print(attempt.attempt, attempt.delay_before, attempt.status_code, attempt.message)
```

Client errors (400, 401, 403, 404, 422) are never retried. Use `extra_exceptions=(MyError,)` to retry your own exception types.

**Turn off one layer of retries.** The OpenAI and Anthropic SDKs already retry 429 / 5xx /
connection errors themselves. With both layers on, one call can turn into many requests. Either
create the client with `max_retries=0`, or pass `retry=False` / `default_retry_enabled=False` here.
Retries also re-run the whole function, so keep side effects (database writes, emails) outside
tracked functions or turn retries off for them.

**Keep your own `except` blocks working.** By default exhausted retries raise
`RetryExhaustedError`. With `LLMTally(raise_original_error=True)` the SDK's own exception (e.g.
`openai.RateLimitError`) is re-raised instead, with the attempt history in
`exc.llm_tally_attempts`. When no retries are configured (`default_retries=0`) the original
exception is always raised.

Only `status_code` / `http_status` (and `status` on objects that also carry `headers`) on a
*returned* value count as an HTTP status, so returning your own `{"status": 503, ...}` payload is
not treated as a server error.

---

## Per-Call Metadata (users, tenants, prompt versions)

```python
tally = LLMTally(metadata={"env": "prod"})            # added to every record

@tally.track(feature="chat", metadata={"prompt_version": "v3"})
def chat(prompt: str): ...

with tally.context(user_id=user.id, tenant=org.slug):  # thread- and asyncio-safe
    chat("hello")

tally.run(other_func, "hi", metadata={"request_id": rid})

report = tally.get_report()
print(report["metadata_costs"]["user_id"][0])  # most expensive user
```

Precedence: instance `metadata` < `tally.context(...)` < decorator / `run(metadata=...)`.

Context metadata follows asyncio tasks automatically, but Python does not pass it to `ThreadPoolExecutor` / `threading.Thread` workers. Wrap the function you submit:

```python
with tally.context(user_id="alice"), ThreadPoolExecutor() as pool:
    pool.map(tally.wrap_context(chat), prompts)
```

---

## Privacy: Prompt Previews

By default the first 140 characters of each input and output are stored for search. Previews show
the arguments the caller passed (without `self`), and the answer text of SDK responses. Values of
arguments or dict keys that look like secrets (`api_key`, `password`, `token`, `authorization`,
`secret`, `credentials`, ...) are replaced with `[redacted]`. To turn previews off or mask other
data:

```python
import re

# Store no prompt/response text, and hide error message text (exception types and status codes are kept)
tally = LLMTally(log_previews=False)

# Mask text in previews, error messages, retry-attempt messages, and string values in `config`
tally = LLMTally(redact=lambda text: re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", text))
```

If your `redact` function raises, the text is stored as `"[redaction failed]"` and a warning is shown. The call is still logged. `metadata` values are stored exactly as you pass them.

---

## Custom Storage & Sinks

Every finished call is passed as a `CallRecord` to your `on_record` callables. Use them to send records to a database, OpenTelemetry, a queue, etc. A failing sink only warns; it never breaks the LLM call.

```python
def ship(record):
    db.insert("llm_calls", record.to_dict())

tally = LLMTally(on_record=ship)                       # JSONL file + your sink
tally = LLMTally(on_record=ship, persist_logs=False)   # your sink only (no local file)
tally = LLMTally(on_record=ship, async_sinks=True)     # run sinks on a background thread
tally.add_sink(another_sink)
```

By default sinks run inside the call, so a slow sink adds its latency to every LLM call. With `async_sinks=True` records go through a bounded background queue (`sink_queue_size`, default 10,000). If the queue fills up, records are dropped with a warning. `tally.flush()` (also run at exit) waits for queued records to be delivered.

Streams produce exactly one record each, once they finish. A stream that is abandoned before it ends is logged with `status="incomplete"`. An SDK stream manager that fails when the `with` block opens (e.g. Anthropic `messages.stream()` failing to connect) is logged as `failed` with the real error. Retries don't apply to it, because the SDK sends that request on entry.

### The JSONL log file
- **Append-only and safe across processes**: every write is one appended line under a POSIX file lock, so multiple threads and processes on one machine never lose records. A finished stream appends its final version, and readers keep the **last line for each `id`**. Run `tally.logger.compact()` or `llm-tally compact` to rewrite the file with one line per call.
- **Incremental reads**: reports and the dashboard only parse what was appended since the last read.
- **Rotation**: `LLMTally(max_log_mb=100, log_backups=3)` moves the log to `logs.jsonl.1` ... `.3` when it would exceed 100 MB. Reports and the dashboard read the current file and the backups, so totals don't reset at a rotation. The oldest backup is deleted when a new one is made.

---

## Custom Models & Pricing

```python
tally.register_model("my-company-fine-tune", input_cost_per_1m=1.20, output_cost_per_1m=4.80)

# Or keep prices in a file you can update without a release:
tally = LLMTally(pricing_file="prices.json")
tally.load_pricing("more_prices.json")
```

```json
{
  "my-company-fine-tune": {"input_cost_per_1m": 1.2, "output_cost_per_1m": 4.8, "provider": "custom"},
  "gpt-4o": {"input_cost_per_1m": 2.5, "output_cost_per_1m": 10.0, "cached_input_cost_per_1m": 1.25}
}
```

**Unknown models are not guessed.** A model that is not in the catalog (or a call whose model
can't be detected, logged as `unknown`) is recorded with cost `$0` and `unpriced=True`. You get one
warning per model (a `UserWarning`, or a message on the `llm_tally` logger if your app configures
logging), and the dashboard shows a notice listing those models. After you register a price, past
calls are re-priced automatically in `get_report()`, `get_records()`, and the dashboard. The CLI
does the same with `--pricing-file prices.json`.

To use a rough placeholder rate instead of `$0`, opt in with
`LLMTally(fallback_pricing=(0.50, 1.50))`. Those calls stay flagged `unpriced`.

The built-in prices were last reviewed in **`llm_tally.pricing.PRICING_AS_OF`** (shown at the
bottom of the dashboard). Provider prices change and new models appear often, so check your
provider's price list and use `pricing_file=` for anything new.

Not modelled: long-context price tiers, Batch API discounts, service tiers, 1-hour cache writes,
image / audio tokens, built-in tool fees, and embeddings. Calls that fail after the provider
charged for them (e.g. your code raises while parsing the response) are logged at `$0`.

---

## Dashboard & Reports

The dashboard is regenerated automatically, but **at most once every `dashboard_update_interval` seconds** (default `2.0`), so the overhead per call stays flat as logs grow. If a rebuild is slow (a very large log), the interval stretches to 10× the rebuild time, so rebuilding never uses more than about 10% of a core. Once the log passes 2 MB (or a rebuild takes over 50 ms) rebuilds run on a background thread, so a tracked call never waits for one. A pending update is written in the background when the interval ends, and at interpreter exit.

How to read it:
- "Which feature costs the most?" adds up every config of a feature; the table below splits them.
- Latency figures use live calls only. Cache hits and failed calls (which include retry waits)
  are left out.
- Days are in the time zone of the machine that built the page; call times are shown in the
  viewer's local time.
- The search table holds the latest `dashboard_recent_calls` calls (default 500).
- `~$` marks estimated tokens; `? $` marks calls whose model has no price.

In production with many workers, set `auto_update_dashboard=False` and build the dashboard from
one place (`llm-tally dashboard`, a cron job, or a sink-backed store passed to
`write_dashboard_file(records, ...)`). Each process keeps the parsed log in memory, so use
`max_log_mb=` to bound it.

```python
tally = LLMTally(dashboard_update_interval=10.0, dashboard_days=30)  # last 30 days only

report = tally.get_report(days=7)
print("Spent this week ($):", report["spend_summary"]["this_week_usd"])
print("Most expensive feature:", report["overview"]["top_cost_feature"])
print("Estimated-token calls:", report["overview"]["estimated_calls"])

tally.flush()                           # write any pending dashboard update now
path = tally.generate_dashboard()       # or force a full rebuild
```

Set `dashboard_update_interval=0` to rebuild on every call, or `auto_update_dashboard=False` to only build it yourself.

### Command line

```bash
llm-tally report                                  # summary of .llm_tally/logs.jsonl
llm-tally report --days 7 --json                  # full analytics as JSON
llm-tally report --pricing-file prices.json       # re-price calls logged as unpriced
llm-tally dashboard --log logs.jsonl --out dash.html --days 30
llm-tally compact --log logs.jsonl                # one line per call
python -m llm_tally report                        # same, without the console script
```

---

## License

MIT
