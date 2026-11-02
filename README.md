# django-llm-tally

Zero-dependency Python & Django package for **self-evicting in-memory LLM caching**, **resilient HTTP 500/network retries**, and **per-feature cost & reliability analytics** with a self-contained **HTML dashboard**.

Works in any Python 3.9+ application with zero external runtime dependencies, and includes optional first-class Django integration via `settings.LLM_TALLY`.

---

## Core Features

### 1. Self-Evicting Local In-Memory Cache (`cached_call` / `@tally.cache`)
The cache function (`tally.cached_call`, `@tally.cache`, or standalone `cached_llm_call`) creates a thread-safe local cache in memory:
- **Input + Settings Snapshot**: Hashes and stores the function's input arguments (`args`, `kwargs`) together with its configuration settings (`model`, `feature`, `config`) alongside the generated `output`.
- **Instant Cache Hits**: Calling the function with the exact same stored input and settings returns the cached output without invoking the LLM.
- **Lifetime (`ttl`) Auto-Vanish**: Every cached entry has a configurable lifetime (`ttl` in seconds, default `300s`). Once the lifetime expires, the next time the entry is fetched it **vanishes itself** from memory and executes the function to generate and store a new response.
- **Kill After `X` Requests (`max_requests`)**: You can configure an entry to **kill itself after `X` requests** (`max_requests=X`). Once the cached entry has been fetched `X` times, it vanishes from memory so the subsequent call generates a fresh response.

### 2. Smart Retry on HTTP 500 & Network Issues (`retry_call` / `@tally.track`)
- Automatically detects **HTTP 500 / 5xx server errors** (both raised exceptions with `status_code >= 500` and response objects with `status_code >= 500`) and **network/timeout issues** (`ConnectionError`, `TimeoutError`, `URLError`, and SDK connection/timeout exceptions).
- Retries **3 times** by default with backoff delays of **`0.5s`, `1.0s`, and `1.5s`** (configurable).
- If all attempts fail, raises `RetryExhaustedError` containing the full list of errors from every attempt (`err.attempts` and `err.errors`).

### 3. Per-Feature Cost Analyzer & Interactive HTML Dashboard
- **Built-in Pricing for OpenAI, Anthropic, and Google Gemini**: Preloaded with token pricing for `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `o1`, `o3`, `claude-sonnet-4`, `claude-3-7-sonnet`, `claude-3-5-sonnet`, `claude-3-5-haiku`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`, and more—while always letting you choose or register any custom model.
- **Self-Contained Static HTML Dashboard (`.llm_tally/dashboard.html`)**: Automatically updates and answers:
  1. **How much money did I spend this week?** (7-day total, daily bar chart, today/all-time spend, and money saved by caching)
  2. **Which feature costs the most?** (Ranked cost table per feature and per config)
  3. **Which calls fail the most?** (Failure rates, retry counts, and recent errors per feature/function)
  4. **Which calls are slow?** (Average, P95, and Max latency leaderboard)
  5. **Interactive Search**: Live client-side search across features, models, configs, inputs/outputs, Statuses, and error messages.

---

## Installation

```bash
pip install django-llm-tally
```

---

## Quick Start

```python
from django_llm_tally import LLMTally, RetryExhaustedError

tally = LLMTally(
    default_model="gpt-4o-mini",
    log_path=".llm_tally/logs.jsonl",
    dashboard_path=".llm_tally/dashboard.html",
    default_ttl=300.0,         # Cache lifetime: 5 minutes
    default_max_requests=10,   # Kill cached entry after 10 fetches
    default_retries=3,         # Retry 3 times on 500 or network error
    retry_delays=(0.5, 1.0, 1.5),
)

# Wrap any sync or async LLM function
@tally.track(
    feature="support_chatbot",
    model="gpt-4o",
    config={"temperature": 0.2, "system": "helpful_agent"},
    cache=True,
    ttl=120.0,         # Vanishes on fetch after 120 seconds
    max_requests=5,    # Kills itself after 5 cached requests
)
def ask_llm(prompt: str) -> dict:
    # Replace with OpenAI / Anthropic / Gemini SDK call
    return {
        "content": f"Answer to: {prompt}",
        "usage": {"input_tokens": 120, "output_tokens": 45},
    }

response = ask_llm("How do I reset my password?")
```

---

## Using the Dedicated Cache Function (Feature 1)

Developers should note how `tally.cached_call` (and standalone `cached_llm_call`) works:
1. It stores `(input + config settings) -> output` in local memory.
2. When called with the same input and settings before expiration, it returns the stored output.
3. If fetched after `ttl` seconds, the entry vanishes itself and generates a fresh response.
4. If `max_requests=X` is configured, the entry kills itself after `X` cached requests so the next request generates a new response.

```python
from django_llm_tally import cached_llm_call

def expensive_prompt(topic: str, style: str = "concise") -> str:
    return f"Generated summary for {topic} ({style})"

# First call: executes `expensive_prompt` and caches the output
out1 = cached_llm_call(
    expensive_prompt,
    "Quantum Computing",
    style="concise",
    ttl=60.0,        # Lifetime of 60 seconds
    max_requests=2,  # Kills itself after 2 cached fetches
    settings={"temperature": 0.5},
)

# Hit #1 and Hit #2 return the cached output; after Hit #2, the entry kills itself.
# The next call will execute `expensive_prompt` again to generate a fresh response.
```

---

## Handling Retries & Aggregated Errors (Feature 2)

```python
from django_llm_tally import LLMTally, RetryExhaustedError

tally = LLMTally()

@tally.track(feature="payment_summary", model="claude-3-7-sonnet")
def flaky_call():
    raise ConnectionError("Upstream LLM gateway reset connection")

try:
    flaky_call()
except RetryExhaustedError as err:
    print(err)
    # Prints all 4 failed attempts (initial + 3 retries at 0.5s, 1.0s, 1.5s)
    for attempt in err.attempts:
        print(attempt.attempt, attempt.delay_before, attempt.error_type, attempt.message)
```

---

## Custom Models & Pricing (Feature 3)

All major OpenAI, Anthropic, and Google Gemini models are included out of the box. You can also register custom models or override existing rates (USD per 1M tokens):

```python
tally.register_model(
    model="my-company-fine-tune",
    input_cost_per_1m=1.20,
    output_cost_per_1m=4.80,
    provider="custom",
)
```

To inspect analytics programmatically or generate the HTML dashboard:

```python
report = tally.get_report()
print("Spent this week ($):", report["spend_summary"]["this_week_usd"])
print("Most expensive feature:", report["overview"]["top_cost_feature"])
print("Most failing call:", report["overview"]["most_failing_call"])
print("Slowest call:", report["overview"]["slowest_call"])

# Write/update the static HTML dashboard file
dashboard_file = tally.generate_dashboard()
print(f"Open {dashboard_file} in your browser!")
```

---

## Optional Django Integration

If you use `django-llm-tally` in a Django project, you can configure defaults in `settings.py`:

```python
# settings.py
LLM_TALLY = {
    "DEFAULT_MODEL": "gemini-2.5-flash",
    "LOG_PATH": ".llm_tally/logs.jsonl",
    "DASHBOARD_PATH": ".llm_tally/dashboard.html",
    "CACHE_ENABLED": True,
    "DEFAULT_TTL": 300.0,
    "DEFAULT_MAX_REQUESTS": 20,
    "RETRY_ENABLED": True,
    "DEFAULT_RETRIES": 3,
    "RETRY_DELAYS": (0.5, 1.0, 1.5),
}
```

And optionally expose the live HTML dashboard in `urls.py`:

```python
# urls.py
from django.urls import path
from django_llm_tally import make_django_dashboard_view

urlpatterns = [
    path("admin/llm-tally/", make_django_dashboard_view(), name="llm_tally_dashboard"),
]
```

---

## License

MIT

