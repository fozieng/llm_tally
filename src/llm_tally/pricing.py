"""
Feature 3 (Part 1): Built-in Model Pricing Catalog & Custom Model Pricing Registry.

Includes default token pricing (USD per 1,000,000 input/output tokens) for:
- OpenAI models (GPT-5.1/GPT-5 series, GPT-4.1 series, GPT-4o series, o1/o3/o4 reasoning series, GPT-4/32k, etc.)
- Anthropic models (Claude 4.5 Opus/Sonnet/Haiku, Claude 4.1/4 Opus/Sonnet, Claude 3.7/3.5/3 series)
- Google Gemini models (Gemini 3 Pro/Flash, Gemini 2.5 Pro/Flash/Flash-Lite, Gemini 2.0/1.5 series)

Supports AWS Bedrock (`anthropic.claude-...-v2:0`), Google Vertex AI (`...@20241022`),
OpenAI fine-tuned (`ft:gpt-4o-...`), and `org/model` identifiers out of the box.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

# Matches version/date/modality/snapshot suffixes without matching distinct tier suffixes (-pro, -mini, -nano, -32k, -turbo, -lite)
_VERSION_SUFFIX_RE = re.compile(
    r"^(?:[-:@]"
    r"(?:"
    r"\d{4}-\d{2}-\d{2}"
    r"|\d{8}"
    r"|\d{3,4}"
    r"|v\d+(?::\d+)?"
    r"|latest"
    r"|preview(?:-[\w.:]+)*"
    r"|exp(?:-[\w.:]+)*"
    r"|free"
    r"|chat(?:-latest|-preview|-\d{4}-\d{2}-\d{2})?"
    r"|audio(?:-preview|-\d{4}-\d{2}-\d{2})?"
    r"|realtime(?:-preview|-\d{4}-\d{2}-\d{2})?"
    r"|search(?:-preview|-\d{4}-\d{2}-\d{2})?"
    r"|transcribe(?:-\d{4}-\d{2}-\d{2})?"
    r"|tts(?:-\d{4}-\d{2}-\d{2})?"
    r"))+$"
)

_BEDROCK_PREFIX_RE = re.compile(r"^(?:[a-z]{2}(?:-[a-z]+-\d+)?\.)?(?:anthropic|amazon|meta|mistral)\.")
_BEDROCK_VERSION_RE = re.compile(r"-v\d+(?::\d+)?$")
_VERTEX_AT_RE = re.compile(r"(?:-v\d+)?@(\d{8}|\d{3,4}|default)$")


@dataclass(frozen=True)
class ModelPricing:
    """Token pricing in USD per 1,000,000 (1M) tokens."""

    model: str
    provider: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    cached_input_cost_per_1m: Optional[float] = None
    cache_creation_cost_per_1m: Optional[float] = None

    def compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        input_excludes_cached: Optional[bool] = None,
    ) -> float:
        """
        Calculate the total USD cost for the given token counts.

        Note on provider & response token conventions:
        - **Anthropic format** (`input_excludes_cached=True`, `provider="anthropic"`, `"claude"` in model,
          `cache_creation_input_tokens > 0`, or `cached_input_tokens > input_tokens`):
          `input_tokens` is ONLY the uncached new input tokens (it already excludes
          `cache_read_input_tokens` and `cache_creation_input_tokens`). Default cached read rate
          is `0.10x` base input rate, and `cache_creation_input_tokens` is `1.25x` base input rate.
        - **OpenAI / Gemini format** (`input_excludes_cached=False`):
          `prompt_tokens` / `input_tokens` includes `cached_tokens` in the total, so billable
          uncached input is `max(0, input_tokens - cached_input_tokens)` and default cached rate
          is `0.50x` base input rate.
        """
        excludes_cached = (
            input_excludes_cached
            if input_excludes_cached is not None
            else (
                self.provider == "anthropic"
                or "claude" in self.model.lower()
                or cache_creation_input_tokens > 0
                or (cached_input_tokens > 0 and cached_input_tokens > input_tokens)
            )
        )

        if excludes_cached:
            billable_regular_input = max(0, input_tokens)
            default_cached_rate = self.input_cost_per_1m * 0.10
        else:
            billable_regular_input = max(0, input_tokens - max(0, cached_input_tokens))
            default_cached_rate = self.input_cost_per_1m * 0.50

        cached_rate = (
            self.cached_input_cost_per_1m
            if self.cached_input_cost_per_1m is not None
            else default_cached_rate
        )
        creation_rate = (
            self.cache_creation_cost_per_1m
            if self.cache_creation_cost_per_1m is not None
            else self.input_cost_per_1m * 1.25
        )

        input_cost = (billable_regular_input / 1_000_000.0) * self.input_cost_per_1m
        cached_cost = (max(0, cached_input_tokens) / 1_000_000.0) * cached_rate
        creation_cost = (max(0, cache_creation_input_tokens) / 1_000_000.0) * creation_rate
        output_cost = (max(0, output_tokens) / 1_000_000.0) * self.output_cost_per_1m
        return round(input_cost + cached_cost + creation_cost + output_cost, 8)


# Comprehensive default pricing catalog (USD per 1M tokens)
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # ---------------- OpenAI Models ----------------
    "gpt-5.1": ModelPricing("gpt-5.1", "openai", 1.25, 10.00, 0.125),
    "gpt-5.1-mini": ModelPricing("gpt-5.1-mini", "openai", 0.25, 2.00, 0.025),
    "gpt-5.1-nano": ModelPricing("gpt-5.1-nano", "openai", 0.05, 0.40, 0.005),
    "gpt-5": ModelPricing("gpt-5", "openai", 1.25, 10.00, 0.125),
    "gpt-5-chat": ModelPricing("gpt-5-chat", "openai", 1.25, 10.00, 0.125),
    "gpt-5-mini": ModelPricing("gpt-5-mini", "openai", 0.25, 2.00, 0.025),
    "gpt-5-nano": ModelPricing("gpt-5-nano", "openai", 0.05, 0.40, 0.005),
    "gpt-5-pro": ModelPricing("gpt-5-pro", "openai", 15.00, 120.00),
    "gpt-4.1": ModelPricing("gpt-4.1", "openai", 2.00, 8.00, 0.50),
    "gpt-4.1-mini": ModelPricing("gpt-4.1-mini", "openai", 0.40, 1.60, 0.10),
    "gpt-4.1-nano": ModelPricing("gpt-4.1-nano", "openai", 0.10, 0.40, 0.025),
    "gpt-4o": ModelPricing("gpt-4o", "openai", 2.50, 10.00, 1.25),
    "chatgpt-4o": ModelPricing("chatgpt-4o", "openai", 5.00, 15.00),
    "gpt-4o-audio": ModelPricing("gpt-4o-audio", "openai", 2.50, 10.00),
    "gpt-4o-realtime": ModelPricing("gpt-4o-realtime", "openai", 5.00, 20.00, 2.50),
    "gpt-4o-search": ModelPricing("gpt-4o-search", "openai", 2.50, 10.00),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", "openai", 0.15, 0.60, 0.075),
    "gpt-4o-mini-audio": ModelPricing("gpt-4o-mini-audio", "openai", 0.15, 0.60),
    "gpt-4o-mini-realtime": ModelPricing("gpt-4o-mini-realtime", "openai", 0.60, 2.40, 0.30),
    "gpt-4o-mini-search": ModelPricing("gpt-4o-mini-search", "openai", 0.15, 0.60),
    "gpt-4.5": ModelPricing("gpt-4.5", "openai", 75.00, 150.00, 37.50),
    "gpt-4.5-preview": ModelPricing("gpt-4.5-preview", "openai", 75.00, 150.00, 37.50),
    "o1": ModelPricing("o1", "openai", 15.00, 60.00, 7.50),
    "o1-pro": ModelPricing("o1-pro", "openai", 150.00, 600.00),
    "o1-mini": ModelPricing("o1-mini", "openai", 1.10, 4.40, 0.55),
    "o3": ModelPricing("o3", "openai", 2.00, 8.00, 0.50),
    "o3-pro": ModelPricing("o3-pro", "openai", 20.00, 80.00),
    "o3-mini": ModelPricing("o3-mini", "openai", 1.10, 4.40, 0.55),
    "o4-mini": ModelPricing("o4-mini", "openai", 1.10, 4.40, 0.275),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", "openai", 10.00, 30.00),
    "gpt-4-32k": ModelPricing("gpt-4-32k", "openai", 60.00, 120.00),
    "gpt-4": ModelPricing("gpt-4", "openai", 30.00, 60.00),
    "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", "openai", 0.50, 1.50),
    # ---------------- Anthropic Models ----------------
    "claude-opus-4-5": ModelPricing("claude-opus-4-5", "anthropic", 5.00, 25.00, 0.50, 6.25),
    "claude-4-5-opus": ModelPricing("claude-4-5-opus", "anthropic", 5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-4-5": ModelPricing("claude-sonnet-4-5", "anthropic", 3.00, 15.00, 0.30, 3.75),
    "claude-4-5-sonnet": ModelPricing("claude-4-5-sonnet", "anthropic", 3.00, 15.00, 0.30, 3.75),
    "claude-haiku-4-5": ModelPricing("claude-haiku-4-5", "anthropic", 1.00, 5.00, 0.10, 1.25),
    "claude-4-5-haiku": ModelPricing("claude-4-5-haiku", "anthropic", 1.00, 5.00, 0.10, 1.25),
    "claude-opus-4-1": ModelPricing("claude-opus-4-1", "anthropic", 15.00, 75.00, 1.50, 18.75),
    "claude-4-1-opus": ModelPricing("claude-4-1-opus", "anthropic", 15.00, 75.00, 1.50, 18.75),
    "claude-opus-4": ModelPricing("claude-opus-4", "anthropic", 15.00, 75.00, 1.50, 18.75),
    "claude-4-opus": ModelPricing("claude-4-opus", "anthropic", 15.00, 75.00, 1.50, 18.75),
    "claude-sonnet-4": ModelPricing("claude-sonnet-4", "anthropic", 3.00, 15.00, 0.30, 3.75),
    "claude-4-sonnet": ModelPricing("claude-4-sonnet", "anthropic", 3.00, 15.00, 0.30, 3.75),
    "claude-3-7-sonnet": ModelPricing("claude-3-7-sonnet", "anthropic", 3.00, 15.00, 0.30, 3.75),
    "claude-sonnet-3-7": ModelPricing("claude-sonnet-3-7", "anthropic", 3.00, 15.00, 0.30, 3.75),
    "claude-3-5-sonnet": ModelPricing("claude-3-5-sonnet", "anthropic", 3.00, 15.00, 0.30, 3.75),
    "claude-sonnet-3-5": ModelPricing("claude-sonnet-3-5", "anthropic", 3.00, 15.00, 0.30, 3.75),
    "claude-3-5-haiku": ModelPricing("claude-3-5-haiku", "anthropic", 0.80, 4.00, 0.08, 1.00),
    "claude-haiku-3-5": ModelPricing("claude-haiku-3-5", "anthropic", 0.80, 4.00, 0.08, 1.00),
    "claude-3-opus": ModelPricing("claude-3-opus", "anthropic", 15.00, 75.00, 1.50, 18.75),
    "claude-3-sonnet": ModelPricing("claude-3-sonnet", "anthropic", 3.00, 15.00, 0.30, 3.75),
    "claude-3-haiku": ModelPricing("claude-3-haiku", "anthropic", 0.25, 1.25, 0.03, 0.3125),
    # ---------------- Google Gemini Models ----------------
    "gemini-3-pro": ModelPricing("gemini-3-pro", "gemini", 2.00, 12.00, 0.20),
    "gemini-3.0-pro": ModelPricing("gemini-3.0-pro", "gemini", 2.00, 12.00, 0.20),
    "gemini-3-flash": ModelPricing("gemini-3-flash", "gemini", 0.50, 3.00, 0.05),
    "gemini-3.0-flash": ModelPricing("gemini-3.0-flash", "gemini", 0.50, 3.00, 0.05),
    "gemini-2.5-pro": ModelPricing("gemini-2.5-pro", "gemini", 1.25, 10.00, 0.3125),
    "gemini-2.5-flash": ModelPricing("gemini-2.5-flash", "gemini", 0.15, 0.60, 0.0375),
    "gemini-2.5-flash-lite": ModelPricing("gemini-2.5-flash-lite", "gemini", 0.075, 0.30, 0.01875),
    "gemini-2.0-flash": ModelPricing("gemini-2.0-flash", "gemini", 0.10, 0.40, 0.025),
    "gemini-2.0-flash-lite": ModelPricing("gemini-2.0-flash-lite", "gemini", 0.075, 0.30, 0.01875),
    "gemini-1.5-pro": ModelPricing("gemini-1.5-pro", "gemini", 1.25, 5.00, 0.3125),
    "gemini-1.5-flash": ModelPricing("gemini-1.5-flash", "gemini", 0.075, 0.30, 0.01875),
    "gemini-1.5-flash-8b": ModelPricing("gemini-1.5-flash-8b", "gemini", 0.0375, 0.15, 0.01),
}


def _normalize_platform_model_id(candidate: str) -> str:
    """
    Normalize platform-wrapped model IDs into canonical base forms:
    - OpenAI fine-tuned models: `ft:gpt-4o-mini-2024-07-18:org:custom:id` -> `gpt-4o-mini-2024-07-18`
    - AWS Bedrock IDs: `us.anthropic.claude-3-5-sonnet-20241022-v2:0` -> `claude-3-5-sonnet-20241022`
    - Google Vertex AI IDs: `claude-3-5-sonnet-v2@20241022` -> `claude-3-5-sonnet-20241022`
    """
    norm = candidate.strip().lower()
    if norm.startswith("ft:"):
        parts = norm.split(":")
        if len(parts) >= 2 and parts[1]:
            norm = parts[1]

    norm = _BEDROCK_PREFIX_RE.sub("", norm)
    norm = _BEDROCK_VERSION_RE.sub("", norm)
    norm = _VERTEX_AT_RE.sub(r"-\1", norm)
    return norm


class PricingRegistry:
    """
    Manages LLM pricing lookups and custom user-registered models.
    Supports exact model name matches (including `org/model` names), Claude dot/dash
    variants, Bedrock/Vertex/fine-tuned IDs, and versioned/dated model snapshots.
    """

    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None) -> None:
        self._catalog: Dict[str, ModelPricing] = dict(DEFAULT_PRICING)
        if custom_pricing:
            for k, v in custom_pricing.items():
                self._catalog[k.strip().lower()] = v

    def register(
        self,
        model: str,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
        provider: str = "custom",
        cached_input_cost_per_1m: Optional[float] = None,
        cache_creation_cost_per_1m: Optional[float] = None,
    ) -> ModelPricing:
        """Register or override token pricing for a specific model."""
        norm = model.strip().lower()
        pricing = ModelPricing(
            model=norm,
            provider=provider,
            input_cost_per_1m=float(input_cost_per_1m),
            output_cost_per_1m=float(output_cost_per_1m),
            cached_input_cost_per_1m=cached_input_cost_per_1m,
            cache_creation_cost_per_1m=cache_creation_cost_per_1m,
        )
        self._catalog[norm] = pricing
        return pricing

    def _lookup_single(self, candidate: str) -> Optional[ModelPricing]:
        """Try exact match or strict version-suffix match for a single normalized string."""
        if candidate in self._catalog:
            return self._catalog[candidate]

        # Strict version/date/modality suffix matching (longest key wins)
        best_key: Optional[str] = None
        for key in self._catalog.keys():
            if candidate.startswith(key):
                suffix = candidate[len(key) :]
                if _VERSION_SUFFIX_RE.match(suffix):
                    if best_key is None or len(key) > len(best_key):
                        best_key = key
        if best_key is not None:
            return self._catalog[best_key]

        # Normalize Claude dot/dash variants (e.g. "claude-3.5-sonnet" or "claude-opus-4.5")
        if candidate.startswith("claude-"):
            dashed = candidate.replace(".", "-")
            if dashed in self._catalog:
                return self._catalog[dashed]
            for key in sorted(self._catalog.keys(), key=len, reverse=True):
                if dashed.startswith(key):
                    suffix = dashed[len(key) :]
                    if _VERSION_SUFFIX_RE.match(suffix):
                        return self._catalog[key]

        return None

    def _lookup_candidate(self, candidate: str) -> Optional[ModelPricing]:
        found = self._lookup_single(candidate)
        if found is not None:
            return found

        platform_norm = _normalize_platform_model_id(candidate)
        if platform_norm != candidate:
            return self._lookup_single(platform_norm)
        return None

    def resolve(self, model: str, warn_on_unknown: bool = True) -> ModelPricing:
        """
        Resolve pricing for `model`:
        1. Full model string match, platform ID normalization, or versioned snapshot match.
        2. Short name after `/` if prefixed with a provider/namespace (e.g. `models/gemini-2.5-flash`).
        3. Fallback default (`$0.50 / $1.50` per 1M) with a `UserWarning` so unknown models are visible.
        """
        norm = (model or "gpt-4o-mini").strip().lower()

        # 1. Check full name first (so custom models like "meta-llama/llama-3-70b" match directly!)
        found = self._lookup_candidate(norm)
        if found is not None:
            return found

        # 2. If `norm` contains `/`, try the segment after the last `/`
        if "/" in norm:
            _, _, short_name = norm.rpartition("/")
            found_short = self._lookup_candidate(short_name)
            if found_short is not None:
                return found_short

        if warn_on_unknown:
            warnings.warn(
                f"Unknown model '{model}' not found in pricing registry; falling back to default rate "
                f"($0.50/1M input, $1.50/1M output). Use `tally.register_model('{model}', ...)` to set exact pricing.",
                UserWarning,
                stacklevel=2,
            )
        return ModelPricing(
            model=norm,
            provider="custom",
            input_cost_per_1m=0.50,
            output_cost_per_1m=1.50,
        )

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        input_excludes_cached: Optional[bool] = None,
        warn_on_unknown: bool = True,
    ) -> float:
        """Compute the cost in USD for a model and token counts."""
        pricing = self.resolve(model, warn_on_unknown=warn_on_unknown)
        return pricing.compute_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            input_excludes_cached=input_excludes_cached,
        )

    def list_models(self) -> Dict[str, ModelPricing]:
        """Return a copy of all registered models and their pricing."""
        return dict(self._catalog)
