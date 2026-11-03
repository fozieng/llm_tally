"""
Feature 3 (Part 3): Self-Contained Interactive HTML Dashboard Generator.

Loads the frontend assets from `src/llm_tally/templates/` (`dashboard.html`,
`dashboard.css`, `dashboard.js`) and injects the `analyze_logs()` JSON payload in a
single regex pass so prompt/title text containing placeholder tokens can never
corrupt the generated HTML or JSON.
"""

from __future__ import annotations

import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

from .tracker import CallRecord, analyze_logs

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_PLACEHOLDER_RE = re.compile(r"\{\{(TITLE|DASHBOARD_CSS|REPORT_JSON|DASHBOARD_JS)\}\}")


def _load_template_asset(filename: str) -> str:
    """Read a template asset (`dashboard.html`, `dashboard.css`, or `dashboard.js`) from `templates/`."""
    asset_path = _TEMPLATES_DIR / filename
    return asset_path.read_text(encoding="utf-8")


def _safe_script_json(data: Any) -> str:
    """
    Serialize `data` to JSON safe for embedding inside an HTML `<script>` block.
    Escapes `<`, `>`, `&`, U+2028, and U+2029 so user prompts/outputs can never
    close the `<script>` tag or break JS string parsing.
    """
    raw = json.dumps(data, ensure_ascii=False, default=str)
    return (
        raw.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_dashboard_html(
    records: Sequence[CallRecord],
    title: str = "llm-tally — Cost & Reliability Dashboard",
) -> str:
    """
    Assemble a complete, self-contained HTML dashboard string by combining the
    `dashboard.html`, `dashboard.css`, and `dashboard.js` template files with
    the JSON analytics report from `analyze_logs(records)` in a single pass.
    """
    report: Dict[str, Any] = analyze_logs(records)
    report_json = _safe_script_json(report)

    html_template = _load_template_asset("dashboard.html")
    css_content = _load_template_asset("dashboard.css")
    js_content = _load_template_asset("dashboard.js")

    replacements = {
        "TITLE": html.escape(title),
        "DASHBOARD_CSS": css_content,
        "REPORT_JSON": report_json,
        "DASHBOARD_JS": js_content,
    }
    return _PLACEHOLDER_RE.sub(lambda match: replacements[match.group(1)], html_template)


def write_dashboard_file(
    records: Sequence[CallRecord],
    dashboard_path: str | Path = ".llm_tally/dashboard.html",
    title: str = "llm-tally — Cost & Reliability Dashboard",
) -> Path:
    """Generate and atomically write the static HTML dashboard file to disk."""
    target = Path(dashboard_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    html_content = render_dashboard_html(records, title=title)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".dashboard_",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(html_content)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target
