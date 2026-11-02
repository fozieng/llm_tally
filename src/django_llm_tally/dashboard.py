"""
Feature 3 (Part 3): Self-Contained Interactive HTML Dashboard Generator.

Loads the frontend assets from `src/django_llm_tally/templates/` (`dashboard.html`,
`dashboard.css`, `dashboard.js`) and injects the `analyze_logs()` JSON payload so
all UI rendering stays cleanly separated in the frontend files.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from .tracker import CallRecord, analyze_logs

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _load_template_asset(filename: str) -> str:
    """Read a template asset (`dashboard.html`, `dashboard.css`, or `dashboard.js`) from `templates/`."""
    asset_path = _TEMPLATES_DIR / filename
    return asset_path.read_text(encoding="utf-8")


def render_dashboard_html(
    records: Sequence[CallRecord],
    title: str = "django-llm-tally — Cost & Reliability Dashboard",
) -> str:
    """
    Assemble a complete, self-contained HTML dashboard string by combining the
    `dashboard.html`, `dashboard.css`, and `dashboard.js` template files with
    the JSON analytics report from `analyze_logs(records)`.
    """
    report: Dict[str, Any] = analyze_logs(records)
    report_json = json.dumps(report, ensure_ascii=False, default=str).replace("</", "<\\/")

    html_template = _load_template_asset("dashboard.html")
    css_content = _load_template_asset("dashboard.css")
    js_content = _load_template_asset("dashboard.js")

    return (
        html_template.replace("{{TITLE}}", html.escape(title))
        .replace("{{DASHBOARD_CSS}}", css_content)
        .replace("{{REPORT_JSON}}", report_json)
        .replace("{{DASHBOARD_JS}}", js_content)
    )


def write_dashboard_file(
    records: Sequence[CallRecord],
    dashboard_path: str | Path = ".llm_tally/dashboard.html",
    title: str = "django-llm-tally — Cost & Reliability Dashboard",
) -> Path:
    """Generate and write the static HTML dashboard file to disk."""
    target = Path(dashboard_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    html_content = render_dashboard_html(records, title=title)
    target.write_text(html_content, encoding="utf-8")
    return target
