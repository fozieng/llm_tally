"""
Optional First-Class Django Integration for `django-llm-tally`.

Works cleanly when Django is NOT installed (returns empty settings dict),
and automatically reads `django.conf.settings.LLM_TALLY` when running inside
a configured Django project.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def get_django_llm_tally_settings() -> Dict[str, Any]:
    """
    If Django is installed and `settings.configured` is True, return the
    `getattr(settings, "LLM_TALLY", {})` dictionary. Otherwise return `{}`.
    """
    try:
        from django.conf import settings  # type: ignore[import-not-found]

        if getattr(settings, "configured", False):
            cfg = getattr(settings, "LLM_TALLY", None)
            if isinstance(cfg, dict):
                return dict(cfg)
    except Exception:
        pass
    return {}


def make_django_dashboard_view(tally_instance: Optional[Any] = None) -> Any:
    """
    Return a Django-compatible view function `view(request)` that renders the
    `django-llm-tally` HTML dashboard directly in the browser.
    """

    def _dashboard_view(request: Any) -> Any:
        from django.http import HttpResponse  # type: ignore[import-not-found]

        from .core import get_default_tally

        active_tally = tally_instance or get_default_tally()
        html_str = active_tally.render_dashboard()
        return HttpResponse(html_str, content_type="text/html; charset=utf-8")

    return _dashboard_view

