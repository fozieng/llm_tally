"""
Command-line interface: `llm-tally dashboard`, `llm-tally report`, and `llm-tally compact`.

    llm-tally dashboard --log .llm_tally/logs.jsonl --out dashboard.html --days 7
    llm-tally report --days 30
    llm-tally compact --log .llm_tally/logs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .core import DEFAULT_DASHBOARD_PATH, DEFAULT_DASHBOARD_TITLE, DEFAULT_LOG_PATH
from .dashboard import write_dashboard_file
from .pricing import PricingRegistry
from .tracker import JSONLLogger, analyze_logs, filter_records_since, reprice_records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-tally",
        description="Build the llm-tally HTML dashboard or print a cost report from a JSONL log.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log", default=DEFAULT_LOG_PATH, help=f"JSONL log file (default: {DEFAULT_LOG_PATH})")
    common.add_argument("--days", type=float, default=None, help="Only include the last N days")
    common.add_argument(
        "--pricing-file",
        default=None,
        help="JSON price list; calls logged as unpriced are re-priced with it",
    )
    common.add_argument(
        "--backups", type=int, default=3, help="Rotated backups (<log>.1 ... .N) to include (default: 3)"
    )

    dash = sub.add_parser("dashboard", parents=[common], help="Write the static HTML dashboard")
    dash.add_argument("--out", default=DEFAULT_DASHBOARD_PATH, help=f"Output HTML file (default: {DEFAULT_DASHBOARD_PATH})")
    dash.add_argument("--title", default=DEFAULT_DASHBOARD_TITLE, help="Dashboard title")

    report = sub.add_parser("report", parents=[common], help="Print a cost & reliability summary")
    report.add_argument("--json", action="store_true", help="Print the full analytics report as JSON")

    compact = sub.add_parser("compact", help="Rewrite the log with one line per call")
    compact.add_argument("--log", default=DEFAULT_LOG_PATH, help=f"JSONL log file (default: {DEFAULT_LOG_PATH})")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "compact":
        kept = JSONLLogger(args.log).compact()
        print(f"Compacted {args.log}: {kept} calls")
        return 0

    if not os.path.exists(args.log):
        print(
            f"llm-tally: no log file at {args.log} (pass --log, or set LLM_TALLY_DIR)",
            file=sys.stderr,
        )
    log = JSONLLogger(args.log, backups=args.backups)
    records = filter_records_since(log.read_all(copy_records=False, include_backups=True), args.days)
    pricing = PricingRegistry()
    if args.pricing_file:
        pricing.load_file(args.pricing_file)
    records = reprice_records(records, pricing)

    if args.command == "dashboard":
        path = write_dashboard_file(records, dashboard_path=args.out, title=args.title)
        print(f"Wrote {path} ({len(records)} calls)")
        return 0

    report = analyze_logs(records)
    if args.json:
        json.dump(report, sys.stdout, indent=2, default=str)
        print()
        return 0

    spend = report["spend_summary"]
    overview = report["overview"]
    print(f"Calls:            {overview['total_calls']} ({overview['cached_calls']} cached, {overview['failed_calls']} failed)")
    print(f"Spend today:      ${spend['today_usd']:.4f}")
    print(f"Spend this week:  ${spend['this_week_usd']:.4f}")
    print(f"Spend all time:   ${spend['all_time_usd']:.4f}")
    print(f"Saved by cache:   ${spend['saved_by_cache_usd']:.4f}")
    if overview["estimated_calls"]:
        print(f"Estimated tokens: {overview['estimated_calls']} calls (${overview['estimated_cost_usd']:.4f})")
    if overview["unpriced_calls"]:
        models = ", ".join(f"{m} ({n})" for m, n in overview["unpriced_models"].items())
        print(f"Unpriced calls:   {overview['unpriced_calls']} — no price for: {models}")
    print(f"Top cost feature: {overview['top_cost_feature']}")
    print(f"Most failing:     {overview['most_failing_call']}")
    print(f"Slowest:          {overview['slowest_call']}")
    for key, rows in report["metadata_costs"].items():
        top = rows[0] if rows else None
        if top:
            print(f"Top {key}: {top['value']} (${top['total_cost_usd']:.4f}, {top['total_calls']} calls)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
