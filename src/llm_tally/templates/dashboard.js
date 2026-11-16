function formatUsd(val) {
  const num = Number(val || 0);
  if (num === 0) return "$0.00";
  const abs = Math.abs(num);
  if (abs >= 1) {
    return "$" + num.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  if (abs >= 0.01) return "$" + num.toFixed(4);
  // Tiny per-call costs: two significant digits (e.g. $0.000071)
  return "$" + Number(num.toPrecision(2)).toFixed(Math.min(10, 1 - Math.floor(Math.log10(abs))));
}

function escapeHtml(str) {
  return String(str == null ? "" : str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatLocalTime(iso) {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return String(iso || "");
  const pad = (n) => String(n).padStart(2, "0");
  return (
    d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) + " " +
    pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds())
  );
}

function timeAgo(iso) {
  const then = new Date(iso).getTime();
  if (isNaN(then)) return "";
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.round(secs / 60);
  if (mins < 60) return mins + " min ago";
  const hours = Math.round(mins / 60);
  if (hours < 48) return hours + " h ago";
  return Math.round(hours / 24) + " days ago";
}

function modelPill(model) {
  const cls = model === "unknown" ? "pill model unknown" : "pill model";
  const title = model === "unknown" ? ' title="No model could be detected for these calls"' : "";
  return '<span class="' + cls + '"' + title + ">" + escapeHtml(model) + "</span>";
}

function costCell(cost, opts) {
  const o = opts || {};
  const num = Number(cost || 0);
  let cls = "num highlight";
  let prefix = "";
  let title = "";
  if (o.unpriced) {
    cls = "num unpriced-cost";
    prefix = "? ";
    title = "No price for this model: cost not counted (or a fallback guess)";
  } else if (o.estimated) {
    prefix = "~";
    title = "Estimated from text length: the response had no token usage";
  }
  if (num === 0 && !o.unpriced) cls = "num zero";
  return (
    '<td class="' + cls + '"' + (title ? ' title="' + escapeHtml(title) + '"' : "") + ">" +
    prefix + formatUsd(num) + "</td>"
  );
}

function renderNotices(report) {
  const overview = report.overview || {};
  const parts = [];
  const unpriced = Number(overview.unpriced_calls || 0);
  if (unpriced > 0) {
    const models = Object.entries(overview.unpriced_models || {})
      .map(([m, n]) => "<code>" + escapeHtml(m) + "</code> (" + Number(n) + ")")
      .join(", ");
    const hasUnknown = Object.prototype.hasOwnProperty.call(overview.unpriced_models || {}, "unknown");
    parts.push(
      '<div class="notice"><strong>' + unpriced + " call" + (unpriced === 1 ? "" : "s") +
        " have no price</strong>, so the totals on this page leave them out. Models: " + models + ". " +
        "Add prices with <code>tally.register_model(name, input_cost_per_1m=..., output_cost_per_1m=...)</code> " +
        "or <code>pricing_file=</code>; past calls are re-priced on the next rebuild." +
        (hasUnknown
          ? " For <code>unknown</code>, the model could not be detected: pass <code>model=</code> or return the SDK response object."
          : "") +
        "</div>"
    );
  }
  const estimated = Number(overview.estimated_calls || 0);
  if (estimated > 0) {
    parts.push(
      '<div class="notice info"><strong>' + estimated + " call" + (estimated === 1 ? "" : "s") +
        " (" + formatUsd(overview.estimated_cost_usd) + ") use estimated tokens</strong> " +
        "(about 4 characters per token) because the function returned no token usage. " +
        "Return the SDK response object for exact costs. These costs are marked <strong>~</strong>.</div>"
    );
  }
  document.getElementById("notices").innerHTML = parts.join("");
}

function renderSummaryAndKpis(report) {
  const spend = report.spend_summary || {};
  const overview = report.overview || {};
  const featureTotals = report.feature_totals || [];

  document.getElementById("headerSubtitle").textContent =
    "Updated " + formatLocalTime(report.generated_at) + " (" + timeAgo(report.generated_at) + ")" +
    " • " + Number(overview.total_calls || 0).toLocaleString() + " calls" +
    " • cache hit rate " + Number(overview.cache_hit_rate_pct || 0) + "%";

  // 1. How much money did I spend this week?
  document.getElementById("q1Answer").textContent = formatUsd(spend.this_week_usd);
  document.getElementById("q1Sub").textContent =
    "Last 7 days. Today: " + formatUsd(spend.today_usd) +
    " • 30 days: " + formatUsd(spend.this_month_usd) +
    " • All time: " + formatUsd(spend.all_time_usd) +
    " • Saved by cache (est.): " + formatUsd(spend.saved_by_cache_usd);

  // 2. Which feature costs the most? (all configs of a feature added together)
  const top = featureTotals[0];
  document.getElementById("q2Answer").textContent = top ? top.feature : "N/A";
  document.getElementById("q2Sub").textContent = top
    ? formatUsd(top.total_cost_usd) + " (" + Number(top.share_pct || 0).toFixed(1) + "% of total spend" +
      (top.configs > 1 ? ", " + top.configs + " configs" : "") + ")"
    : "";

  // 3. Which calls fail the most?
  document.getElementById("q3Answer").textContent = overview.most_failing_call || "None (0 failures)";
  document.getElementById("q3Sub").textContent =
    "Failed calls: " + Number(overview.failed_calls || 0) +
    " • Retries: " + Number(overview.total_retries || 0);

  // 4. Which calls are slow?
  document.getElementById("q4Answer").textContent = overview.slowest_call || "N/A";
}

function renderWeeklyChart(report) {
  const container = document.getElementById("weeklyChart");
  const days = report.weekly_daily_spend || [];
  const maxCost = Math.max(...days.map((d) => Number(d.cost_usd || 0)), 0) || 1.0;
  document.getElementById("weeklyDesc").textContent =
    "Daily LLM cost and call volume. Days are in " + (report.timezone || "UTC") +
    ", the time zone of the machine that built this page.";

  container.innerHTML = days
    .map((day) => {
      const cost = Number(day.cost_usd || 0);
      const pct = cost > 0 ? Math.min(100.0, Math.max(4.0, (cost / maxCost) * 100.0)) : 3.0;
      const barClass = cost > 0 ? "bar-fill active" : "bar-fill empty";
      const title =
        day.date + ": " + formatUsd(cost) + " (" + Number(day.calls || 0) + " calls, " +
        Number(day.cached_calls || 0) + " cached, " + Number(day.failed_calls || 0) + " failed)";
      return (
        '<div class="day-col" title="' + escapeHtml(title) + '">' +
        '<div class="day-val">' + formatUsd(cost) + "</div>" +
        '<div class="bar-track"><div class="' + barClass + '" style="height: ' + pct.toFixed(1) + '%"></div></div>' +
        '<div class="day-label">' + escapeHtml(day.label) + "</div>" +
        '<div class="day-sub">' + Number(day.calls || 0) + " calls</div>" +
        "</div>"
      );
    })
    .join("");
}

function renderFeatureCostsTable(featureCosts) {
  const tbody = document.getElementById("featureCostsBody");
  const list = featureCosts || [];
  if (list.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-row">No LLM calls recorded yet.</td></tr>';
    return;
  }

  tbody.innerHTML = list
    .map((fc, idx) => {
      const rankClass = idx === 0 ? "rank-badge top" : "rank-badge";
      const modelsBadges = (fc.models || []).map(modelPill).join(" ");
      const share = Math.min(100.0, Number(fc.share_pct || 0));
      const extras = [];
      if (Number(fc.cached_calls || 0) > 0) extras.push(Number(fc.cached_calls) + " cached");
      if (Number(fc.failed_calls || 0) > 0) extras.push(Number(fc.failed_calls) + " failed");
      if (Number(fc.unpriced_calls || 0) > 0) extras.push(Number(fc.unpriced_calls) + " unpriced");
      return (
        "<tr>" +
        '<td><span class="' + rankClass + '">#' + (idx + 1) + "</span> <strong>" + escapeHtml(fc.feature) + "</strong></td>" +
        "<td><code>" + escapeHtml(fc.config_label) + "</code></td>" +
        "<td>" + modelsBadges + "</td>" +
        costCell(fc.total_cost_usd, { unpriced: Number(fc.unpriced_calls || 0) > 0 && Number(fc.total_cost_usd || 0) === 0 }) +
        costCell(fc.week_cost_usd) +
        '<td><div class="progress-wrap"><div class="progress-bar" style="width: ' + share.toFixed(1) +
        '%"></div><span>' + Number(fc.share_pct || 0).toFixed(1) + "%</span></div></td>" +
        '<td class="num">' + Number(fc.total_calls || 0).toLocaleString() +
        (extras.length ? '<br><span class="muted">' + escapeHtml(extras.join(", ")) + "</span>" : "") + "</td>" +
        '<td class="num">' + Number(fc.input_tokens || 0).toLocaleString() + " / " +
        Number(fc.output_tokens || 0).toLocaleString() + "</td>" +
        "</tr>"
      );
    })
    .join("");
}

function expandable(summaryHtml, bodyHtml) {
  return (
    '<details class="expand"><summary>' + summaryHtml + '</summary><div class="detail-body">' +
    bodyHtml + "</div></details>"
  );
}

function detailBlock(label, text, cls) {
  if (text == null || text === "") return "";
  return (
    '<div><div class="detail-label">' + escapeHtml(label) + '</div><pre class="' + (cls || "") + '">' +
    escapeHtml(text) + "</pre></div>"
  );
}

function renderFailuresTable(failureStats) {
  const tbody = document.getElementById("failureStatsBody");
  const list = (failureStats || []).filter(
    (fs) => Number(fs.failed_calls || 0) > 0 || Number(fs.total_retries || 0) > 0
  );
  if (list.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-row">No failures or retries recorded.</td></tr>';
    return;
  }

  tbody.innerHTML = list
    .map((fs) => {
      const statusCls = Number(fs.failed_calls || 0) > 0 ? "danger" : "warn";
      const err = String(fs.last_error || "");
      const firstLine = err.split("\n")[0];
      const errCell = err
        ? expandable('<span class="err-text">' + escapeHtml(firstLine) + "</span>", detailBlock("Last error", err, "err-text"))
        : '<span class="muted">—</span>';
      return (
        "<tr>" +
        "<td><strong>" + escapeHtml(fs.feature) + "</strong></td>" +
        "<td><code>" + escapeHtml(fs.function_name) + "</code></td>" +
        '<td class="num"><span class="pill ' + statusCls + '">' + Number(fs.failed_calls || 0) + " / " +
        Number(fs.total_calls || 0) + " (" + Number(fs.failure_rate_pct || 0).toFixed(1) + "%)</span></td>" +
        '<td class="num">' + Number(fs.total_retries || 0) + "</td>" +
        "<td>" + errCell + "</td>" +
        "</tr>"
      );
    })
    .join("");
}

function renderLatencyTable(latencyStats) {
  const tbody = document.getElementById("latencyStatsBody");
  const list = latencyStats || [];
  if (list.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-row">No latency records available yet.</td></tr>';
    return;
  }

  tbody.innerHTML = list
    .map((ls) => {
      const avg = Number(ls.avg_latency_ms || 0);
      const slowCls = avg >= 1000 ? "warn" : "ok";
      const modelsBadges = (ls.models || []).map(modelPill).join(" ");
      return (
        "<tr>" +
        "<td><strong>" + escapeHtml(ls.feature) + "</strong></td>" +
        "<td><code>" + escapeHtml(ls.function_name) + "</code></td>" +
        "<td>" + modelsBadges + "</td>" +
        '<td class="num"><span class="pill ' + slowCls + '">' + avg.toFixed(1) + " ms</span></td>" +
        '<td class="num">' + Number(ls.p95_latency_ms || 0).toFixed(1) + " ms</td>" +
        '<td class="num">' + Number(ls.max_latency_ms || 0).toFixed(1) + " ms</td>" +
        '<td class="num">' + Number(ls.live_calls == null ? ls.total_calls : ls.live_calls) + " / " +
        Number(ls.total_calls || 0) + "</td>" +
        "</tr>"
      );
    })
    .join("");
}

function callDetails(rec) {
  const config = rec.config && Object.keys(rec.config).length > 0 ? JSON.stringify(rec.config) : "";
  const meta = rec.metadata && Object.keys(rec.metadata).length > 0 ? JSON.stringify(rec.metadata) : "";
  const err = String(rec.error_message || "");
  const summaryText = err ? err.split("\n")[0] : rec.input_preview || config || meta || "";
  const summary = err
    ? '<span class="err-text">' + escapeHtml(summaryText) + "</span>"
    : '<span class="muted">' + escapeHtml(summaryText || "details") + "</span>";
  const attempts = (rec.attempt_errors || [])
    .map(
      (a) =>
        "#" + a.attempt + (a.delay_before ? " after " + a.delay_before + "s" : "") +
        (a.status_code ? " [HTTP " + a.status_code + "]" : "") + " " + a.error_type + ": " + a.message
    )
    .join("\n");
  const body =
    detailBlock("Error", err, "err-text") +
    detailBlock("Attempts", attempts, "err-text") +
    detailBlock("Input", rec.input_preview) +
    detailBlock("Output", rec.output_preview) +
    detailBlock("Config", config) +
    detailBlock("Metadata", meta) +
    detailBlock("Call id", rec.id);
  return expandable(summary, body);
}

function renderSearchableCalls(report) {
  const recentCalls = report.recent_calls || [];
  const query = (document.getElementById("searchInput").value || "").toLowerCase().trim();
  const statusFilter = document.getElementById("statusFilter").value;
  const tbody = document.getElementById("callsTableBody");

  const filtered = recentCalls.filter((rec) => {
    if (statusFilter === "unpriced") {
      if (!rec.unpriced) return false;
    } else if (statusFilter !== "all" && rec.status !== statusFilter) {
      return false;
    }
    if (!query) return true;
    const haystack = [
      rec.feature,
      rec.function_name,
      rec.model,
      rec.status,
      JSON.stringify(rec.config || {}),
      JSON.stringify(rec.metadata || {}),
      rec.input_preview || "",
      rec.output_preview || "",
      rec.error_message || "",
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });

  if (filtered.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-row">No matching call logs found.</td></tr>';
    return;
  }

  tbody.innerHTML = filtered
    .map((rec) => {
      return (
        "<tr>" +
        '<td class="muted" title="' + escapeHtml(rec.timestamp) + '">' + escapeHtml(formatLocalTime(rec.timestamp)) + "</td>" +
        "<td><strong>" + escapeHtml(rec.feature) + '</strong><br><span class="muted">' + escapeHtml(rec.function_name) + "</span></td>" +
        "<td>" + modelPill(rec.model) + "</td>" +
        '<td><span class="pill ' + escapeHtml(rec.status) + '">' + escapeHtml(rec.status) + "</span></td>" +
        costCell(rec.cost_usd, { unpriced: rec.unpriced, estimated: rec.estimated_tokens }) +
        '<td class="num">' + Number(rec.latency_ms || 0).toFixed(1) + " ms</td>" +
        '<td class="num">' + Number(rec.input_tokens || 0).toLocaleString() + " / " +
        Number(rec.output_tokens || 0).toLocaleString() + "</td>" +
        '<td class="num">' + Number(rec.retries || 0) + "</td>" +
        "<td>" + callDetails(rec) + "</td>" +
        "</tr>"
      );
    })
    .join("");
}

function renderMetadataCosts(metadataCosts) {
  const groups = metadataCosts || {};
  const keys = Object.keys(groups);
  if (keys.length === 0) return;
  document.getElementById("metadataSection").hidden = false;
  document.getElementById("metadataCostsBody").innerHTML = keys
    .map((key) =>
      (groups[key] || [])
        .map(
          (row) =>
            "<tr><td><code>" + escapeHtml(key) + "</code></td><td><strong>" + escapeHtml(row.value) +
            "</strong></td>" + costCell(row.total_cost_usd) +
            '<td class="num">' + Number(row.total_calls || 0).toLocaleString() + "</td></tr>"
        )
        .join("")
    )
    .join("");
}

function initDashboard(report) {
  renderNotices(report);
  renderSummaryAndKpis(report);
  renderWeeklyChart(report);
  renderFeatureCostsTable(report.feature_costs);
  renderMetadataCosts(report.metadata_costs);
  renderFailuresTable(report.failure_stats);
  renderLatencyTable(report.latency_stats);

  const shown = (report.recent_calls || []).length;
  const total = Number(report.recent_calls_total == null ? shown : report.recent_calls_total);
  document.getElementById("callsDesc").innerHTML =
    (shown < total
      ? "Searching the latest <strong>" + shown.toLocaleString() + "</strong> of " + total.toLocaleString() +
        " calls (raise <code>dashboard_recent_calls</code> to keep more). "
      : "Searching all " + total.toLocaleString() + " calls. ") +
    "Click a row's details to see the full input, output, error, and retry attempts. Costs marked " +
    "<strong>~</strong> are estimated from text length; <strong>?</strong> means the model has no price.";
  document.getElementById("timeHeader").textContent = "Time (your clock)";
  document.getElementById("footer").textContent =
    "Built-in prices last reviewed " + (report.pricing_as_of || "unknown") +
    ". Check them against your provider's price list; newer models need register_model() or pricing_file=.";

  const rerenderCalls = () => renderSearchableCalls(report);
  document.getElementById("searchInput").addEventListener("input", rerenderCalls);
  document.getElementById("statusFilter").addEventListener("change", rerenderCalls);
  rerenderCalls();
}

initDashboard(REPORT_DATA);
