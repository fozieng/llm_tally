function formatUsd(val) {
  const num = Number(val || 0);
  if (num === 0) return "$0.00";
  if (num < 0.01) return "$" + num.toFixed(6);
  return "$" + num.toLocaleString("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
}

function escapeHtml(str) {
  return String(str == null ? "" : str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderSummaryAndKpis(report) {
  const spend = report.spend_summary || {};
  const overview = report.overview || {};
  const featureCosts = report.feature_costs || [];

  document.getElementById("headerSubtitle").innerHTML =
    "Updated: " +
    escapeHtml(report.generated_at || "") +
    " &bull; Total Calls: " +
    Number(overview.total_calls || 0) +
    " &bull; Cache Hit Rate: " +
    Number(overview.cache_hit_rate_pct || 0) +
    "%" +
    (Number(overview.estimated_calls || 0) > 0
      ? " &bull; ~" +
        Number(overview.estimated_calls) +
        " calls (" +
        formatUsd(overview.estimated_cost_usd) +
        ") use estimated tokens"
      : "");

  // 1. How much money did I spend this week?
  document.getElementById("q1Answer").textContent = formatUsd(spend.this_week_usd);
  document.getElementById("q1Sub").innerHTML =
    "Today: " +
    formatUsd(spend.today_usd) +
    " &bull; All-time: " +
    formatUsd(spend.all_time_usd) +
    " &bull; Saved by cache: " +
    formatUsd(spend.saved_by_cache_usd);

  // 2. Which feature costs the most?
  document.getElementById("q2Answer").textContent = overview.top_cost_feature || "N/A";
  const topCost = featureCosts.length > 0 ? formatUsd(featureCosts[0].total_cost_usd) : "$0.00";
  const topShare = featureCosts.length > 0 ? featureCosts[0].share_pct : 0;
  document.getElementById("q2Sub").textContent = topCost + " (" + topShare + "% of total spend)";

  // 3. Which calls fail the most?
  document.getElementById("q3Answer").textContent = overview.most_failing_call || "None (0 failures)";
  document.getElementById("q3Sub").innerHTML =
    "Total Failed Calls: " +
    Number(overview.failed_calls || 0) +
    " &bull; Total Retries Triggered: " +
    Number(overview.total_retries || 0);

  // 4. Which calls are slow?
  document.getElementById("q4Answer").textContent = overview.slowest_call || "N/A";
}

function renderWeeklyChart(weeklyDays) {
  const container = document.getElementById("weeklyChart");
  const days = weeklyDays || [];
  const maxCost = Math.max(...days.map((d) => Number(d.cost_usd || 0)), 0) || 1.0;

  container.innerHTML = days
    .map((day) => {
      const cost = Number(day.cost_usd || 0);
      const pct = cost > 0 ? Math.min(100.0, Math.max(4.0, (cost / maxCost) * 100.0)) : 3.0;
      const barClass = cost > 0 ? "bar-fill active" : "bar-fill empty";
      const title = escapeHtml(day.date) + ": " + formatUsd(cost) + " (" + day.calls + " calls)";
      return (
        '<div class="day-col" title="' +
        title +
        '">' +
        '<div class="day-val">' +
        formatUsd(cost) +
        "</div>" +
        '<div class="bar-track">' +
        '<div class="' +
        barClass +
        '" style="height: ' +
        pct.toFixed(1) +
        '%"></div>' +
        "</div>" +
        '<div class="day-label">' +
        escapeHtml(day.label) +
        "</div>" +
        '<div class="day-sub">' +
        Number(day.calls || 0) +
        " calls</div>" +
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
      const modelsBadges = (fc.models || [])
        .map((m) => '<span class="pill model">' + escapeHtml(m) + "</span>")
        .join(" ");
      const share = Math.min(100.0, Number(fc.share_pct || 0));
      return (
        "<tr>" +
        '<td><span class="' +
        rankClass +
        '">#' +
        (idx + 1) +
        "</span> <strong>" +
        escapeHtml(fc.feature) +
        "</strong></td>" +
        "<td><code>" +
        escapeHtml(fc.config_label) +
        "</code></td>" +
        "<td>" +
        modelsBadges +
        "</td>" +
        '<td class="num highlight">' +
        formatUsd(fc.total_cost_usd) +
        "</td>" +
        '<td class="num">' +
        formatUsd(fc.week_cost_usd) +
        "</td>" +
        "<td>" +
        '<div class="progress-wrap">' +
        '<div class="progress-bar" style="width: ' +
        share.toFixed(1) +
        '%"></div>' +
        "<span>" +
        Number(fc.share_pct || 0).toFixed(1) +
        "%</span>" +
        "</div>" +
        "</td>" +
        '<td class="num">' +
        Number(fc.total_calls || 0) +
        ' <span class="muted">(' +
        Number(fc.cached_calls || 0) +
        " cached)</span></td>" +
        '<td class="num">' +
        Number(fc.input_tokens || 0).toLocaleString() +
        " / " +
        Number(fc.output_tokens || 0).toLocaleString() +
        "</td>" +
        "</tr>"
      );
    })
    .join("");
}

function renderFailuresTable(failureStats) {
  const tbody = document.getElementById("failureStatsBody");
  const list = failureStats || [];
  if (list.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-row">No call failures recorded.</td></tr>';
    return;
  }

  tbody.innerHTML = list
    .map((fs) => {
      const statusCls = Number(fs.failed_calls || 0) > 0 ? "danger" : "ok";
      return (
        "<tr>" +
        "<td><strong>" +
        escapeHtml(fs.feature) +
        "</strong></td>" +
        "<td><code>" +
        escapeHtml(fs.function_name) +
        "</code></td>" +
        '<td class="num"><span class="pill ' +
        statusCls +
        '">' +
        Number(fs.failed_calls || 0) +
        " / " +
        Number(fs.total_calls || 0) +
        " (" +
        Number(fs.failure_rate_pct || 0).toFixed(1) +
        "%)</span></td>" +
        '<td class="num">' +
        Number(fs.total_retries || 0) +
        "</td>" +
        '<td class="err-cell">' +
        escapeHtml(fs.last_error || "—") +
        "</td>" +
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
      const modelsBadges = (ls.models || [])
        .map((m) => '<span class="pill model">' + escapeHtml(m) + "</span>")
        .join(" ");
      return (
        "<tr>" +
        "<td><strong>" +
        escapeHtml(ls.feature) +
        "</strong></td>" +
        "<td><code>" +
        escapeHtml(ls.function_name) +
        "</code></td>" +
        "<td>" +
        modelsBadges +
        "</td>" +
        '<td class="num"><span class="pill ' +
        slowCls +
        '">' +
        avg.toFixed(1) +
        " ms</span></td>" +
        '<td class="num">' +
        Number(ls.p95_latency_ms || 0).toFixed(1) +
        " ms</td>" +
        '<td class="num">' +
        Number(ls.max_latency_ms || 0).toFixed(1) +
        " ms</td>" +
        '<td class="num">' +
        Number(ls.total_calls || 0) +
        "</td>" +
        "</tr>"
      );
    })
    .join("");
}

function renderSearchableCalls(recentCalls) {
  const query = (document.getElementById("searchInput").value || "").toLowerCase().trim();
  const statusFilter = document.getElementById("statusFilter").value;
  const tbody = document.getElementById("callsTableBody");
  const calls = recentCalls || [];

  const filtered = calls.filter((rec) => {
    if (statusFilter !== "all" && rec.status !== statusFilter) return false;
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
      const meta = rec.metadata && Object.keys(rec.metadata).length > 0
        ? " <code>" + escapeHtml(JSON.stringify(rec.metadata)) + "</code>"
        : "";
      const detail = rec.error_message
        ? '<span class="err-cell">' + escapeHtml(rec.error_message) + "</span>" + meta
        : "<code>" +
          escapeHtml(JSON.stringify(rec.config || {})) +
          "</code>" +
          meta +
          ' <span class="muted">' +
          escapeHtml(rec.input_preview || "") +
          "</span>";
      const costLabel = (rec.estimated_tokens ? "~" : "") + formatUsd(rec.cost_usd);
      return (
        "<tr>" +
        '<td class="muted">' +
        escapeHtml(String(rec.timestamp || "").replace("T", " ").slice(0, 19)) +
        "</td>" +
        "<td><strong>" +
        escapeHtml(rec.feature) +
        '</strong><br><span class="muted">' +
        escapeHtml(rec.function_name) +
        "</span></td>" +
        '<td><span class="pill model">' +
        escapeHtml(rec.model) +
        "</span></td>" +
        '<td><span class="pill ' +
        escapeHtml(rec.status) +
        '">' +
        escapeHtml(rec.status) +
        "</span></td>" +
        '<td class="num highlight">' +
        costLabel +
        "</td>" +
        '<td class="num">' +
        Number(rec.latency_ms || 0).toFixed(1) +
        " ms</td>" +
        '<td class="num">' +
        Number(rec.input_tokens || 0) +
        " / " +
        Number(rec.output_tokens || 0) +
        "</td>" +
        '<td class="num">' +
        Number(rec.retries || 0) +
        "</td>" +
        "<td>" +
        detail +
        "</td>" +
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
            "<tr><td><code>" +
            escapeHtml(key) +
            "</code></td><td><strong>" +
            escapeHtml(row.value) +
            '</strong></td><td class="num highlight">' +
            formatUsd(row.total_cost_usd) +
            '</td><td class="num">' +
            Number(row.total_calls || 0) +
            "</td></tr>"
        )
        .join("")
    )
    .join("");
}

function initDashboard(report) {
  renderSummaryAndKpis(report);
  renderWeeklyChart(report.weekly_daily_spend);
  renderFeatureCostsTable(report.feature_costs);
  renderMetadataCosts(report.metadata_costs);
  renderFailuresTable(report.failure_stats);
  renderLatencyTable(report.latency_stats);

  const rerenderCalls = () => renderSearchableCalls(report.recent_calls);
  document.getElementById("searchInput").addEventListener("input", rerenderCalls);
  document.getElementById("statusFilter").addEventListener("change", rerenderCalls);
  rerenderCalls();
}

initDashboard(REPORT_DATA);

