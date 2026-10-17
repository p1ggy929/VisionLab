/* ============================================================
   视界智研 —— UI 组件与图表（Plotly）
   ============================================================ */
((global) => {
  "use strict";

  const Fmt = global.VL.Fmt;

  /* ---------------- 图表主题 ---------------- */
  const PALETTE = ["#58a6ff", "#3fb950", "#f0883e", "#d29922", "#bc8cff", "#ff7b72",
                   "#39c5cf", "#a5d6ff", "#7ee787", "#ffa657"];
  const LAYOUT_BASE = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#c9d1d9", size: 12, family: "Microsoft YaHei, sans-serif" },
    margin: { l: 54, r: 18, t: 34, b: 42 },
    xaxis: { gridcolor: "#1f2833", zerolinecolor: "#263040" },
    yaxis: { gridcolor: "#1f2833", zerolinecolor: "#263040" },
    legend: { orientation: "h", y: -0.18, font: { size: 11 } },
    hoverlabel: { bgcolor: "#1d2530", bordercolor: "#263040", font: { color: "#e6edf3" } },
  };
  const CONFIG = { displaylogo: false, responsive: true,
                   modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

  function baseLayout(extra) {
    return Object.assign(JSON.parse(JSON.stringify(LAYOUT_BASE)), extra || {});
  }

  function render(id, traces, layout, config) {
    const el = document.getElementById(id);
    if (!el || !global.Plotly) return;
    global.Plotly.newPlot(el, traces, baseLayout(layout), Object.assign({}, CONFIG, config || {}));
  }

  function empty(id, message) {
    const el = document.getElementById(id);
    if (!el) return;
    if (global.Plotly && el.data) { global.Plotly.purge(el); }
    el.innerHTML = '<div class="empty">' + Fmt.escape(message || "暂无数据") + "</div>";
  }

  /* ---------------- 常用图表 ---------------- */
  const Chart = {
    bar(id, categories, values, name, layout) {
      if (!values || !values.length) return empty(id, "暂无数据");
      render(id, [{ type: "bar", x: categories, y: values, name: name || "数值",
                    marker: { color: PALETTE[0] },
                    text: values.map(v => (v === null ? "" : Number(v).toFixed(3))),
                    textposition: "outside", textfont: { size: 10 } }],
        Object.assign({ margin: { l: 54, r: 18, t: 30, b: 70 } }, layout || {}));
    },
    line(id, categories, series, layout) {
      if (!series || !series.length) return empty(id, "暂无数据");
      const traces = series.map((s, i) => {
        return { type: "scatter", mode: "lines+markers", x: categories, y: s.data,
                 name: s.name, line: { color: PALETTE[i % PALETTE.length], width: 2 },
                 marker: { size: 6 } };
      });
      render(id, traces, layout);
    },
    lineWithBand(id, categories, mean, std, name) {
      if (!mean || !mean.length) return empty(id, "暂无数据");
      const traces = [];
      if (std && std.length) {
        traces.push({ type: "scatter", mode: "lines", x: categories, y: mean.map((v, i) => v + (std[i] || 0)),
                      line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
        traces.push({ type: "scatter", mode: "lines", x: categories, y: mean.map((v, i) => v - (std[i] || 0)),
                      fill: "tonexty", fillcolor: "rgba(88,166,255,.14)", line: { width: 0 },
                      name: "±1 标准差", hoverinfo: "skip" });
      }
      traces.push({ type: "scatter", mode: "lines+markers", x: categories, y: mean,
                    name: name || "平均绝对误差(m)", line: { color: PALETTE[0], width: 2.4 },
                    marker: { size: 7 } });
      render(id, traces, {});
    },
    box(id, groups, layout) {
      if (!groups || !groups.length) return empty(id, "暂无数据");
      const traces = groups.map((g, i) => {
        return { type: "box", y: g.values, name: g.name, boxpoints: "outliers",
                 marker: { color: PALETTE[i % PALETTE.length] } };
      });
      render(id, traces, layout);
    },
    heatmap(id, x, y, z, layout) {
      if (!x || !x.length || !y || !y.length) return empty(id, "暂无数据");
      render(id, [{ type: "heatmap", x: x, y: y, z: z, colorscale: "YlOrRd",
                    colorbar: { title: { text: "MAE(m)", font: { size: 11 } }, thickness: 12 },
                    hovertemplate: "%{y} @ %{x}m<br>MAE %{z:.3f} m<extra></extra>" }],
        Object.assign({ margin: { l: 76, r: 20, t: 20, b: 46 } }, layout || {}));
    },
    radar(id, labels, series, layout) {
      if (!series || !series.length) return empty(id, "暂无数据");
      const traces = series.map((s, i) => {
        return { type: "scatterpolar", r: s.data, theta: labels, fill: "toself",
                 name: s.name, line: { color: PALETTE[i % PALETTE.length] } };
      });
      render(id, traces, Object.assign({
        polar: { bgcolor: "rgba(0,0,0,0)",
                 radialaxis: { gridcolor: "#1f2833", linecolor: "#263040" },
                 angularaxis: { gridcolor: "#1f2833" } },
        margin: { l: 52, r: 52, t: 30, b: 30 },
      }, layout || {}));
    },
    scatter(id, before, after, layout) {
      const traces = [];
      if (before && before.length) {
        traces.push({ type: "scatter", mode: "markers", x: before.map(p => p.measured),
                      y: before.map(p => p.actual), name: "校正前",
                      marker: { color: PALETTE[2], size: 7, opacity: 0.75 } });
      }
      if (after && after.length) {
        traces.push({ type: "scatter", mode: "markers", x: after.map(p => p.corrected),
                      y: after.map(p => p.actual), name: "校正后",
                      marker: { color: PALETTE[1], size: 7, symbol: "triangle-up" } });
      }
      if (!traces.length) return empty(id, "暂无数据");
      const all = traces.reduce((acc, t) => { return acc.concat(t.x, t.y); }, []);
      const max = Math.max.apply(null, all) * 1.05;
      traces.push({ type: "scatter", mode: "lines", x: [0, max], y: [0, max], name: "理想 y=x",
                    line: { color: "#6b7887", dash: "dash", width: 1 } });
      render(id, traces, Object.assign({ xaxis: { title: { text: "测量值 (m)" } },
                                         yaxis: { title: { text: "真值 (m)" } } }, layout || {}));
    },
    pie(id, labels, values) {
      if (!values || !values.length) return empty(id, "暂无数据");
      render(id, [{ type: "pie", labels: labels, values: values, hole: 0.45,
                    marker: { colors: PALETTE },
                    textinfo: "label+percent", textfont: { size: 11 } }],
        { margin: { l: 10, r: 10, t: 20, b: 10 } });
    },
    current(id) { const el = document.getElementById(id); return el ? (el.data || []) : []; },
  };

  /* ---------------- 组件 ---------------- */
  const UI = {
    kpi(label, value, sub, cls) {
      return '<div class="card"><div class="kpi ' + (cls || "") + '">' +
        '<span class="label">' + Fmt.escape(label) + "</span>" +
        '<span class="value">' + (value === null || value === undefined ? "-" : value) + "</span>" +
        (sub ? '<span class="sub">' + Fmt.escape(sub) + "</span>" : "") + "</div></div>";
    },

    table(columns, rows, options) {
      options = options || {};
      if (!rows || !rows.length) {
        return '<div class="empty">' + Fmt.escape(options.empty || "暂无数据") + "</div>";
      }
      const head = columns.map((c) => {
        return '<th class="' + (c.num ? "num" : "") + '">' + Fmt.escape(c.title) + "</th>";
      }).join("");
      const body = rows.map((row, index) => {
        return "<tr>" + columns.map((c) => {
          let value;
          if (typeof c.render === "function") value = c.render(row, index);
          else value = Fmt.escape(row[c.key] === undefined ? "" : row[c.key]);
          return '<td class="' + (c.num ? "num" : "") + '">' + value + "</td>";
        }).join("") + "</tr>";
      }).join("");
      const wrap = options.scroll === false ? "" : '<div class="table-wrap">';
      const wrapEnd = options.scroll === false ? "" : "</div>";
      return wrap + '<table><thead><tr>' + head + "</tr></thead><tbody>" + body +
             "</tbody></table>" + wrapEnd;
    },

    pager(total, page, pageSize, onPage) {
      const pages = Math.max(1, Math.ceil((total || 0) / (pageSize || 20)));
      const id = "pager-" + Math.random().toString(36).slice(2, 8);
      setTimeout(() => {
        const box = document.getElementById(id);
        if (!box) return;
        box.querySelectorAll("button[data-page]").forEach((btn) => {
          btn.onclick = () => { onPage(parseInt(btn.dataset.page, 10)); };
        });
      }, 0);
      return '<div class="inline" id="' + id + '" style="justify-content:space-between;margin-top:10px">' +
        '<span class="muted small">共 ' + (total || 0) + " 条 / " + pages + " 页</span>" +
        '<span class="inline">' +
        '<button class="sm" data-page="' + Math.max(1, page - 1) + '"' + (page <= 1 ? " disabled" : "") + ">上一页</button>" +
        '<span class="small muted">第 ' + page + " 页</span>" +
        '<button class="sm" data-page="' + Math.min(pages, page + 1) + '"' + (page >= pages ? " disabled" : "") + ">下一页</button>" +
        "</span></div>";
    },

    gallery(items, urlFn, captionFn) {
      if (!items || !items.length) return '<div class="empty">暂无样本</div>';
      return '<div class="gallery">' + items.map((item) => {
        const url = urlFn(item);
        const caption = captionFn(item);
        if (!url) return '<figure><figcaption>' + Fmt.escape(caption) + "（无媒体）</figcaption></figure>";
        return '<figure><img loading="lazy" src="' + url + '" alt="">' +
               "<figcaption>" + Fmt.escape(caption) + "</figcaption></figure>";
      }).join("") + "</div>";
    },

    optionList(items, valueKey, labelKey, selected) {
      return (items || []).map((item) => {
        const value = item[valueKey || "key"];
        const label = item[labelKey || "label"];
        return '<option value="' + Fmt.escape(value) + '"' +
               (String(value) === String(selected) ? " selected" : "") + ">" +
               Fmt.escape(label) + "</option>";
      }).join("");
    },

    select(id, items, selected, placeholder) {
      return '<select id="' + id + '">' +
        (placeholder ? '<option value="">' + Fmt.escape(placeholder) + "</option>" : "") +
        UI.optionList(items, "key", "label", selected) + "</select>";
    },

    checkboxGroup(name, items, checked) {
      const set = new Set(checked || []);
      return '<div class="inline">' + (items || []).map((item) => {
        const value = item.key !== undefined ? item.key : item;
        const label = item.label !== undefined ? item.label : item;
        return '<label class="inline small"><input type="checkbox" name="' + name +
          '" value="' + Fmt.escape(value) + '"' + (set.has(value) ? " checked" : "") + "> " +
          Fmt.escape(label) + "</label>";
      }).join("") + "</div>";
    },

    checkedValues(name) {
      return Array.prototype.slice.call(document.querySelectorAll('input[name="' + name + '"]:checked'))
        .map((el) => { return el.value; });
    },

    statusLine(items) {
      return '<div class="status-line">' + items.map((item) => {
        return "<span>" + Fmt.escape(item[0]) + " <b>" + (item[1] === null || item[1] === undefined ? "-" : item[1]) + "</b></span>";
      }).join("") + "</div>";
    },

    defList(pairs) {
      return '<table class="kv">' + pairs.map((pair) => {
        return "<tr><th>" + Fmt.escape(pair[0]) + "</th><td>" +
          (typeof pair[1] === "string" && pair[1].indexOf("<") === 0 ? pair[1] : Fmt.escape(pair[1])) +
          "</td></tr>";
      }).join("") + "</table>";
    },

    loading(target, text) {
      const el = typeof target === "string" ? document.getElementById(target) : target;
      if (el) el.innerHTML = '<div class="loading">' + Fmt.escape(text || "加载中…") + "</div>";
    },
  };

  global.VL.UI = UI;
  global.VL.Chart = Chart;
  global.VL.PALETTE = PALETTE;
})(window);
