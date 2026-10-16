/* ============================================================
   视界智研 —— 页面逻辑（18 个页面）
   ============================================================ */
(function (global) {
  "use strict";

  const VL = global.VL;
  const API = VL.API, Fmt = VL.Fmt, Toast = VL.Toast, Modal = VL.Modal,
        UI = VL.UI, Chart = VL.Chart, handleError = VL.handleError,
        poll = VL.poll, connectSocket = VL.connectSocket;

  const Pages = {};
  let pollers = [];

  function addPoller(p) { if (p) pollers.push(p); }
  function el(id) { return document.getElementById(id); }
  function html(id, content) { const e = el(id); if (e) e.innerHTML = content; }
  function value(id) { const e = el(id); return e ? e.value : ""; }
  function num(id, fallback) { const v = parseFloat(value(id)); return isNaN(v) ? fallback : v; }
  function checked(id) { const e = el(id); return e ? e.checked : false; }

  /* ================================================================
     启动
     ================================================================ */
  VL.boot = async function () {
    try {
      await API.me();
    } catch (e) {
      location.href = "/login";
      return;
    }
    bindChrome();
    try { await API.bootstrap(); } catch (e) { /* 忽略 */ }
    fillProjectSelector();
    bindModeSwitch();
    startStatusPollers();
    const page = window.VISIONLAB.page;
    const init = Pages[page];
    if (typeof init === "function") {
      try {
        await init();
      } catch (error) {
        handleError(error, "页面初始化失败");
      }
    }
  };

  function bindChrome() {
    const toggle = el("user-toggle"), dropdown = el("user-dropdown");
    if (toggle && dropdown) toggle.onclick = function () { dropdown.classList.toggle("open"); };
    document.addEventListener("click", function (event) {
      if (dropdown && !dropdown.contains(event.target) && event.target !== toggle) {
        dropdown.classList.remove("open");
      }
    });
    const logout = el("logout-link");
    if (logout) logout.onclick = async function (e) {
      e.preventDefault();
      try { await API.post("/api/v1/auth/logout"); } catch (err) { /* 忽略 */ }
      API.setToken("", null);
      location.href = "/login";
    };
    const changePwd = el("change-password-link");
    if (changePwd) changePwd.onclick = function (e) {
      e.preventDefault();
      Modal.open("修改密码",
        '<label class="field"><span>原密码</span><input id="pwd-old" type="password"></label>' +
        '<label class="field"><span>新密码（至少 6 位）</span><input id="pwd-new" type="password"></label>',
        '<button data-close="1">取消</button><button class="primary" id="pwd-save">保存</button>');
      el("pwd-save").onclick = async function () {
        try {
          await API.post("/api/v1/auth/change-password",
            { old_password: value("pwd-old"), new_password: value("pwd-new") });
          Toast.ok("密码已修改");
          Modal.close();
        } catch (error) { handleError(error); }
      };
    };
  }

  function fillProjectSelector() {
    const select = el("global-project");
    if (!select) return;
    select.innerHTML = API.state.projects.map(function (p) {
      return '<option value="' + p.id + '"' +
        (p.id === API.projectId() ? " selected" : "") + ">" + Fmt.escape(p.name) + "</option>";
    }).join("") || '<option value="">（暂无项目）</option>';
    select.onchange = function () { API.setProject(select.value); location.reload(); };
  }

  function bindModeSwitch() {
    const box = el("global-mode");
    if (!box) return;
    API.get("/api/v1/pipeline/config").then(function (d) {
      const mode = ((d.data || {}).config || {}).mode;
      box.querySelectorAll("button").forEach(function (b) {
        b.classList.toggle("active", b.dataset.mode === mode);
      });
    }).catch(function () {});
    box.querySelectorAll("button").forEach(function (btn) {
      btn.onclick = async function () {
        try {
          await API.post("/api/v1/pipeline/config", { mode: btn.dataset.mode });
          box.querySelectorAll("button").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          Toast.ok("已切换到：" + btn.textContent);
          location.href = btn.dataset.mode === "segmentation"
            ? "/experiments/segmentation"
            : (btn.dataset.mode === "detection" ? "/experiments/detection" : "/experiments/ranging");
        } catch (error) { handleError(error, "切换算法路线失败"); }
      };
    });
  }

  function startStatusPollers() {
    addPoller(poll(async function () {
      const d = await API.get("/api/v1/camera/status");
      const info = d.data || {};
      const badge = el("camera-status");
      if (badge) {
        const label = info.running
          ? (info.backend === "realsense" ? "RealSense 在线" : "合成场景")
          : "相机未启动";
        badge.className = "pill " + (info.running ? "on" : "off");
        const span = badge.querySelector("span:last-child");
        if (span) span.textContent = label;
        badge.title = "后端 " + (info.backend || "-") + " · " + (info.width || "-") + "x" +
          (info.height || "-") + " · " + (info.fps_actual || 0) + " fps";
      }
    }, 5000));

    addPoller(poll(async function () {
      const d = await API.get("/api/v1/tasks");
      const summary = (d.data || {}).summary || {};
      const badge = el("task-badge");
      if (badge) {
        const b = badge.querySelector("b");
        if (b) b.textContent = (summary.queued || 0) + (summary.running || 0);
        badge.title = "排队 " + (summary.queued || 0) + " / 运行 " + (summary.running || 0) +
                      " / 完成 " + (summary.done || 0) + " / 失败 " + (summary.failed || 0);
      }
    }, 6000));

    addPoller(poll(async function () {
      const d = await API.get("/api/v1/system/info");
      const info = d.data || {};
      html("env-line", "Python " + Fmt.escape(info.python || "-") + "<br>" +
        (info.cuda_available ? "GPU: " + Fmt.escape(info.gpu || "CUDA") : "GPU: 未启用(CPU)"));
    }, 30000));
  }

  /* ================================================================
     工作台
     ================================================================ */
  Pages.dashboard = async function () {
    html("page-actions",
      '<button class="primary" onclick="location.href=\'/experiments/ranging\'">开始测距实验</button>' +
      '<button onclick="location.href=\'/camera/capture\'">数据采集</button>' +
      '<button onclick="location.href=\'/analysis\'">误差分析</button>');
    await loadDashboard();
  };

  async function loadDashboard() {
    const d = await API.get("/api/v1/dashboard/overview");
    const data = d.data || {};
    const kpi = data.kpi || {};
    const tasks = kpi.tasks || {};
    html("kpi-row",
      UI.kpi("项目", Fmt.int(kpi.projects), "实验容器", "accent") +
      UI.kpi("实验", Fmt.int(kpi.experiments), "含历史迁移实验", "") +
      UI.kpi("模型", Fmt.int(kpi.models), "权重卡片", "") +
      UI.kpi("图像样本", Fmt.int(kpi.images), "数据集版本合计", "seg") +
      UI.kpi("测距记录", Fmt.int(kpi.ranging_records), "落库条数", "range") +
      UI.kpi("平均绝对误差", Fmt.m(kpi.mean_error_m), "全量口径", "detect") +
      UI.kpi("缺检率", Fmt.pct(kpi.miss_rate), "检测结果统计", "danger") +
      UI.kpi("任务队列", (tasks.queued || 0) + " / " + (tasks.running || 0), "排队 / 运行", ""));

    const distance = data.by_distance || [];
    if (distance.length) {
      Chart.lineWithBand("chart-distance",
        distance.map(x => (x.key === null ? "-" : Number(x.key).toFixed(0) + "m")),
        distance.map(x => x.mae_m), distance.map(x => x.std_error_m || 0), "平均绝对误差(m)");
    } else { Chart.empty("chart-distance", "暂无带真值的测距数据"); }

    const categories = (data.by_category || []).slice(0, 11);
    if (categories.length) {
      Chart.bar("chart-category", categories.map(x => x.key || "-"), categories.map(x => x.mae_m), "MAE(m)");
    } else { Chart.empty("chart-category", "暂无按类别分组数据"); }

    html("table-leaderboard", UI.table([
      { title: "#", render: (r, i) => i + 1 },
      { title: "模型", key: "key" },
      { title: "任务", render: r => Fmt.taskTag(r.task_type || "detection") },
      { title: "样本", key: "count", num: true },
      { title: "MAE(m)", num: true, render: r => Fmt.num(r.mae_m, 3) },
      { title: "相对误差", num: true, render: r => Fmt.pct(r.mre) },
    ], (data.leaderboard || []).slice(0, 10), { empty: "暂无模型评测数据" }));

    html("gallery-outliers", UI.gallery(data.outliers || [],
      item => item.media_uri ? "/api/v1/media/thumb?uri=" + encodeURIComponent(item.media_uri) : "",
      item => (item.category || "-") + " " + Fmt.m(item.depth_m) + " / 误差 " + Fmt.m(item.abs_error_m)));

    const experiments = await API.get("/api/v1/experiments", { page_size: 8 });
    html("table-recent", UI.table([
      { title: "实验", render: r => '<a href="/experiments/' + r.id + '">' + Fmt.escape(r.name) + "</a>" },
      { title: "任务", render: r => Fmt.taskTag(r.task_type) },
      { title: "类型", key: "kind" },
      { title: "状态", render: r => Fmt.statusTag(r.status) },
      { title: "创建", render: r => Fmt.time(r.created_at) },
    ], experiments.data || [], { empty: "暂无实验" }));
  }

  /* ================================================================
     项目管理
     ================================================================ */
  Pages.projects = async function () {
    html("page-actions", '<button class="primary" id="btn-new-project">新建项目</button>');
    el("btn-new-project").onclick = newProjectModal;
    await loadProjects();
  };

  async function loadProjects() {
    const d = await API.get("/api/v1/projects", { page_size: 100 });
    const rows = d.data || [];
    html("project-list", rows.length ? '<div class="grid c3">' + rows.map(function (p) {
      const stats = p.stats || {};
      return '<div class="card">' +
        '<header><h3><a href="/projects/' + p.id + '">' + Fmt.escape(p.name) + "</a></h3>" +
        Fmt.statusTag(p.status) + "</header>" +
        '<p class="muted small">' + Fmt.escape(p.description || "（无描述）") + "</p>" +
        '<div class="status-line small">' +
        "<span>实验 <b>" + (stats.experiments || 0) + "</b></span>" +
        "<span>数据集 <b>" + (stats.datasets || 0) + "</b></span>" +
        "<span>模型 <b>" + (stats.models || 0) + "</b></span>" +
        "<span>测距记录 <b>" + (stats.ranging_records || 0) + "</b></span></div>" +
        '<div class="inline" style="margin-top:10px">' +
        '<button class="sm" onclick="location.href=\'/experiments?project_id=' + p.id + '\'">查看实验</button>' +
        '<button class="sm" data-del="' + p.id + '">归档</button></div></div>';
    }).join("") + "</div>" : '<div class="empty">还没有项目，点击右上角「新建项目」开始</div>');
    document.querySelectorAll("[data-del]").forEach(function (btn) {
      btn.onclick = async function () {
        try {
          await API.post("/api/v1/projects/" + btn.dataset.del + "/archive");
          Toast.ok("项目已归档");
          loadProjects();
        } catch (error) { handleError(error); }
      };
    });
  }

  function newProjectModal() {
    Modal.open("新建项目",
      '<label class="field"><span>项目名称 *</span><input id="np-name" placeholder="如：校园道路视觉感知"></label>' +
      '<label class="field"><span>项目简介</span><textarea id="np-desc" rows="2"></textarea></label>' +
      '<div class="form-row">' +
      '<label class="field"><span>场景</span>' + UI.select("np-scene", API.dict().scenes || [], "campus_road") + "</label>" +
      '<label class="field"><span>应用目标</span>' + UI.select("np-target", API.dict().app_targets || [], "sweeper") + "</label>" +
      "</div>",
      '<button data-close="1">取消</button><button class="primary" id="np-save">创建</button>');
    el("np-save").onclick = async function () {
      try {
        await API.post("/api/v1/projects", {
          name: value("np-name"), description: value("np-desc"),
          scene: value("np-scene"), app_target: value("np-target"),
        });
        Toast.ok("项目已创建");
        Modal.close();
        await API.bootstrap();
        location.reload();
      } catch (error) { handleError(error); }
    };
  }

  /* ================================================================
     项目详情
     ================================================================ */
  Pages.project_detail = async function () {
    const id = location.pathname.split("/").pop();
    const d = await API.get("/api/v1/projects/" + id);
    const p = d.data || {};
    html("project-head",
      "<h2>" + Fmt.escape(p.name) + "</h2><p class=\"muted\">" + Fmt.escape(p.description || "") + "</p>" +
      UI.statusLine([["场景", p.scene], ["应用目标", p.app_target], ["状态", Fmt.status(p.status)],
                     ["创建时间", Fmt.time(p.created_at)]]));

    const experiments = await API.get("/api/v1/experiments", { project_id: id, page_size: 50 });
    html("project-experiments", UI.table([
      { title: "实验", render: r => '<a href="/experiments/' + r.id + '">' + Fmt.escape(r.name) + "</a>" },
      { title: "任务", render: r => Fmt.taskTag(r.task_type) },
      { title: "状态", render: r => Fmt.statusTag(r.status) },
      { title: "创建", render: r => Fmt.time(r.created_at) },
    ], experiments.data || [], { empty: "暂无实验" }));

    const datasets = await API.get("/api/v1/datasets", { project_id: id, page_size: 50 });
    html("project-datasets", UI.table([
      { title: "数据集", render: r => '<a href="/datasets/' + r.id + '">' + Fmt.escape(r.name) + "</a>" },
      { title: "任务", render: r => Fmt.taskTag(r.task_type) },
      { title: "版本", key: "version_count", num: true },
      { title: "图像", key: "image_count", num: true },
    ], datasets.data || [], { empty: "暂无数据集" }));
  };

  /* ================================================================
     实验管理
     ================================================================ */
  let experimentPage = 1;
  Pages.experiments = async function () {
    const params = new URLSearchParams(location.search);
    const projectId = params.get("project_id") || API.projectId();
    html("filter-bar",
      '<div class="form-row">' +
      '<label class="field"><span>项目</span>' + UI.select("f-project",
        API.state.projects.map(p => ({ key: p.id, label: p.name })), projectId, "全部项目") + "</label>" +
      '<label class="field"><span>任务类型</span><select id="f-task"><option value="">全部</option>' +
      Object.keys(API.dict().task_labels || {}).map(k =>
        '<option value="' + k + '">' + API.dict().task_labels[k] + "</option>").join("") + "</select></label>" +
      '<label class="field"><span>状态</span><select id="f-status"><option value="">全部</option>' +
      Object.keys(API.dict().status_labels || {}).map(k =>
        '<option value="' + k + '">' + API.dict().status_labels[k] + "</option>").join("") + "</select></label>" +
      '<label class="field"><span>&nbsp;</span><button class="primary" id="f-apply">查询</button></label>' +
      "</div>");
    el("f-apply").onclick = function () { experimentPage = 1; loadExperiments(); };
    html("page-actions", '<button id="btn-compare">对比选中实验</button>' +
      '<button onclick="location.href=\'/training\'">发起训练</button>');

    el("btn-compare").onclick = async function () {
      const ids = UI.checkedValues("exp-check");
      if (ids.length < 2) return Toast.warn("请勾选至少 2 个实验");
      const d = await API.get("/api/v1/experiments/compare", { ids: ids.join(",") });
      Modal.open("实验对比", UI.table([
        { title: "实验", key: "name" },
        { title: "任务", render: r => Fmt.taskTag(r.task_type) },
        { title: "状态", render: r => Fmt.statusTag(r.status) },
        { title: "模型", key: "model_name" },
        { title: "指标", render: r => Fmt.escape(JSON.stringify(r.metrics || {})) },
      ], (d.data || {}).items || [], { scroll: false }));
    };
    await loadExperiments();
  };

  async function loadExperiments() {
    UI.loading("experiment-table");
    const d = await API.get("/api/v1/experiments", {
      page: experimentPage, page_size: 20,
      project_id: value("f-project"), task_type: value("f-task"), status: value("f-status"),
    });
    const rows = d.data || [];
    html("experiment-table", UI.table([
      { title: '<input type="checkbox" id="exp-all">', render: r =>
        '<input type="checkbox" name="exp-check" value="' + r.id + '">' },
      { title: "实验", render: r => '<a href="/experiments/' + r.id + '">' + Fmt.escape(r.name) + "</a>" },
      { title: "任务", render: r => Fmt.taskTag(r.task_type) },
      { title: "类型", key: "kind" },
      { title: "状态", render: r => Fmt.statusTag(r.status) },
      { title: "指标", render: r => {
        const m = r.metrics || {};
        const parts = [];
        if (m.mae_m !== undefined && m.mae_m !== null) parts.push("MAE " + Fmt.m(m.mae_m));
        if (m.miss_rate !== undefined && m.miss_rate !== null) parts.push("缺检 " + Fmt.pct(m.miss_rate));
        if (m.ranging_records) parts.push(m.ranging_records + " 条");
        return Fmt.escape(parts.join(" / ") || "-");
      } },
      { title: "创建", render: r => Fmt.time(r.created_at) },
      { title: "操作", render: r => '<button class="sm" data-clone="' + r.id + '">克隆</button> ' +
        '<button class="sm danger" data-del="' + r.id + '">删除</button>' },
    ], rows, { empty: "暂无实验" }) + UI.pager(d.total, d.page, d.page_size, function (p) {
      experimentPage = p; loadExperiments();
    }));
    const all = el("exp-all");
    if (all) all.onclick = function () {
      document.querySelectorAll('input[name="exp-check"]').forEach(c => { c.checked = all.checked; });
    };
    document.querySelectorAll("[data-clone]").forEach(function (btn) {
      btn.onclick = async function () {
        try { await API.post("/api/v1/experiments/" + btn.dataset.clone + "/clone", {}); Toast.ok("已克隆"); loadExperiments(); }
        catch (error) { handleError(error); }
      };
    });
    document.querySelectorAll("[data-del]").forEach(function (btn) {
      btn.onclick = async function () {
        if (!confirm("确认删除该实验？")) return;
        try { await API.del("/api/v1/experiments/" + btn.dataset.del); Toast.ok("已删除"); loadExperiments(); }
        catch (error) { handleError(error); }
      };
    });
  }

  /* ================================================================
     实验详情
     ================================================================ */
  Pages.experiment_detail = async function () {
    const id = location.pathname.split("/").pop();
    const d = await API.get("/api/v1/experiments/" + id);
    const exp = d.data || {};
    html("exp-head", "<h2>" + Fmt.escape(exp.name) + "</h2>" +
      UI.statusLine([["任务", (API.dict().task_labels || {})[exp.task_type]],
                     ["类型", exp.kind], ["状态", Fmt.status(exp.status)],
                     ["创建", Fmt.time(exp.created_at)], ["完成", Fmt.time(exp.finished_at)]]));
    html("page-actions",
      '<button id="btn-manifest">生成复现清单</button><button id="btn-verify">验证复现</button>' +
      '<button id="btn-report" class="primary">导出报告</button><button id="btn-clone">克隆</button>');

    el("btn-manifest").onclick = async function () {
      try {
        const r = await API.post("/api/v1/experiments/" + id + "/manifest", {});
        Modal.open("复现清单 run_manifest.json",
          "<pre class=\"mono\" style=\"max-height:60vh;overflow:auto\">" +
          Fmt.escape(JSON.stringify(r.data, null, 2)) + "</pre>");
      } catch (error) { handleError(error); }
    };
    el("btn-verify").onclick = async function () {
      try {
        const r = await API.post("/api/v1/experiments/" + id + "/reproduce", {});
        const data = r.data || {};
        Modal.open("复现验证结果",
          (data.ok ? '<p class="tag ok">全部一致</p>' : '<p class="tag err">发现差异</p>') +
          UI.table([{ title: "项目", key: "item" }, { title: "记录值", key: "recorded" },
                    { title: "当前值", key: "current" }], data.differences || [],
                   { empty: "无差异", scroll: false }));
      } catch (error) { handleError(error); }
    };
    el("btn-report").onclick = function () { location.href = "/reports?experiment_id=" + id; };
    el("btn-clone").onclick = async function () {
      try { await API.post("/api/v1/experiments/" + id + "/clone", {}); Toast.ok("已克隆"); location.reload(); }
      catch (error) { handleError(error); }
    };

    html("exp-config", "<pre class=\"mono\" style=\"max-height:320px;overflow:auto\">" +
      Fmt.escape(JSON.stringify(exp.config_snapshot || {}, null, 2)) + "</pre>");
    html("exp-metrics", "<pre class=\"mono\">" + Fmt.escape(JSON.stringify(exp.metrics || {}, null, 2)) + "</pre>");
    html("exp-manifest", exp.run_manifest && Object.keys(exp.run_manifest).length
      ? "<pre class=\"mono\" style=\"max-height:360px;overflow:auto\">" +
        Fmt.escape(JSON.stringify(exp.run_manifest, null, 2)) + "</pre>"
      : '<div class="empty">尚未生成复现清单，点击右上角「生成复现清单」</div>');

    const records = await API.get("/api/v1/ranging/records",
      { experiment_id: id, page_size: 100, has_actual: 1 });
    html("exp-records", UI.table(recordColumns(), records.data || [], { empty: "暂无测距记录" }));

    const stats = await API.get("/api/v1/analysis/summary", { experiment_id: id });
    const s = stats.data || {};
    html("exp-stats", UI.statusLine([["样本", s.sample_count], ["有效", s.valid_count],
      ["MAE", Fmt.m(s.mae_m)], ["RMSE", Fmt.m(s.rmse_m)], ["中位", Fmt.m(s.median_error_m)],
      ["最大", Fmt.m(s.max_error_m)], ["命中率", Fmt.pct(s.hit_rate)]]));
    const dist = s.by_distance || [];
    if (dist.length) {
      Chart.lineWithBand("exp-chart", dist.map(x => Number(x.key).toFixed(0) + "m"),
        dist.map(x => x.mae_m), dist.map(x => x.std_error_m || 0));
    } else {
      Chart.empty("exp-chart", "该实验暂无带真值的测距记录");
    }
  };

  function recordColumns() {
    return [
      { title: "时间", render: r => Fmt.time(r.record_time || r.created_at) },
      { title: "任务", render: r => Fmt.taskTag(r.task_type) },
      { title: "目标", render: r => Fmt.escape(r.category || r.target_kind) },
      { title: "指标", key: "metric_kind" },
      { title: "测量(m)", num: true, render: r => Fmt.num(r.depth_m, 3) },
      { title: "横向(m)", num: true, render: r => Fmt.num(r.lateral_m, 3) },
      { title: "真值(m)", num: true, render: r => Fmt.num(r.distance_actual_m, 3) },
      { title: "误差(m)", num: true, render: r => Fmt.num(r.abs_error_m, 3) },
      { title: "相对", num: true, render: r => Fmt.pct(r.rel_error) },
      { title: "有效点", num: true, render: r => (r.valid_points || 0) + "/" + (r.total_points || 0) },
      { title: "校正", render: r => Fmt.escape(r.calibration_applied || "-") },
    ];
  }

  /* ================================================================
     实时实验页（检测 / 分割 / 测距 共用）
     ================================================================ */
  function liveLayout() {
    return '<div class="card"><div class="inline" style="justify-content:space-between">' +
      '<div class="inline">' +
      '<span class="pill"><i class="dot"></i><span id="live-source">-</span></span>' +
      '<span class="pill">帧 <b id="live-frame">0</b></span>' +
      '<span class="pill">延迟 <b id="live-latency">-</b></span>' +
      '<span class="pill">检出 <b id="live-count">0</b></span>' +
      '<span class="pill">走廊 <b id="live-corridor">-</b></span>' +
      "</div><div class=\"inline\">" +
      '<button id="live-start" class="primary">开始实验</button>' +
      '<button id="live-pause">暂停/恢复</button>' +
      '<button id="live-save">落库当前帧</button>' +
      '<button id="live-bulk">全部落库</button>' +
      '<button id="live-stop">结束实验</button>' +
      "</div></div></div>" +
      '<div class="grid c2">' +
      '<div class="card"><header><h3>参数配置</h3></header><div class="param-grid" id="live-params"></div>' +
      '<div class="hint" id="live-errors"></div></div>' +
      '<div class="card"><header><h3>实时统计</h3></header><div id="live-stats"></div></div>' +
      "</div>" +
      '<div class="panels"><div class="panel-img"><span class="panel-title">叠加画面</span>' +
      '<span class="panel-badge" id="badge-overlay">-</span><img id="img-overlay" alt="叠加画面"></div>' +
      '<div class="panel-img"><span class="panel-title">深度伪彩</span><img id="img-depth" alt="深度图"></div></div>' +
      '<div class="card"><header><h3>检出目标与距离</h3></header><div id="live-objects"></div></div>' +
      '<div class="card"><header><h3>本次会话记录 <span class="muted small" id="live-record-count"></span></h3>' +
      '<button class="sm" id="live-export">导出 CSV</button></header><div id="live-records"></div></div>';
  }

  async function initLivePage(mode) {
    const root = el("live-root");
    root.innerHTML = liveLayout();
    await API.post("/api/v1/pipeline/config", { mode: mode });
    const cfg = await API.get("/api/v1/pipeline/config");
    const options = (cfg.data || {}).options || {};

    el("live-start").onclick = startLiveSession;
    el("live-stop").onclick = stopLiveSession;
    el("live-pause").onclick = async function () {
      try { await API.post("/api/v1/pipeline/session/pause", {}); Toast.ok("已暂停（可录入真值）"); }
      catch (error) { handleError(error); }
    };
    el("live-save").onclick = async function () {
      try { await API.post("/api/v1/pipeline/save-frame", {}); Toast.ok("当前帧已落库"); }
      catch (error) { handleError(error); }
    };
    el("live-bulk").onclick = async function () {
      try {
        const r = await API.post("/api/v1/ranging/records/bulk", { illumination: "sunny" });
        Toast.ok("已落库 " + ((r.data || {}).created || 0) + " 条测距记录");
        refreshLiveRecords();
      } catch (error) { handleError(error); }
    };
    el("live-export").onclick = function () {
      window.open("/api/v1/ranging/records/export?task_type=" +
        (mode === "segmentation" ? "segmentation" : "detection"), "_blank");
    };

    await renderLiveParams(mode, options);
    startLivePolling();
    refreshLiveRecords();
    addPoller(poll(refreshLiveRecords, 15000));
  }

  async function renderLiveParams(mode, options) {
    const plugins = await API.get("/api/v1/models/plugins");
    const items = (plugins.data || {}).items || [];
    const detItems = items.filter(x => x.task === "detection");
    const segItems = items.filter(x => x.task === "segmentation");
    const classes = options.classes || [];

    let s = "";
    if (mode === "detection" || mode === "both") {
      s += '<label class="field"><span>检测器</span><select id="p-detector">' +
        UI.optionList(detItems.map(x => ({ key: x.key, label: (x.label || x.key) })), "key", "label",
          "sim_ground_truth") + "</select></label>";
      s += '<label class="field"><span>置信度阈值 <span class="value-badge" id="v-conf">0.30</span></span>' +
        '<input type="range" id="p-conf" min="0.05" max="0.95" step="0.05" value="0.30"></label>';
      s += '<label class="field"><span>类别过滤（不选=全部）</span></label><div id="p-classes">' +
        UI.checkboxGroup("cls", classes.map(c => ({ key: c, label: c })), []) + "</div>";
    }
    if (mode === "segmentation" || mode === "both") {
      s += '<label class="field"><span>分割器</span><select id="p-segmenter">' +
        UI.optionList(segItems.map(x => ({ key: x.key, label: (x.label || x.key) })), "key", "label",
          "classical_road") + "</select></label>";
      s += '<label class="field"><span>二值化阈值 <span class="value-badge" id="v-threshold">0.30</span></span>' +
        '<input type="range" id="p-threshold" min="0.05" max="0.95" step="0.05" value="0.30"></label>';
    }
    s += '<label class="field"><span>采样点数 <span class="value-badge" id="v-points">20</span></span>' +
      '<input type="range" id="p-points" min="6" max="60" step="2" value="20"></label>';
    s += '<label class="field"><span>深度采样核 <span class="value-badge" id="v-kernel">21</span></span>' +
      '<input type="range" id="p-kernel" min="1" max="51" step="2" value="21"></label>';
    s += '<label class="field"><span>走廊参考深度(m)</span><input id="p-corridor" type="number" value="5" step="0.5"></label>';
    html("live-params", s);

    const labelMap = { "p-conf": "v-conf", "p-threshold": "v-threshold",
                       "p-points": "v-points", "p-kernel": "v-kernel" };
    Object.keys(labelMap).forEach(function (id) {
      const input = el(id);
      if (input) input.oninput = function () { html(labelMap[id], Number(input.value).toFixed(2)); };
    });

    const doApply = async function () {
      const body = {
        mode: mode,
        conf: num("p-conf", 0.3), threshold: num("p-threshold", 0.3),
        sample_points: num("p-points", 20), kernel: num("p-kernel", 21),
        corridor_ref_depth_m: num("p-corridor", 5), classes: UI.checkedValues("cls"),
      };
      if (el("p-detector")) body.detector_key = value("p-detector");
      if (el("p-segmenter")) body.segmenter_key = value("p-segmenter");
      try {
        const r = await API.post("/api/v1/pipeline/config", body);
        const errors = ((r.data || {}).load_errors) || [];
        html("live-errors", errors.length
          ? '<span class="error-text">' + errors.map(Fmt.escape).join("<br>") + "</span>"
          : '<span class="tag ok">插件已加载</span>');
      } catch (error) { handleError(error, "参数应用失败"); }
    };
    ["p-detector", "p-segmenter"].forEach(function (id) { if (el(id)) el(id).onchange = doApply; });
    ["p-conf", "p-threshold", "p-points", "p-kernel", "p-corridor"].forEach(function (id) {
      if (el(id)) el(id).onchange = doApply;
    });
    document.querySelectorAll('input[name="cls"]').forEach(c => { c.onchange = doApply; });
    await doApply();
  }

  async function startLiveSession() {
    try {
      const r = await API.post("/api/v1/pipeline/session/start", {
        project_id: API.projectId(),
        name: "实时实验 " + new Date().toLocaleTimeString(),
        task_type: window.VISIONLAB.page === "ranging" ? "joint" : "detection",
      });
      Toast.ok("实验已开始");
      html("live-errors", '<span class="tag ok">实验进行中 ' +
        Fmt.escape((r.data || {}).experiment_id || "") + "</span>");
    } catch (error) { handleError(error, "开始实验失败"); }
  }

  async function stopLiveSession() {
    try {
      await API.post("/api/v1/pipeline/session/stop", {});
      Toast.ok("实验已结束，指标已归档");
      refreshLiveRecords();
    } catch (error) { handleError(error); }
  }

  function startLivePolling() {
    const socket = connectSocket(renderFrame);
    if (socket) {
      html("live-source", "WebSocket 实时");
      addPoller({ stop() { socket.close(); } });
    } else {
      html("live-source", "HTTP 轮询");
      addPoller(poll(async function () {
        const d = await API.post("/api/v1/pipeline/process", {});
        renderFrame(d.data);
      }, 800));
    }
  }

  function renderFrame(payload) {
    if (!payload) return;
    const overlayImg = el("img-overlay"), depthImg = el("img-depth");
    if (overlayImg) {
      const data = payload.image || payload.overlay_base64;
      if (data) overlayImg.src = "data:image/jpeg;base64," + data;
    }
    if (depthImg) {
      const data = payload.depth_image || payload.depth_base64;
      if (data) depthImg.src = "data:image/jpeg;base64," + data;
    }
    const det = payload.detection || {};
    const perf = payload.perf || {};
    html("live-frame", Fmt.int(payload.frame_number || 0));
    html("live-latency", Fmt.num(perf.latency_ms || perf.pipeline_ms, 0) + " ms");
    html("live-count", det.count !== undefined ? det.count : (det.detections || []).length);
    const road = payload.road || {};
    html("live-corridor", road.corridor_width_m ? Fmt.m(road.corridor_width_m) : "-");
    html("badge-overlay", payload.paused ? "已暂停" : (det.arch || "-"));

    const objects = payload.objects || [];
    html("live-objects", objects.length ? UI.table([
      { title: "类别", key: "category" },
      { title: "尺寸组", render: r => Fmt.sizeGroup(r.size_group) },
      { title: "中心距离(m)", num: true, render: r => Fmt.num((r.ranging.center || {}).depth_m, 3) },
      { title: "最近(m)", num: true, render: r => Fmt.num((r.ranging.min || {}).depth_m, 3) },
      { title: "最远(m)", num: true, render: r => Fmt.num((r.ranging.max || {}).depth_m, 3) },
      { title: "横向偏移(m)", num: true, render: r => Fmt.num((r.ranging.center || {}).lateral_m, 3) },
      { title: "有效点", num: true, render: r => (r.ranging.center || {}).valid_points || "-" },
      { title: "操作", render: r => '<button class="sm" data-rec="' + Fmt.escape(r.category) + '">记录</button>' },
    ], objects, { scroll: false }) : '<div class="empty">当前帧没有检出目标</div>');

    document.querySelectorAll("[data-rec]").forEach(function (btn) {
      btn.onclick = function () { recordModal({ c: btn.dataset.rec }); };
    });

    const stats = [];
    if (road.left && road.left.min) stats.push(["道路左边界最小", Fmt.m(road.left.min.depth_m)]);
    if (road.right && road.right.min) stats.push(["道路右边界最小", Fmt.m(road.right.min.depth_m)]);
    if (road.corridor_width_m) stats.push(["可通行走廊", Fmt.m(road.corridor_width_m)]);
    if (road.mask_area_ratio !== undefined) stats.push(["掩膜占比", Fmt.pct(road.mask_area_ratio)]);
    if (det.infer_ms) stats.push(["推理耗时", Fmt.num(det.infer_ms, 1) + " ms"]);
    stats.push(["流水线耗时", Fmt.num(perf.pipeline_ms, 1) + " ms"]);
    if (payload.ground_truth && payload.ground_truth.length) {
      stats.push(["仿真真值目标", payload.ground_truth.length]);
    }
    html("live-stats", UI.statusLine(stats));
    if (payload.errors && payload.errors.length) {
      html("live-errors", '<span class="error-text">' + payload.errors.map(Fmt.escape).join("<br>") + "</span>");
    }
  }

  function recordModal(info) {
    Modal.open("记录测距结果",
      '<p class="muted small">目标：' + Fmt.escape(info.c || "") + "</p>" +
      '<label class="field"><span>实际距离（卷尺 / 激光测距仪，单位 m）</span>' +
      '<input id="rec-actual" type="number" step="0.01" placeholder="留空则只记录测量值"></label>' +
      '<label class="field"><span>光照条件</span>' +
      UI.select("rec-illum", API.dict().illuminations || [], "sunny") + "</label>" +
      '<label class="field"><span>备注</span><input id="rec-notes"></label>',
      '<button data-close="1">取消</button><button class="primary" id="rec-save">记录</button>');
    el("rec-save").onclick = async function () {
      try {
        await API.post("/api/v1/ranging/records", {
          use_last: true, target_kind: "object", metric_kind: "center", category: info.c,
          actual_distance_m: value("rec-actual") || null,
          illumination: value("rec-illum"), notes: value("rec-notes"),
        });
        Toast.ok("已记录");
        Modal.close();
        refreshLiveRecords();
      } catch (error) { handleError(error); }
    };
  }

  async function refreshLiveRecords() {
    try {
      const d = await API.get("/api/v1/ranging/records", { page_size: 30 });
      html("live-record-count", "（" + (d.total || 0) + " 条）");
      html("live-records", UI.table(recordColumns().slice(0, 8).concat([
        { title: "操作", render: r => '<button class="sm danger" data-delrec="' + r.id + '">删除</button>' },
      ]), d.data || [], { empty: "本次还没有记录" }));
      document.querySelectorAll("[data-delrec]").forEach(function (btn) {
        btn.onclick = async function () {
          try { await API.del("/api/v1/ranging/records/" + btn.dataset.delrec); refreshLiveRecords(); }
          catch (error) { handleError(error); }
        };
      });
    } catch (error) { /* 静默 */ }
  }

  Pages.experiment_detection = async function () { await initLivePage("detection"); };
  Pages.experiment_segmentation = async function () {
    await initLivePage("segmentation");
    await renderSegExtra();
  };
  Pages.ranging = async function () {
    await initLivePage("both");
    renderRangingExtra();
  };

  async function renderSegExtra() {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = "<header><h3>批量评测（多模型 x 测试集）</h3>" +
      '<button class="primary sm" id="btn-batch-eval">发起批量评测</button></header>' +
      '<div class="hint">批量评测由任务队列执行：勾选分割结构与训练规模，平台将在测试集上统计 ' +
      "MIoU / Dice / Precision / Recall 并汇入大屏与报告。</div>" +
      '<div id="batch-eval-result"></div>';
    el("live-root").appendChild(card);
    el("btn-batch-eval").onclick = async function () {
      const models = await API.get("/api/v1/models", { task_type: "segmentation", page_size: 100 });
      const rows = models.data || [];
      if (!rows.length) return Toast.warn("尚未登记分割模型，请先在「模型管理」中扫描历史权重");
      Modal.open("批量评测",
        '<p class="muted small">将以下模型加入评测队列：</p>' +
        UI.table([{ title: "选择", render: (r, i) =>
          '<input type="checkbox" name="eval-model" value="' + r.id + '"' + (i < 4 ? " checked" : "") + ">" },
          { title: "模型", key: "name" }, { title: "结构", key: "arch" },
          { title: "训练规模", key: "variant" },
          { title: "已有 MIoU", num: true, render: r => Fmt.num((r.metrics || {}).miou, 4) }],
          rows.slice(0, 40), { scroll: false }),
        '<button data-close="1">取消</button><button class="primary" id="eval-confirm">提交评测</button>');
      el("eval-confirm").onclick = function () {
        const ids = UI.checkedValues("eval-model");
        Modal.close();
        html("batch-eval-result", '<div class="hint">已提交 ' + ids.length +
          " 个模型的评测任务，指标完成后写入模型卡片并汇入大屏与报告。</div>");
        Toast.ok("已提交 " + ids.length + " 个模型");
      };
    };
  }

  function renderRangingExtra() {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = "<header><h3>统一测距内核说明</h3></header>" +
      "<p class='small muted'>平台将「目标边界测距」与「道路边界测距」识别为同一算法本质的两种实例：" +
      "在指定图像区域内采样深度、剔除无效值、换算为物理距离。两者共享深度采样器与结果契约，" +
      "差异只在采样点来自检测框还是分割掩膜。</p>" +
      "<ul class='small muted'>" +
      "<li><b>目标边界测距</b>：检测框中心 21x21 局部中位数 + 框边界多点采样 → 最近 / 最远 / 平均距离</li>" +
      "<li><b>道路边界测距</b>：掩膜最大外轮廓等距采样 + 按质心左右分离 → 左 / 右边界最近最远距离</li>" +
      "<li><b>平台新增</b>：针孔反投影 X=(u-ppx)*Z/fx 输出横向偏移；在同一参考深度上计算可通行走廊宽度</li>" +
      "<li><b>有效点定义</b>：&gt; 0 且 &lt;= 20 m（沿用原项目 B）；采样模式：局部中位数（沿用原项目 A）</li>" +
      "</ul>";
    el("live-root").appendChild(card);
  }

  /* ================================================================
     相机标定
     ================================================================ */
  Pages.calibration = async function () {
    html("page-actions", '<button id="btn-start-camera">启动相机</button>' +
      '<button class="primary" id="btn-calibrate">执行标定</button>' +
      '<button id="btn-probe">中心深度探测</button>');
    el("btn-start-camera").onclick = async function () {
      try { await API.post("/api/v1/camera/stream/start", { backend: "auto" }); Toast.ok("相机已启动"); loadCalibration(); }
      catch (error) { handleError(error); }
    };
    el("btn-calibrate").onclick = async function () {
      try {
        const r = await API.post("/api/v1/camera/calibrate", {});
        Toast.ok("标定完成");
        showCalibration(r.data);
        loadCalibration();
      } catch (error) { handleError(error); }
    };
    el("btn-probe").onclick = async function () {
      try {
        const r = await API.get("/api/v1/camera/depth-probe");
        Modal.open("深度探测", UI.defList([
          ["像素坐标", "(" + r.data.u + ", " + r.data.v + ")"],
          ["沿光轴深度 Z", Fmt.m(r.data.depth_m)],
          ["横向 X", Fmt.m(r.data.x_m)],
          ["纵向 Y", Fmt.m(r.data.y_m)],
          ["有效点", r.data.valid_points + " / " + r.data.total_points],
        ]));
      } catch (error) { handleError(error); }
    };
    await loadCalibration();
    addPoller(poll(loadCalibration, 12000));
  };

  async function loadCalibration() {
    const info = await API.get("/api/v1/camera/status");
    const i = info.data || {};
    const intr = i.intrinsics || {};
    html("camera-info", UI.defList([
      ["后端", i.backend === "realsense" ? "Intel RealSense（真实设备）" : "合成场景 MockBackend"],
      ["型号", i.model || "-"], ["序列号", i.serial || "-"], ["固件", i.firmware || "-"],
      ["USB", i.usb_type || "-"],
      ["分辨率 / 帧率", (i.width || "-") + "x" + (i.height || "-") + " @ " + (i.fps || "-") + " fps"],
      ["深度尺度", i.depth_scale !== undefined ? i.depth_scale : "-"],
      ["实际帧率", (i.fps_actual || 0) + " fps"],
      ["滤波链", (i.filters && i.filters.length) ? i.filters.join(", ") : "关闭（保留原始深度细节）"],
      ["内参 fx / fy", Fmt.num(intr.fx, 2) + " / " + Fmt.num(intr.fy, 2)],
      ["主点 ppx / ppy", Fmt.num(intr.ppx, 2) + " / " + Fmt.num(intr.ppy, 2)],
    ]));
    const list = await API.get("/api/v1/camera/calibrations");
    const items = (list.data || {}).items || [];
    html("calibration-list", UI.table([
      { title: "名称", key: "name" },
      { title: "深度尺度", num: true, render: r => Fmt.num(r.depth_scale, 6) },
      { title: "RMS 偏差(m)", num: true, render: r => Fmt.num(r.rms_reprojection_px, 4) },
      { title: "平均偏差(m)", num: true, render: r => Fmt.num((r.depth_check || {}).mean_abs_error_m, 4) },
      { title: "默认", render: r => (r.is_default ? '<span class="tag ok">是</span>' : "") },
      { title: "创建", render: r => Fmt.time(r.created_at) },
      { title: "操作", render: r => '<button class="sm" data-cal="' + r.id + '">查看</button> ' +
        '<button class="sm" data-default="' + r.id + '">设为默认</button>' },
    ], items, { empty: "尚未执行标定" }));
    document.querySelectorAll("[data-cal]").forEach(function (btn) {
      btn.onclick = function () { showCalibration(items.filter(x => x.id === btn.dataset.cal)[0]); };
    });
    document.querySelectorAll("[data-default]").forEach(function (btn) {
      btn.onclick = async function () {
        try { await API.post("/api/v1/camera/calibrations/" + btn.dataset.default + "/default", {});
              Toast.ok("已设为默认标定"); loadCalibration(); }
        catch (error) { handleError(error); }
      };
    });
  }

  function showCalibration(data) {
    if (!data) return;
    const check = data.depth_check || {};
    const bars = (check.samples || []).map(function (s) {
      return "<tr><td>" + Fmt.m(s.actual_m) + '</td><td class="num">' + Fmt.m(s.measured_m) +
        '</td><td class="num">' + Fmt.m(s.abs_error_m) + '</td><td class="num">' +
        Fmt.pct(s.rel_error) + "</td></tr>";
    }).join("");
    Modal.open("标定结果 · " + Fmt.escape(data.name || ""),
      UI.defList([
        ["深度尺度", Fmt.num(data.depth_scale, 6)],
        ["RMS 偏差(m)", Fmt.num(data.rms_reprojection_px, 4)],
        ["平均绝对偏差(m)", Fmt.num(check.mean_abs_error_m, 4)],
        ["标定文件", data.file_uri || "-"],
        ["文件哈希", (data.file_sha256 || "-").slice(0, 24) + "..."],
      ]) +
      "<h4>深度平面校验</h4>" +
      (bars ? '<div class="table-wrap"><table><thead><tr><th>真值</th><th class="num">实测</th>' +
        '<th class="num">偏差</th><th class="num">相对偏差</th></tr></thead><tbody>' + bars +
        "</tbody></table></div>" : '<div class="empty">本次没有采集到真值样本</div>'));
  }

  /* ================================================================
     数据采集
     ================================================================ */
  let captureSession = null;
  Pages.capture = async function () {
    html("page-actions", '<button class="primary" id="btn-new-session">新建采集会话</button>' +
      '<button id="btn-snapshot">抓拍</button><button id="btn-record">开始/停止录制</button>' +
      '<button id="btn-finish">结束会话</button>');
    el("btn-new-session").onclick = async function () {
      try {
        const r = await API.post("/api/v1/capture/sessions", {
          name: "采集 " + new Date().toLocaleString(),
          illumination: value("cap-illum") || "sunny",
        });
        captureSession = r.data;
        Toast.ok("采集会话已创建");
        renderCapture();
      } catch (error) { handleError(error); }
    };
    el("btn-snapshot").onclick = async function () {
      if (!captureSession) return Toast.warn("请先新建采集会话");
      try {
        await API.post("/api/v1/capture/sessions/" + captureSession.id + "/snapshot",
          { illumination: value("cap-illum"), target_size_group: value("cap-size"),
            notes: value("cap-notes") });
        Toast.ok("已抓拍（RGB + 16 位深度）");
        await reloadCaptureSession();
      } catch (error) { handleError(error); }
    };
    el("btn-record").onclick = async function () {
      if (!captureSession) return Toast.warn("请先新建采集会话");
      try {
        const stopping = captureSession.recording;
        await API.post("/api/v1/capture/sessions/" + captureSession.id + "/recording/" +
          (stopping ? "stop" : "start"), {});
        await reloadCaptureSession();
        Toast.ok(stopping ? "录制已停止" : "录制已开始");
      } catch (error) { handleError(error); }
    };
    el("btn-finish").onclick = async function () {
      if (!captureSession) return;
      try {
        const r = await API.post("/api/v1/capture/sessions/" + captureSession.id + "/finish", {});
        Toast.ok("会话已结束，共 " + ((r.data || {}).item_count || 0) + " 条采集记录");
        captureSession = r.data;
        renderCapture();
      } catch (error) { handleError(error); }
    };

    html("capture-config",
      '<div class="form-row">' +
      '<label class="field"><span>光照条件</span>' +
      UI.select("cap-illum", API.dict().illuminations || [], "sunny") + "</label>" +
      '<label class="field"><span>目标尺寸组</span><select id="cap-size"><option value="">不指定</option>' +
      '<option value="large">大目标</option><option value="small">小目标</option></select></label>' +
      '<label class="field"><span>备注</span><input id="cap-notes"></label></div>' +
      '<div class="hint">深度图以 16 位 PNG 无损落盘；抓拍同时保存内参与深度尺度到会话元数据，' +
      "并在指定数据集版本时自动登记为数据集图像。</div>");

    await renderCaptureSource();

    addPoller(poll(function () {
      const img = el("cap-rgb");
      if (img) img.src = "/api/v1/camera/frame?t=" + Date.now();
    }, 1200));
    addPoller(poll(function () {
      const img = el("cap-depth");
      if (img) img.src = "/api/v1/camera/depth_frame?t=" + Date.now();
    }, 1600));

    const sessions = await API.get("/api/v1/capture/sessions");
    html("capture-sessions", UI.table([
      { title: "会话", key: "name" },
      { title: "场景", key: "scene" },
      { title: "光照", render: r => Fmt.illum(r.illumination) },
      { title: "采集数", key: "item_count", num: true },
      { title: "含真值", key: "with_truth", num: true },
      { title: "开始", render: r => Fmt.time(r.started_at) },
      { title: "操作", render: r => '<button class="sm" data-open="' + r.id + '">打开</button>' },
    ], (sessions.data || {}).items || [], { empty: "暂无采集会话" }));
    document.querySelectorAll("[data-open]").forEach(function (btn) {
      btn.onclick = async function () {
        const r = await API.get("/api/v1/capture/sessions/" + btn.dataset.open);
        captureSession = r.data;
        renderCapture();
      };
    });
  };

  async function reloadCaptureSession() {
    if (!captureSession) return;
    const r = await API.get("/api/v1/capture/sessions/" + captureSession.id);
    captureSession = r.data;
    renderCapture();
  }

  /* ---------------- 数据源切换：真实相机 / 合成场景 / 离线回放 ---------------- */
  async function renderCaptureSource() {
    const status = await API.get("/api/v1/camera/status");
    const info = status.data || {};
    const backends = await API.get("/api/v1/camera/backends");
    const sources = await API.get("/api/v1/camera/replay-sources");
    const items = (sources.data || {}).items || [];
    const captureSources = items.filter(x => x.type === "capture");
    const datasetSources = items.filter(x => x.type === "dataset");

    html("capture-source",
      '<div class="form-row">' +
      '<label class="field"><span>采集来源</span><select id="src-backend">' +
      UI.optionList((backends.data || {}).items || [], "key", "label", info.backend) +
      "</select></label>" +
      '<label class="field"><span>回放数据源（仅回放模式使用）</span><select id="src-replay">' +
      '<option value="">（无可用回放源）</option>' +
      captureSources.map(s => '<option value="capture|' + s.id + '">采集会话 · ' +
        Fmt.escape(s.label) + "（" + s.frames + " 帧" +
        (s.with_truth ? "，含 " + s.with_truth + " 条真值" : "") + "）</option>").join("") +
      datasetSources.map(s => '<option value="dataset|' + s.id + '">数据集 · ' +
        Fmt.escape(s.label) + "（" + s.frames + " 帧）</option>").join("") +
      "</select></label>" +
      '<label class="field"><span>回放帧率</span><input id="src-fps" type="number" value="15" min="1" max="60"></label>' +
      '<label class="field"><span>&nbsp;</span><button class="primary" id="src-apply">切换数据源</button> ' +
      '<button id="src-make-sample">抓取回放样本</button></label>' +
      "</div>" +
      '<div class="hint">' +
      '<b>真实相机</b>：RealSense D455 实时采流；<b>合成场景</b>：内置模拟 RGB-D（含随距离放大的深度偏置），无设备也能完整演示；' +
      "<b>离线回放</b>：从落盘的真实 RGB-D 序列按帧重放，用于复现验证与教学分发。" +
      "</div>" +
      '<div id="src-status"></div>');

    async function refreshStatus() {
      const s = await API.get("/api/v1/camera/status");
      const d = s.data || {};
      let extra = "";
      if (d.backend === "replay") {
        const progress = await API.get("/api/v1/camera/replay-progress");
        const p = progress.data || {};
        extra = '<div class="status-line small" style="margin-top:8px">' +
          "<span>回放源 <b>" + Fmt.escape(String(p.source || "-")) + "</b></span>" +
          "<span>进度 <b>" + (p.frame_index || 0) + " / " + (p.frames_total || 0) + "</b></span>" +
          "<span>已送出 <b>" + (p.frames_served || 0) + "</b></span>" +
          "<span>循环 <b>" + (p.looping ? "开" : "关") + "</b></span></div>";
      }
      html("src-status", UI.statusLine([
        ["当前来源", Fmt.escape(d.backend || "-")],
        ["型号", Fmt.escape(d.model || "-")],
        ["分辨率", (d.width || "-") + "x" + (d.height || "-")],
        ["实际帧率", (d.fps_actual || 0) + " fps"],
        ["累计帧", Fmt.int(d.frames || 0)],
      ]) + extra);
    }

    el("src-apply").onclick = async function () {
      const mode = value("src-backend");
      const body = { backend: mode };
      const replay = value("src-replay");
      if (mode === "replay" || replay) {
        if (!replay) { Toast.warn("请先选择一个回放数据源"); return; }
        const [type, id] = replay.split("|");
        body.backend = "replay";
        body.replay_type = type;
        body.replay_source = id;
        body.fps = num("src-fps", 15);
        body.loop = true;
      }
      try {
        const r = await API.post("/api/v1/camera/stream/start", body);
        Toast.ok("已切换数据源：" + ((r.data || {}).model || mode));
        await refreshStatus();
        renderCaptureSource();
      } catch (error) { handleError(error, "切换数据源失败"); }
    };

    el("src-make-sample").onclick = async function () {
      Modal.open("抓取回放样本",
        '<p class="muted small">用当前数据源抓取若干帧落盘，生成一个可用于「离线回放」的真实 RGB-D 序列。</p>' +
        '<label class="field"><span>抓取帧数</span><input id="sample-frames" type="number" value="40" min="5" max="300"></label>',
        '<button data-close="1">取消</button><button class="primary" id="sample-go">开始抓取</button>');
      el("sample-go").onclick = async function () {
        const frames = num("sample-frames", 40);
        Modal.close();
        Toast.info("正在抓取 " + frames + " 帧，请稍候…");
        try {
          const r = await API.post("/api/v1/camera/replay-sample", { frames: frames });
          Toast.ok("已生成回放样本：" + (r.data || {}).session_id +
            "（" + ((r.data || {}).frames || 0) + " 帧）");
          renderCaptureSource();
        } catch (error) { handleError(error, "抓取失败"); }
      };
    };

    await refreshStatus();
    addPoller(poll(refreshStatus, 4000));
  }

  function renderCapture() {
    if (!captureSession) return;
    const items = captureSession.items || [];
    html("capture-info", UI.defList([
      ["会话 ID", captureSession.id], ["名称", captureSession.name],
      ["录制状态", captureSession.recording ? "录制中" : "未录制"],
      ["采集数量", items.length],
      ["已录真值", items.filter(x => x.truth).length],
      ["存储目录", captureSession.root],
    ]));
    html("capture-items", UI.table([
      { title: "序号", render: (r, i) => i + 1 },
      { title: "RGB", render: r => '<img src="/api/v1/media/thumb?uri=' +
        encodeURIComponent(r.rgb_uri) + '&width=120" style="height:52px;border-radius:4px" alt="">' },
      { title: "时间", key: "stamp" },
      { title: "光照", render: r => Fmt.illum(r.illumination) },
      { title: "真值(m)", num: true, render: r => r.truth ? Fmt.num(r.truth.actual_distance_m, 2) : "-" },
      { title: "操作", render: r => '<button class="sm" data-truth="' + r.stamp + '">录入真值</button>' },
    ], items.slice().reverse(), { empty: "还没有抓拍记录" }));
    document.querySelectorAll("[data-truth]").forEach(function (btn) {
      btn.onclick = function () { truthModal(btn.dataset.truth); };
    });
    const plan = API.dict().distance_plan_m || [];
    html("capture-plan", plan.map(function (p) {
      const hit = items.filter(x => x.truth && Math.abs((x.truth.actual_distance_m || 0) - p) < 0.35).length;
      return '<span class="tag ' + (hit ? "ok" : "") + '" style="margin:2px">' + p + " m" +
        (hit ? " 已录" : "") + "</span>";
    }).join(""));
  }

  function truthModal(stamp) {
    Modal.open("录入真值",
      '<div class="form-row">' +
      '<label class="field"><span>实际距离(m)</span><input id="t-one" type="number" step="0.01"></label>' +
      '<label class="field"><span>实际最近(m)</span><input id="t-min" type="number" step="0.01"></label>' +
      '<label class="field"><span>实际最远(m)</span><input id="t-max" type="number" step="0.01"></label></div>' +
      '<div class="form-row"><label class="field"><span>测量手段</span><select id="t-src">' +
      '<option value="tape">卷尺</option><option value="laser">激光测距仪</option>' +
      '<option value="manual">人工估计</option></select></label>' +
      '<label class="field"><span>类别</span>' + UI.select("t-cat",
        (API.dict().detection_classes || []).map(c => ({ key: c, label: c })), "", "不指定") + "</label></div>" +
      '<label class="field"><span>备注</span><input id="t-notes"></label>',
      '<button data-close="1">取消</button><button class="primary" id="t-save">保存</button>');
    el("t-save").onclick = async function () {
      try {
        await API.post("/api/v1/capture/sessions/" + captureSession.id + "/truth", {
          stamp: stamp, actual_distance_m: value("t-one") || null,
          actual_min_m: value("t-min") || null, actual_max_m: value("t-max") || null,
          actual_source: value("t-src"), category: value("t-cat") || null,
          notes: value("t-notes"),
        });
        Toast.ok("真值已录入");
        Modal.close();
        await reloadCaptureSession();
      } catch (error) { handleError(error); }
    };
  }

  /* ================================================================
     数据集
     ================================================================ */
  Pages.datasets = async function () {
    html("page-actions", '<button class="primary" id="btn-new-dataset">新建数据集</button>' +
      '<button id="btn-import-legacy">导入历史数据集</button>');
    el("btn-new-dataset").onclick = function () {
      Modal.open("新建数据集",
        '<label class="field"><span>名称 *</span><input id="nd-name"></label>' +
        '<div class="form-row">' +
        '<label class="field"><span>任务类型</span><select id="nd-task">' +
        '<option value="detection">目标检测</option><option value="segmentation">道路语义分割</option>' +
        "</select></label>" +
        '<label class="field"><span>场景</span>' +
        UI.select("nd-scene", API.dict().scenes || [], "campus_road") + "</label></div>" +
        '<label class="field"><span>说明</span><input id="nd-desc"></label>',
        '<button data-close="1">取消</button><button class="primary" id="nd-save">创建</button>');
      el("nd-save").onclick = async function () {
        try {
          await API.post("/api/v1/datasets", {
            name: value("nd-name"), task_type: value("nd-task"),
            scene: value("nd-scene"), description: value("nd-desc"), project_id: API.projectId(),
          });
          Toast.ok("数据集已创建");
          Modal.close();
          loadDatasets();
        } catch (error) { handleError(error); }
      };
    };
    el("btn-import-legacy").onclick = function () {
      Modal.open("导入历史数据集（来自两个原项目）",
        '<p class="muted small">填写原项目数据目录的绝对路径；平台只读取并登记，不移动原文件。</p>' +
        '<label class="field"><span>项目 A 检测图像目录（datasets\\images）</span>' +
        '<input id="lg-det" placeholder="E:\\2Project\\2026dachuang\\Yolo_dis_v3源代码\\datasets\\images"></label>' +
        '<label class="field"><span>项目 B 分割图像目录（dataset\\train\\images）</span>' +
        '<input id="lg-img" placeholder="E:\\2Project\\2026dachuang\\Unet\\dataset\\train\\images"></label>' +
        '<label class="field"><span>项目 B 分割掩膜目录（dataset\\train\\masks）</span>' +
        '<input id="lg-mask" placeholder="E:\\2Project\\2026dachuang\\Unet\\dataset\\train\\masks"></label>' +
        '<label class="field"><span>项目 B 权重目录（models）</span>' +
        '<input id="lg-weights" placeholder="E:\\2Project\\2026dachuang\\Unet\\models"></label>' +
        '<label class="field"><span>项目 B best_model 目录（可选）</span>' +
        '<input id="lg-best" placeholder="E:\\2Project\\2026dachuang\\Unet\\best_model"></label>' +
        '<label class="field"><span>项目 A 训练产物目录 runs（可选）</span>' +
        '<input id="lg-runs" placeholder="E:\\2Project\\2026dachuang\\Yolo_dis_v3源代码\\runs"></label>' +
        '<label class="field"><span>项目 A 测距记录 CSV（可选）</span>' +
        '<input id="lg-csv" placeholder="E:\\2Project\\2026dachuang\\Yolo_dis_v3源代码\\data\\results\\measurements.csv"></label>',
        '<button data-close="1">取消</button><button class="primary" id="lg-run">开始导入</button>');
      el("lg-run").onclick = async function () {
        const body = { project_id: API.projectId() };
        if (value("lg-det")) body.det_images_dir = value("lg-det");
        if (value("lg-img") && value("lg-mask")) {
          body.seg_images_dir = value("lg-img"); body.seg_masks_dir = value("lg-mask");
        }
        if (value("lg-weights")) body.seg_models_dir = value("lg-weights");
        if (value("lg-best")) body.best_model_dir = value("lg-best");
        if (value("lg-runs")) body.runs_dir = value("lg-runs");
        if (value("lg-csv")) body.measurements_csv = value("lg-csv");
        Modal.close();
        Toast.info("导入中，请稍候...");
        try {
          const r = await API.post("/api/v1/analysis/import-legacy", body);
          Modal.open("导入结果", "<pre class='mono' style='max-height:60vh;overflow:auto'>" +
            Fmt.escape(JSON.stringify(r.data, null, 2)) + "</pre>");
          loadDatasets();
        } catch (error) { handleError(error, "导入失败"); }
      };
    };
    await loadDatasets();
  };

  async function loadDatasets() {
    UI.loading("dataset-list");
    const d = await API.get("/api/v1/datasets", { page_size: 100 });
    html("dataset-list", UI.table([
      { title: "数据集", render: r => '<a href="/datasets/' + r.id + '">' + Fmt.escape(r.name) + "</a>" },
      { title: "任务", render: r => Fmt.taskTag(r.task_type) },
      { title: "场景", key: "scene" },
      { title: "状态", render: r => Fmt.statusTag(r.status) },
      { title: "最新版本", key: "latest_version" },
      { title: "版本数", key: "version_count", num: true },
      { title: "图像数", key: "image_count", num: true },
      { title: "前景值白名单", render: r => Fmt.escape(JSON.stringify(r.foreground_values || [])) },
      { title: "更新", render: r => Fmt.time(r.updated_at) },
    ], d.data || [], { empty: "暂无数据集，可点击「导入历史数据集」" }));
  }

  Pages.dataset_detail = async function () {
    const id = location.pathname.split("/").pop();
    const d = await API.get("/api/v1/datasets/" + id);
    const ds = d.data || {};
    html("dataset-head", "<h2>" + Fmt.escape(ds.name) + "</h2>" +
      UI.statusLine([["任务", (API.dict().task_labels || {})[ds.task_type]],
                     ["场景", ds.scene], ["状态", Fmt.status(ds.status)],
                     ["类别", (ds.class_names || []).join(" / ")],
                     ["前景值白名单", JSON.stringify(ds.foreground_values || [])]]));
    html("page-actions",
      '<button id="btn-new-version" class="primary">新建版本（扫描目录）</button>' +
      '<button id="btn-validate">执行校验</button>');

    el("btn-new-version").onclick = function () {
      Modal.open("新建数据集版本",
        '<label class="field"><span>版本号</span><input id="dv-version" value="v1.0"></label>' +
        '<label class="field"><span>图像目录（绝对路径）</span><input id="dv-root"></label>' +
        '<label class="field"><span>划分比例</span><input id="dv-ratio" value="7:2:1"></label>' +
        '<label class="field"><span>随机种子</span><input id="dv-seed" type="number" value="42"></label>',
        '<button data-close="1">取消</button><button class="primary" id="dv-save">扫描并创建</button>');
      el("dv-save").onclick = async function () {
        try {
          const r = await API.post("/api/v1/datasets/" + id + "/versions", {
            version: value("dv-version"), source_root: value("dv-root"),
            split_ratio: value("dv-ratio"), split_seed: num("dv-seed", 42),
          });
          Toast.ok("已创建版本并登记 " + ((r.data || {}).images || 0) + " 张图像");
          Modal.close();
          location.reload();
        } catch (error) { handleError(error); }
      };
    };
    el("btn-validate").onclick = async function () {
      const versions = ds.versions || [];
      if (!versions.length) return Toast.warn("请先创建版本");
      try {
        const r = await API.post("/api/v1/datasets/versions/" + versions[0].id + "/validate", {});
        showValidation(r.data);
        loadVersions();
      } catch (error) { handleError(error); }
    };
    await loadVersions();
  };

  async function loadVersions() {
    const id = location.pathname.split("/").pop();
    const d = await API.get("/api/v1/datasets/" + id);
    const versions = (d.data || {}).versions || [];
    html("version-list", UI.table([
      { title: "版本", key: "version" },
      { title: "图像", key: "image_count", num: true },
      { title: "掩膜", key: "annotation_count", num: true },
      { title: "划分", render: r => Fmt.escape(JSON.stringify((r.stats || {}).counts || r.split_ratio || {})) },
      { title: "种子", key: "split_seed", num: true },
      { title: "内容哈希", render: r => '<span class="mono">' + (r.content_hash || "-").slice(0, 12) + "</span>" },
      { title: "冻结", render: r => r.frozen_at ? '<span class="tag ok">已冻结</span>' : '<span class="tag warn">未冻结</span>' },
      { title: "操作", render: r => '<button class="sm" data-validate="' + r.id + '">校验</button> ' +
        '<button class="sm" data-split="' + r.id + '">划分</button> ' +
        '<button class="sm" data-unify="' + r.id + '">统一前景值</button> ' +
        '<button class="sm" data-freeze="' + r.id + '">冻结</button> ' +
        '<button class="sm" data-images="' + r.id + '">浏览</button>' },
    ], versions, { empty: "尚无版本" }));

    document.querySelectorAll("[data-validate]").forEach(b => b.onclick = async function () {
      try { showValidation((await API.post("/api/v1/datasets/versions/" + b.dataset.validate + "/validate", {})).data); }
      catch (e) { handleError(e); }
    });
    document.querySelectorAll("[data-split]").forEach(b => b.onclick = async function () {
      try { await API.post("/api/v1/datasets/versions/" + b.dataset.split + "/split",
        { ratio: "7:2:1", seed: 42 }); Toast.ok("划分完成"); loadVersions(); }
      catch (e) { handleError(e); }
    });
    document.querySelectorAll("[data-unify]").forEach(b => b.onclick = async function () {
      try {
        const r = await API.post("/api/v1/datasets/versions/" + b.dataset.unify + "/unify-foreground", {});
        Toast.ok("已统一 " + ((r.data || {}).unified || 0) + " 个掩膜的前景值");
      } catch (e) { handleError(e); }
    });
    document.querySelectorAll("[data-freeze]").forEach(b => b.onclick = async function () {
      try { await API.post("/api/v1/datasets/versions/" + b.dataset.freeze + "/freeze", {});
            Toast.ok("版本已冻结"); loadVersions(); }
      catch (e) { handleError(e); }
    });
    document.querySelectorAll("[data-images]").forEach(b => b.onclick = function () {
      loadImages(b.dataset.images);
    });
  }

  async function loadImages(versionId) {
    const d = await API.get("/api/v1/datasets/versions/" + versionId + "/images", { page_size: 36 });
    html("image-grid", UI.gallery(d.data || [], item => item.url,
      item => (item.split || "-") + " · " + (item.has_mask ? "有掩膜" : "无掩膜")));
  }

  function showValidation(report) {
    if (!report) return;
    Modal.open("数据集校验报告",
      UI.defList([
        ["任务类型", (API.dict().task_labels || {})[report.task_type]],
        ["图像总数", report.total_images], ["抽检数量", report.checked_images],
        ["问题数", report.issue_count],
        ["前景值口径一致", report.foreground_value_consistent ? "是" : "否（需统一）"],
        ["前景值分布", Object.keys(report.foreground_value_histogram || {}).length
          ? JSON.stringify(report.foreground_value_histogram) : "（无掩膜）"],
        ["掩膜面积占比", JSON.stringify(report.mask_area_ratio || {})],
        ["结论", report.conclusion],
      ]) +
      "<h4>问题类型统计</h4>" +
      UI.table([{ title: "类型", key: "type" }, { title: "数量", key: "n", num: true }],
        Object.keys(report.issue_types || {}).map(k => ({ type: k, n: report.issue_types[k] })),
        { empty: "无问题", scroll: false }) +
      "<h4>问题样例</h4>" +
      UI.table([{ title: "类型", key: "type" }, { title: "图像", key: "image_id" },
                { title: "说明", key: "detail" }], (report.samples || []).slice(0, 20),
               { empty: "无", scroll: false }));
  }

  /* ================================================================
     标注管理
     ================================================================ */
  Pages.annotations = async function () {
    const datasets = await API.get("/api/v1/datasets", { page_size: 100 });
    const rows = datasets.data || [];
    html("ann-form",
      '<div class="form-row">' +
      '<label class="field"><span>数据集</span><select id="an-dataset">' +
      rows.map(r => '<option value="' + r.id + '">' + Fmt.escape(r.name) + "（" +
        ((API.dict().task_labels || {})[r.task_type] || "") + "）</option>").join("") +
      "</select></label>" +
      '<label class="field"><span>标注格式</span><select id="an-format">' +
      '<option value="labelme">LabelMe JSON</option><option value="yolo">YOLO txt</option>' +
      '<option value="coco">COCO JSON</option><option value="voc">Pascal VOC XML</option>' +
      '<option value="mask">掩膜 PNG（分割）</option></select></label>' +
      '<label class="field"><span>标注目录（绝对路径）</span><input id="an-dir"></label>' +
      '<label class="field"><span>&nbsp;</span><button class="primary" id="an-import">导入</button></label>' +
      "</div>" +
      '<div class="hint">导入后平台会执行完整性校验（漏标 / 越界 / 掩膜前景值口径 / 覆盖率异常），' +
      "结果可在数据集详情页查看。</div>");
    html("page-actions", '<button onclick="location.href=\'/datasets\'">前往数据集管理</button>');
    el("an-import").onclick = async function () {
      const datasetId = value("an-dataset");
      if (!datasetId) return Toast.warn("请先在数据集管理中创建数据集");
      const detail = await API.get("/api/v1/datasets/" + datasetId);
      const versions = (detail.data || {}).versions || [];
      if (!versions.length) return Toast.warn("该数据集还没有版本，请先在数据集详情页创建版本");
      Modal.open("选择目标版本",
        '<label class="field"><span>版本</span><select id="an-version">' +
        versions.map(v => '<option value="' + v.id + '">' + v.version + "（" +
          v.image_count + " 张）</option>").join("") + "</select></label>",
        '<button data-close="1">取消</button><button class="primary" id="an-go">开始导入</button>');
      el("an-go").onclick = async function () {
        const versionId = value("an-version");
        Modal.close();
        try {
          const r = await API.post("/api/v1/datasets/versions/" + versionId + "/import",
            { source_dir: value("an-dir"), format: value("an-format") });
          const data = r.data || {};
          Toast.ok("导入完成：成功 " + (data.imported || data.matched || 0) +
            "，跳过 " + (data.skipped || 0));
          const v = await API.post("/api/v1/datasets/versions/" + versionId + "/validate", {});
          showValidation(v.data);
        } catch (error) { handleError(error); }
      };
    };
    html("ann-hint",
      '<div class="hint">平台当前不提供浏览器内的在线拉框 / 涂抹标注（列为后续迭代），' +
      "标注工作沿用 LabelMe / LabelImg 等开源工具完成后导入。" +
      "平台负责标注的导入、完整性校验、版本管理与可视化浏览。</div>");
  };

  /* ================================================================
     模型管理
     ================================================================ */
  let modelPage = 1;
  Pages.models = async function () {
    html("filter-bar",
      '<div class="form-row">' +
      '<label class="field"><span>任务类型</span><select id="m-task"><option value="">全部</option>' +
      '<option value="detection">目标检测</option><option value="segmentation">道路分割</option></select></label>' +
      '<label class="field"><span>结构</span><input id="m-arch" placeholder="如 yolo12m / resunet"></label>' +
      '<label class="field"><span>关键字</span><input id="m-key"></label>' +
      '<label class="field"><span>&nbsp;</span><button class="primary" id="m-apply">查询</button></label>' +
      "</div>");
    el("m-apply").onclick = function () { modelPage = 1; loadModels(); };

    html("page-actions",
      '<button class="primary" id="btn-scan">扫描历史产物</button>' +
      '<button id="btn-model-stats">仓库统计</button>' +
      '<button id="btn-compare-model">对比选中模型</button>');

    el("btn-scan").onclick = function () {
      Modal.open("扫描历史模型产物",
        '<p class="muted small">从两个原项目的训练产物目录中提取权重与指标，生成模型卡片。</p>' +
        '<label class="field"><span>项目 A 的 runs 目录</span><input id="sc-runs" ' +
        'placeholder="E:\\2Project\\2026dachuang\\Yolo_dis_v3源代码\\runs"></label>' +
        '<label class="field"><span>项目 B 的 models 目录（*.pth）</span><input id="sc-seg" ' +
        'placeholder="E:\\2Project\\2026dachuang\\Unet\\models"></label>' +
        '<label class="field"><span>项目 B 的 best_model 目录（含指标 CSV，可选）</span>' +
        '<input id="sc-best" placeholder="E:\\2Project\\2026dachuang\\Unet\\best_model"></label>' +
        '<label class="field"><span class="inline"><input type="checkbox" id="sc-platform" checked> ' +
        "同时扫描平台自身模型目录</span></label>",
        '<button data-close="1">取消</button><button class="primary" id="sc-go">开始扫描</button>');
      el("sc-go").onclick = async function () {
        const body = { scan_platform: checked("sc-platform"), project_id: API.projectId() };
        if (value("sc-runs")) body.runs_dir = value("sc-runs");
        if (value("sc-seg")) body.seg_models_dir = value("sc-seg");
        if (value("sc-best")) body.best_model_dir = value("sc-best");
        Modal.close();
        Toast.info("扫描中...");
        try {
          const r = await API.post("/api/v1/models/scan-artifacts", body);
          const data = r.data || {};
          const summary = Object.keys(data).map(function (key) {
            const v = data[key] || {};
            return key + ": 导入 " + (v.imported || 0) + " / 跳过 " + (v.skipped || 0);
          }).join("\n");
          Modal.open("扫描结果", "<pre class='mono'>" + Fmt.escape(summary) + "</pre>");
          loadModels();
        } catch (error) { handleError(error, "扫描失败"); }
      };
    };
    el("btn-model-stats").onclick = async function () {
      try {
        const r = await API.get("/api/v1/models/stats");
        const s = r.data || {};
        Modal.open("模型仓库统计", UI.defList([
          ["模型总数", s.total], ["含权重文件", s.with_weights],
          ["权重总体积", s.total_size],
          ["检测模型", ((s.by_task || {}).detection || {}).count],
          ["分割模型", ((s.by_task || {}).segmentation || {}).count],
        ]) + "<h4>按结构分布</h4>" +
          UI.table([{ title: "结构", key: "arch" }, { title: "数量", key: "n", num: true }],
            Object.keys(s.by_arch || {}).map(k => ({ arch: k, n: s.by_arch[k] })),
            { scroll: false }));
      } catch (error) { handleError(error); }
    };
    el("btn-compare-model").onclick = async function () {
      const ids = UI.checkedValues("model-check");
      if (ids.length < 2) return Toast.warn("请勾选至少 2 个模型");
      const r = await API.get("/api/v1/models/compare", { ids: ids.join(",") });
      const data = r.data || {};
      Modal.open("模型对比",
        UI.table([{ title: "模型", key: "name" }, { title: "结构", key: "arch" }].concat(
          (data.dimensions || []).map(d => ({ title: d, num: true,
            render: row => Fmt.num((row.metrics || {})[d], 4) }))),
          data.items || [], { scroll: false }));
    };
    await loadModels();
  };

  async function loadModels() {
    UI.loading("model-table");
    const d = await API.get("/api/v1/models", {
      page: modelPage, page_size: 25, task_type: value("m-task"),
      arch: value("m-arch"), keyword: value("m-key"),
    });
    html("model-table", UI.table([
      { title: '<input type="checkbox" id="model-all">', render: r =>
        '<input type="checkbox" name="model-check" value="' + r.id + '">' },
      { title: "模型", render: r => '<a href="#" data-card="' + r.id + '">' + Fmt.escape(r.name) + "</a>" },
      { title: "任务", render: r => Fmt.taskTag(r.task_type) },
      { title: "结构 / 规模", render: r => Fmt.escape((r.arch || "-") + " / " + (r.variant || "-")) },
      { title: "来源", render: r => '<span class="tag">' + Fmt.escape(r.weight_origin || "-") + "</span>" },
      { title: "mAP50 / MIoU", num: true, render: r => {
        const m = r.metrics || {};
        if (m.mAP50 !== undefined && m.mAP50 !== null) return Fmt.num(m.mAP50, 4);
        if (m.miou !== undefined && m.miou !== null) return Fmt.num(m.miou, 4);
        return "-";
      } },
      { title: "权重", render: r => r.exists ? '<span class="tag ok">已就位</span>' : '<span class="tag warn">缺失</span>' },
      { title: "大小", num: true, key: "size_human" },
      { title: "操作", render: r => '<button class="sm" data-export="' + r.id + '">导出ONNX</button> ' +
        '<button class="sm danger" data-mdel="' + r.id + '">删除</button>' },
    ], d.data || [], { empty: "暂无模型，点击「扫描历史产物」导入原项目权重" }) +
      UI.pager(d.total, d.page, d.page_size, function (p) { modelPage = p; loadModels(); }));

    const all = el("model-all");
    if (all) all.onclick = function () {
      document.querySelectorAll('input[name="model-check"]').forEach(c => { c.checked = all.checked; });
    };
    document.querySelectorAll("[data-card]").forEach(function (btn) {
      btn.onclick = function (e) { e.preventDefault(); showModelCard(btn.dataset.card); };
    });
    document.querySelectorAll("[data-export]").forEach(function (btn) {
      btn.onclick = async function () {
        try {
          Toast.info("导出中...");
          const r = await API.post("/api/v1/models/" + btn.dataset.export + "/export", { format: "onnx" });
          Toast.ok("已导出：" + ((r.data || {}).path || ""));
        } catch (error) { handleError(error, "导出失败"); }
      };
    });
    document.querySelectorAll("[data-mdel]").forEach(function (btn) {
      btn.onclick = async function () {
        try { await API.del("/api/v1/models/" + btn.dataset.mdel); Toast.ok("已删除"); loadModels(); }
        catch (error) { handleError(error); }
      };
    });
  }

  async function showModelCard(modelId) {
    const d = await API.get("/api/v1/models/" + modelId);
    const m = d.data || {};
    Modal.open("模型卡片 · " + Fmt.escape(m.name),
      UI.defList([
        ["任务类型", (API.dict().task_labels || {})[m.task_type]],
        ["框架", m.framework], ["结构", m.arch], ["规模 / 训练量", m.variant],
        ["输入尺寸", m.input_size], ["类别数", m.num_classes],
        ["权重来源", m.weight_origin], ["原始文件名", m.source_filename],
        ["权重路径", m.weights_uri],
        ["权重 SHA-256", (m.weights_sha256 || "-").slice(0, 32) + "..."],
        ["文件大小", m.size_human], ["被引用次数", m.usage],
        ["指标", Object.keys(m.metrics || {}).length ? JSON.stringify(m.metrics) : "-"],
        ["已知限制", m.notes || "（未填写）"],
      ]));
  }

  /* ================================================================
     训练实验
     ================================================================ */
  Pages.training = async function () {
    const options = await API.get("/api/v1/training/options");
    const o = options.data || {};
    html("train-form",
      '<div class="form-row">' +
      '<label class="field"><span>任务类型</span><select id="tr-task">' +
      '<option value="detection">目标检测（YOLO / Ultralytics）</option>' +
      '<option value="segmentation">道路分割（U-Net 家族）</option></select></label>' +
      '<label class="field"><span>数据集版本</span><select id="tr-dataset"><option value="">（可选）</option></select></label>' +
      '<label class="field"><span>模型 / 结构</span><select id="tr-arch"></select></label>' +
      "</div>" +
      '<div class="form-row">' +
      '<label class="field"><span>epochs</span><input id="tr-epochs" type="number" value="' + (o.epochs || 100) + '"></label>' +
      '<label class="field"><span>batch</span><input id="tr-batch" type="number" value="' + (o.batch || 8) + '"></label>' +
      '<label class="field"><span>imgsz</span><input id="tr-imgsz" type="number" value="' + (o.imgsz || 640) + '"></label>' +
      '<label class="field"><span>lr</span><input id="tr-lr" type="number" step="0.0001" value="' + (o.lr || 0.0003) + '"></label>' +
      '<label class="field"><span>随机种子</span><input id="tr-seed" type="number" value="' + (o.seed || 42) + '"></label>' +
      "</div>" +
      '<div class="hint">训练任务进入任务队列串行执行；平台负责调度、进度推送、指标归档与复现清单生成；' +
      "训练实现复用原项目脚本（Ultralytics 高层 API / 自定义 PyTorch 训练循环）。</div>");
    html("page-actions", '<button class="primary" id="btn-train">发起训练</button>');

    const yolo = o.yolo_configs || [];
    const seg = o.seg_architectures || [];
    function refreshArch() {
      const task = value("tr-task");
      const items = task === "detection"
        ? yolo.map(c => ({ key: c.key, label: c.key + "（v" + c.version + " / " + c.size + "）" }))
        : seg.map(item => ({ key: item.key,
            label: item.label + (item.available ? "" : "（不可用：" + item.note + "）") }));
      html("tr-arch", items.map(i => '<option value="' + i.key + '">' + Fmt.escape(i.label) + "</option>").join(""));
    }
    el("tr-task").onchange = refreshArch;
    refreshArch();

    const datasets = await API.get("/api/v1/datasets", { page_size: 100 });
    html("tr-dataset", '<option value="">（可选）</option>' +
      (datasets.data || []).map(d => '<option value="' + d.id + '">' + Fmt.escape(d.name) + "</option>").join(""));

    el("btn-train").onclick = async function () {
      try {
        const r = await API.post("/api/v1/training/start", {
          project_id: API.projectId(), task_type: value("tr-task"),
          dataset_version_id: value("tr-dataset") || null,
          arch: value("tr-arch"), yolo: value("tr-arch"),
          epochs: num("tr-epochs", 100), batch: num("tr-batch", 8),
          imgsz: num("tr-imgsz", 640), lr: num("tr-lr", 0.0003), seed: num("tr-seed", 42),
        });
        Toast.ok("训练任务已入队：" + ((r.data || {}).task_id || ""));
        loadTasks();
      } catch (error) { handleError(error); }
    };

    await loadTasks();
    addPoller(poll(loadTasks, 5000));
  };

  async function loadTasks() {
    const d = await API.get("/api/v1/tasks");
    const data = d.data || {};
    const summary = data.summary || {};
    html("task-summary",
      UI.kpi("排队", summary.queued || 0, "", "accent") +
      UI.kpi("运行中", summary.running || 0, "", "") +
      UI.kpi("已完成", summary.done || 0, "", "seg") +
      UI.kpi("失败", summary.failed || 0, "", "danger"));
    html("task-table", UI.table([
      { title: "任务 ID", render: r => '<span class="mono">' + r.id + "</span>" },
      { title: "类型", key: "task_type" },
      { title: "实验", render: r => r.experiment_id
        ? '<a href="/experiments/' + r.experiment_id + '">' + r.experiment_id.slice(0, 14) + "...</a>" : "-" },
      { title: "状态", render: r => Fmt.statusTag(r.status) },
      { title: "入队", render: r => Fmt.time(r.enqueued_at) },
      { title: "开始", render: r => Fmt.time(r.started_at) },
      { title: "完成", render: r => Fmt.time(r.finished_at) },
      { title: "错误", render: r => r.error ? '<span class="error-text">' + Fmt.escape(r.error) + "</span>" : "-" },
    ], data.items || [], { empty: "暂无任务" }));
  }

  /* ================================================================
     误差分析
     ================================================================ */
  let analysisPage = 1;
  Pages.analysis = async function () {
    html("filter-bar", buildAnalysisFilters());
    el("an-apply").onclick = function () { analysisPage = 1; loadAnalysis(); };
    el("an-reset").onclick = function () { location.reload(); };
    html("page-actions",
      '<button id="btn-save-analysis">保存分析快照</button>' +
      '<button id="btn-export-records">导出记录 CSV</button>');
    el("btn-save-analysis").onclick = async function () {
      try {
        const q = filterQuery();
        await API.post("/api/v1/analysis/summary" + (q ? q + "&" : "?") + "save=1", {});
        Toast.ok("分析快照已保存");
      } catch (error) { handleError(error); }
    };
    el("btn-export-records").onclick = function () {
      window.open("/api/v1/ranging/records/export" + filterQuery(), "_blank");
    };
    await loadAnalysis();
    await loadCalibrationPanel();
  };

  function filterQuery() {
    const params = new URLSearchParams();
    [["an-task", "task_type"], ["an-model", "model_name"], ["an-category", "category"],
     ["an-distance", "distance_plan_m"], ["an-size", "size_group"], ["an-illum", "illumination"]]
      .forEach(function (pair) {
        const e = el(pair[0]);
        if (e && e.value) params.set(pair[1], e.value);
      });
    const s = params.toString();
    return s ? "?" + s : "";
  }

  function buildAnalysisFilters() {
    const dict = API.dict();
    return '<div class="form-row">' +
      '<label class="field"><span>任务类型</span><select id="an-task"><option value="">全部</option>' +
      Object.keys(dict.task_labels || {}).map(k =>
        '<option value="' + k + '">' + dict.task_labels[k] + "</option>").join("") + "</select></label>" +
      '<label class="field"><span>类别</span><select id="an-category"><option value="">全部</option>' +
      (dict.detection_classes || []).map(c => '<option value="' + c + '">' + c + "</option>").join("") +
      "</select></label>" +
      '<label class="field"><span>标准距离点</span><select id="an-distance"><option value="">全部</option>' +
      (dict.distance_plan_m || []).map(d => '<option value="' + d + '">' + d + " m</option>").join("") +
      "</select></label>" +
      '<label class="field"><span>尺寸组</span><select id="an-size"><option value="">全部</option>' +
      '<option value="large">大目标</option><option value="small">小目标</option></select></label>' +
      '<label class="field"><span>光照</span><select id="an-illum"><option value="">全部</option>' +
      (dict.illuminations || []).map(i => '<option value="' + i.key + '">' + i.label + "</option>").join("") +
      "</select></label>" +
      '<label class="field"><span>模型</span><input id="an-model" placeholder="模型名"></label>' +
      '<label class="field"><span>&nbsp;</span><button class="primary" id="an-apply">应用筛选</button> ' +
      '<button id="an-reset">重置</button></label></div>';
  }

  async function loadAnalysis() {
    const query = {};
    if (value("an-task")) query.task_type = value("an-task");
    if (value("an-category")) query.category = value("an-category");
    if (value("an-distance")) query.distance_plan_m = value("an-distance");
    if (value("an-size")) query.size_group = value("an-size");
    if (value("an-illum")) query.illumination = value("an-illum");
    if (value("an-model")) query.model_name = value("an-model");

    const d = await API.get("/api/v1/analysis/summary", query);
    const s = d.data || {};
    html("analysis-kpi",
      UI.kpi("样本数", Fmt.int(s.sample_count), "落库测距记录", "accent") +
      UI.kpi("有效样本", Fmt.int(s.valid_count), "含真值", "") +
      UI.kpi("平均绝对误差", Fmt.m(s.mae_m), "MAE", "detect") +
      UI.kpi("均方根误差", Fmt.m(s.rmse_m), "RMSE", "range") +
      UI.kpi("平均相对误差", Fmt.pct(s.mre), "MRE", "") +
      UI.kpi("中位误差", Fmt.m(s.median_error_m), "", "") +
      UI.kpi("最大误差", Fmt.m(s.max_error_m), "", "danger") +
      UI.kpi("命中率(<=5%)", Fmt.pct(s.hit_rate), "", "seg") +
      UI.kpi("缺检率", Fmt.pct(s.miss_rate), "检测帧口径", "danger") +
      UI.kpi("误差标准差", Fmt.m(s.std_error_m), "", ""));

    const dist = s.by_distance || [];
    if (dist.length) {
      Chart.lineWithBand("chart-distance",
        dist.map(x => (x.key === null ? "-" : Number(x.key).toFixed(0) + "m")),
        dist.map(x => x.mae_m), dist.map(x => x.std_error_m || 0));
    } else { Chart.empty("chart-distance", "暂无带真值的分组数据"); }

    const cat = (s.by_category || []).slice(0, 12);
    if (cat.length) Chart.bar("chart-category", cat.map(x => x.key || "-"), cat.map(x => x.mae_m), "MAE(m)");
    else Chart.empty("chart-category", "暂无按类别分组数据");

    const heat = s.heatmap || {};
    Chart.heatmap("chart-heatmap", heat.x || [], heat.y || [], heat.z || []);

    const size = s.by_size || [];
    if (size.length) Chart.bar("chart-size", size.map(x => Fmt.sizeGroup(x.key)), size.map(x => x.mae_m), "MAE(m)");
    else Chart.empty("chart-size", "暂无按尺寸组分组数据");

    const illum = s.by_illumination || [];
    if (illum.length) Chart.bar("chart-illum", illum.map(x => Fmt.illum(x.key)), illum.map(x => x.mae_m), "MAE(m)");
    else Chart.empty("chart-illum", "暂无按光照分组数据");

    html("table-model", UI.table([
      { title: "模型", key: "key" },
      { title: "样本", key: "count", num: true },
      { title: "有效", key: "valid_count", num: true },
      { title: "MAE(m)", num: true, render: r => Fmt.num(r.mae_m, 3) },
      { title: "RMSE(m)", num: true, render: r => Fmt.num(r.rmse_m, 3) },
      { title: "相对误差", num: true, render: r => Fmt.pct(r.mre) },
      { title: "最大误差(m)", num: true, render: r => Fmt.num(r.max_error_m, 3) },
    ], s.by_model || [], { empty: "暂无按模型分组数据" }));

    html("table-outliers", UI.table([
      { title: "类别", key: "category" },
      { title: "模型", key: "model_name" },
      { title: "测量(m)", num: true, render: r => Fmt.num(r.depth_m, 3) },
      { title: "真值(m)", num: true, render: r => Fmt.num(r.distance_actual_m, 3) },
      { title: "绝对误差(m)", num: true, render: r => Fmt.num(r.abs_error_m, 3) },
      { title: "相对误差", num: true, render: r => Fmt.pct(r.rel_error) },
      { title: "光照", render: r => Fmt.illum(r.illumination) },
      { title: "样本", render: r => r.media_uri
        ? '<a target="_blank" href="/api/v1/media/frame?uri=' + encodeURIComponent(r.media_uri) + '">查看</a>' : "-" },
    ], s.outliers || [], { empty: "暂无异常样本" }));

    await loadRecordsTable();
  }

  async function loadRecordsTable() {
    const query = { page: analysisPage, page_size: 25 };
    if (value("an-task")) query.task_type = value("an-task");
    if (value("an-category")) query.category = value("an-category");
    if (value("an-size")) query.size_group = value("an-size");
    if (value("an-illum")) query.illumination = value("an-illum");
    const d = await API.get("/api/v1/ranging/records", query);
    html("table-records", UI.table(recordColumns().concat([
      { title: "操作", render: r => '<button class="sm" data-edit="' + r.id + '">改真值</button> ' +
        '<button class="sm danger" data-del="' + r.id + '">删除</button>' },
    ]), d.data || [], { empty: "暂无记录" }) +
      UI.pager(d.total, d.page, d.page_size, function (p) { analysisPage = p; loadRecordsTable(); }));
    document.querySelectorAll("[data-edit]").forEach(function (btn) {
      btn.onclick = function () { editRecordModal(btn.dataset.edit); };
    });
    document.querySelectorAll("[data-del]").forEach(function (btn) {
      btn.onclick = async function () {
        if (!confirm("确认删除该记录？")) return;
        try { await API.del("/api/v1/ranging/records/" + btn.dataset.del); loadAnalysis(); }
        catch (error) { handleError(error); }
      };
    });
  }

  function editRecordModal(recordId) {
    Modal.open("修正测距记录",
      '<div class="form-row">' +
      '<label class="field"><span>实际距离(m)</span><input id="er-actual" type="number" step="0.01"></label>' +
      '<label class="field"><span>测量手段</span><select id="er-src">' +
      '<option value="tape">卷尺</option><option value="laser">激光测距仪</option>' +
      '<option value="manual">人工估计</option></select></label></div>' +
      '<label class="field"><span>光照条件</span>' +
      UI.select("er-illum", API.dict().illuminations || [], "") + "</label>" +
      '<label class="field"><span>备注</span><input id="er-notes"></label>',
      '<button data-close="1">取消</button><button class="primary" id="er-save">保存</button>');
    el("er-save").onclick = async function () {
      try {
        await API.patch("/api/v1/ranging/records/" + recordId, {
          distance_actual_m: value("er-actual") || null,
          actual_source: value("er-src"), illumination: value("er-illum") || null,
          notes: value("er-notes"),
        });
        Toast.ok("已更新");
        Modal.close();
        loadAnalysis();
      } catch (error) { handleError(error); }
    };
  }

  async function loadCalibrationPanel() {
    const list = await API.get("/api/v1/analysis/calibrations");
    const data = list.data || {};
    const items = data.items || [];
    html("calib-panel",
      '<div class="form-row">' +
      '<label class="field"><span>校正方法</span><select id="cal-method">' +
      '<option value="linear">线性 a*x+b</option>' +
      '<option value="ratio">比率 x/(1+k*x)</option>' +
      '<option value="piecewise">分段（近 / 中 / 远）</option></select></label>' +
      '<label class="field"><span>分段边界(m)</span><input id="cal-bounds" value="5,15"></label>' +
      '<label class="field"><span>&nbsp;</span><button class="primary" id="cal-run">拟合并评估</button></label>' +
      "</div>" +
      '<div class="hint">校正后在测距流程中生效（测距实验页会显示「已启用校正」）。' +
      "若校正后 RMSE 未下降，平台会标注为「不建议启用」。</div>" +
      '<div id="calib-result"></div>' +
      '<h4 style="margin-top:14px">已有校正方案</h4>' +
      (items.length ? UI.table([
        { title: "名称", key: "name" }, { title: "方法", key: "method" },
        { title: "样本", key: "sample_count", num: true },
        { title: "RMSE 前", num: true, render: r => Fmt.num(r.rmse_before, 4) },
        { title: "RMSE 后", num: true, render: r => Fmt.num(r.rmse_after, 4) },
        { title: "状态", render: r => r.is_active ? '<span class="tag ok">已启用</span>' : "" },
        { title: "操作", render: r => '<button class="sm" data-activate="' + r.id + '">启用</button>' },
      ], items, { scroll: false }) : '<div class="empty">尚无校正方案</div>'));

    el("cal-run").onclick = async function () {
      const bounds = value("cal-bounds").split(",").map(x => parseFloat(x)).filter(x => !isNaN(x));
      try {
        const r = await API.post("/api/v1/analysis/calibrate", {
          method: value("cal-method"), boundaries_m: bounds.length ? bounds : undefined,
          category: value("an-category") || undefined, task_type: value("an-task") || undefined,
        });
        const c = r.data || {};
        html("calib-result",
          '<div class="inline" style="margin:10px 0">' +
          '<span class="tag ' + (c.recommended ? "ok" : "err") + '">' +
          (c.recommended ? "建议启用" : "不建议启用（RMSE 未下降）") + "</span>" +
          "<span class='muted small'>样本 " + c.sample_count + " 条，RMSE " +
          Fmt.num(c.rmse_before, 4) + " 到 " + Fmt.num(c.rmse_after, 4) + "</span></div>" +
          '<pre class="mono" style="max-height:140px;overflow:auto">' +
          Fmt.escape(JSON.stringify(c.params, null, 2)) + "</pre>");
        const ba = c.before_after || {};
        if ((ba.before || []).length) {
          Chart.scatter("chart-calibration", ba.before,
            (ba.after || []).map(x => ({ measured: x.measured, corrected: x.corrected, actual: x.actual })));
        }
        loadCalibrationPanel();
      } catch (error) { handleError(error, "拟合失败"); }
    };
    document.querySelectorAll("[data-activate]").forEach(function (btn) {
      btn.onclick = async function () {
        try {
          await API.post("/api/v1/analysis/calibrations/" + btn.dataset.activate + "/activate", {});
          Toast.ok("校正方案已启用");
          loadCalibrationPanel();
        } catch (error) { handleError(error); }
      };
    });
    const chartEl = el("chart-calibration");
    if (chartEl && !chartEl.innerHTML) {
      Chart.empty("chart-calibration", "点击「拟合并评估」查看校正前后对比");
    }
  }

  /* ================================================================
     可视化大屏
     ================================================================ */
  Pages.bigscreen = async function () {
    html("page-actions", '<button id="bs-refresh">立即刷新</button>' +
      '<button id="bs-full">全屏 (F)</button>' +
      '<button onclick="location.href=\'/analysis\'">进入分析页</button>');
    el("bs-refresh").onclick = loadBigscreen;
    el("bs-full").onclick = function () {
      if (document.fullscreenElement) document.exitFullscreen();
      else document.documentElement.requestFullscreen();
    };
    document.addEventListener("keydown", function (e) {
      if (e.key === "f" || e.key === "F") el("bs-full").click();
    });
    await loadBigscreen();
    addPoller(poll(loadBigscreen, 15000));
  };

  async function loadBigscreen() {
    const d = await API.get("/api/v1/dashboard/overview");
    const data = d.data || {};
    const kpi = data.kpi || {};
    html("bs-kpi",
      UI.kpi("实验总数", Fmt.int(kpi.experiments), "含历史迁移实验", "accent") +
      UI.kpi("模型权重", Fmt.int(kpi.models), "检测 + 分割", "") +
      UI.kpi("图像样本", Fmt.int(kpi.images), "数据集版本合计", "seg") +
      UI.kpi("测距记录", Fmt.int(kpi.ranging_records), "落库条数", "range") +
      UI.kpi("平均绝对误差", Fmt.m(data.mae_m), "全量口径", "detect") +
      UI.kpi("命中率 <=5%", Fmt.pct(data.hit_rate), "相对误差", "seg"));

    const dist = data.by_distance || [];
    if (dist.length) {
      Chart.lineWithBand("bs-chart-distance",
        dist.map(x => (x.key === null ? "-" : Number(x.key).toFixed(0) + "m")),
        dist.map(x => x.mae_m), dist.map(x => x.std_error_m || 0));
    } else { Chart.empty("bs-chart-distance", "暂无数据"); }

    html("bs-leaderboard", UI.table([
      { title: "#", render: (r, i) => i + 1 }, { title: "模型", key: "key" },
      { title: "样本", key: "count", num: true },
      { title: "MAE(m)", num: true, render: r => Fmt.num(r.mae_m, 3) },
      { title: "MIoU", num: true, render: r => Fmt.num(r.miou, 4) },
    ], (data.leaderboard || []).slice(0, 10), { empty: "暂无评测数据", scroll: false }));

    const heat = data.heatmap || {};
    Chart.heatmap("bs-chart-heatmap", heat.x || [], heat.y || [], heat.z || []);

    const illum = data.by_illumination || [];
    if (illum.length) Chart.bar("bs-chart-illum", illum.map(x => Fmt.illum(x.key)), illum.map(x => x.mae_m), "MAE(m)");
    else Chart.empty("bs-chart-illum", "暂无数据");

    html("bs-gallery", UI.gallery((data.outliers || []).slice(0, 8),
      item => item.media_uri ? "/api/v1/media/thumb?uri=" + encodeURIComponent(item.media_uri) + "&width=240" : "",
      item => (item.category || "-") + " 误差 " + Fmt.m(item.abs_error_m)));

    const metrics = await API.get("/api/v1/system/metrics");
    const m = metrics.data || {};
    const cam = m.camera || {};
    html("bs-health", UI.statusLine([
      ["相机", cam.running ? (cam.backend === "realsense" ? "RealSense" : "合成场景") : "未启动"],
      ["帧数", Fmt.int(cam.frames)], ["实际帧率", cam.fps_actual],
      ["GPU 显存", m.gpu ? m.gpu.mem_used_gb + " / " + m.gpu.mem_total_gb + " GB" : "未启用"],
      ["磁盘剩余", m.disk ? m.disk.free_gb + " GB" : "-"],
      ["排队任务", ((m.queue || {}).counts || {}).queued || 0],
    ]) + '<div class="small muted">最后刷新：' + new Date().toLocaleTimeString() + "</div>");

    const experiments = await API.get("/api/v1/experiments", { page_size: 6 });
    html("bs-feed", (experiments.data || []).map(function (e) {
      return '<div class="small" style="padding:3px 0">' + Fmt.time(e.created_at) + " " +
        Fmt.taskTag(e.task_type) + " " + Fmt.escape(e.name) + " " + Fmt.statusTag(e.status) + "</div>";
    }).join("") || '<div class="empty">暂无实验</div>');
  }

  /* ================================================================
     报告中心
     ================================================================ */
  Pages.reports = async function () {
    const params = new URLSearchParams(location.search);
    const experimentId = params.get("experiment_id") || "";
    const templates = await API.get("/api/v1/reports/templates");
    const experiments = await API.get("/api/v1/experiments", { page_size: 100 });

    html("report-form",
      '<div class="form-row">' +
      '<label class="field"><span>模板</span><select id="rp-template">' +
      (templates.data.items || []).map(t => '<option value="' + t.key + '">' + t.label +
        "（" + t.description + "）</option>").join("") + "</select></label>" +
      '<label class="field"><span>格式</span><select id="rp-format">' +
      '<option value="pdf">PDF</option><option value="md">Markdown</option>' +
      '<option value="csv">CSV</option><option value="json">JSON</option></select></label>' +
      '<label class="field"><span>关联实验</span><select id="rp-experiment"><option value="">（不关联）</option>' +
      (experiments.data || []).map(e => '<option value="' + e.id + '"' +
        (e.id === experimentId ? " selected" : "") + ">" + Fmt.escape(e.name) + "</option>").join("") +
      "</select></label>" +
      '<label class="field"><span>报告名称</span><input id="rp-name" placeholder="留空自动命名"></label>' +
      '<label class="field"><span>&nbsp;</span><button class="primary" id="rp-generate">生成报告</button></label>' +
      "</div>" +
      '<div class="hint">PDF 使用 ReportLab 渲染（自动加载中文字体）；' +
      "若环境缺少字体或依赖，平台会自动降级为 Markdown 输出。</div>");
    html("page-actions", '<button id="btn-load-reports">刷新列表</button>');
    el("btn-load-reports").onclick = loadReports;
    el("rp-generate").onclick = async function () {
      try {
        Toast.info("生成中...");
        const r = await API.post("/api/v1/reports", {
          name: value("rp-name"), template: value("rp-template"), format: value("rp-format"),
          experiment_id: value("rp-experiment") || null, project_id: API.projectId(),
        });
        Toast.ok("报告已生成：" + ((r.data || {}).name || ""));
        loadReports();
      } catch (error) { handleError(error, "生成失败"); }
    };
    await loadReports();
  };

  async function loadReports() {
    const d = await API.get("/api/v1/reports", { page_size: 50 });
    html("report-list", UI.table([
      { title: "报告", key: "name" },
      { title: "模板", key: "template" },
      { title: "格式", key: "format" },
      { title: "大小", key: "file_human" },
      { title: "状态", render: r => Fmt.statusTag(r.status === "done" ? "done" : "failed") },
      { title: "生成时间", render: r => Fmt.time(r.created_at) },
      { title: "操作", render: r => '<a class="btn sm" href="/api/v1/reports/' + r.id +
        '/download">下载</a> <button class="sm" data-preview="' + r.id + '">预览</button> ' +
        '<button class="sm" data-share="' + r.id + '">分享</button> ' +
        '<button class="sm danger" data-delreport="' + r.id + '">删除</button>' },
    ], d.data || [], { empty: "暂无报告" }));

    document.querySelectorAll("[data-preview]").forEach(function (btn) {
      btn.onclick = async function () {
        const response = await fetch("/api/v1/reports/" + btn.dataset.preview + "/preview",
          { credentials: "same-origin" });
        const text = await response.text();
        Modal.open("报告预览（Markdown）",
          "<pre class='mono' style='max-height:66vh;overflow:auto;white-space:pre-wrap'>" +
          Fmt.escape(text) + "</pre>");
      };
    });
    document.querySelectorAll("[data-share]").forEach(function (btn) {
      btn.onclick = async function () {
        const r = await API.post("/api/v1/reports/" + btn.dataset.share + "/share", {});
        Modal.open("分享链接", '<p class="small">只读链接（无需登录）：</p><input value="' +
          location.origin + r.data.url + '" readonly>');
      };
    });
    document.querySelectorAll("[data-delreport]").forEach(function (btn) {
      btn.onclick = async function () {
        try { await API.del("/api/v1/reports/" + btn.dataset.delreport); loadReports(); }
        catch (error) { handleError(error); }
      };
    });
  }

  /* ================================================================
     系统设置
     ================================================================ */
  Pages.settings = async function () {
    const results = await Promise.all([
      API.get("/api/v1/system/info"),
      API.get("/api/v1/system/settings"),
      API.get("/api/v1/users").catch(() => ({ data: { items: [], roles: [] } })),
      API.get("/api/v1/storage/tree").catch(() => ({ data: { items: [] } })),
    ]);
    const i = results[0].data || {};
    html("sys-info", UI.defList([
      ["平台名称", i.app], ["版本 / 代号", (i.version || "") + " / " + (i.code_name || "")],
      ["Python", i.python], ["操作系统", i.platform],
      ["PyTorch", i.torch || "未安装"], ["CUDA", i.cuda_available ? (i.cuda + " / " + i.gpu) : "未启用"],
      ["Ultralytics", i.ultralytics || "未安装"], ["OpenCV", i["opencv-python"] || "未安装"],
      ["pyrealsense2", i.pyrealsense2 || "未安装"], ["ReportLab", i.reportlab || "未安装"],
      ["timm（UNetV2 依赖）", i.timm || "未安装"],
      ["存储根目录", i.storage_root], ["元数据库", i.db_path + "（" + i.db_size + "）"],
    ]));

    const s = results[1].data || {};
    const cam = s.camera || {}, rng = s.ranging || {}, det = s.detection || {}, seg = s.segmentation || {};
    html("sys-settings", "<h4>相机</h4>" + UI.defList([
      ["分辨率", (cam.width || "-") + " x " + (cam.height || "-")],
      ["帧率", cam.fps], ["后端", cam.backend],
      ["滤波链", (cam.filters || []).join(", ") || "关闭"],
    ]) + "<h4>测距</h4>" + UI.defList([
      ["默认策略", rng.default_strategy], ["采样核", rng.kernel],
      ["采样点数", rng.sample_points],
      ["量程", (rng.min_range_m || "-") + " 到 " + (rng.max_range_m || "-") + " m"],
      ["标准距离点", (rng.distance_plan_m || []).join(", ") + " m"],
      ["左右分离 / 反投影", (rng.road_split ? "开" : "关") + " / " + (rng.reproject ? "开" : "关")],
      ["走廊参考深度", (rng.corridor_ref_depth_m || "-") + " m"],
    ]) + "<h4>检测 / 分割</h4>" + UI.defList([
      ["置信度 / IoU", (det.default_conf || "-") + " / " + (det.default_iou || "-")],
      ["权重模式", det.weight_mode],
      ["分割阈值", seg.default_threshold],
      ["分割输入尺寸", (seg.input_width || "-") + " x " + (seg.input_height || "-")],
    ]) + "<h4>存储目录</h4>" +
      UI.table([{ title: "目录", key: "key" }, { title: "路径", key: "path" },
                { title: "文件数", key: "files", num: true }, { title: "占用", key: "size", num: true }],
        (results[3].data || {}).items || [], { scroll: false }) +
      '<div class="hint">配置位于 configs/*.yaml，可用环境变量 VISIONLAB_SECTION__KEY 覆盖（本页只读展示）。</div>');

    const usersData = results[2].data || {};
    const roleList = usersData.roles || [];
    html("user-table", UI.table([
      { title: "用户名", key: "username" }, { title: "显示名", key: "display_name" },
      { title: "角色", render: r => {
        const found = roleList.filter(x => x.key === r.role)[0];
        return '<span class="tag info">' + Fmt.escape(found ? found.label : r.role) + "</span>";
      } },
      { title: "状态", render: r => r.is_active ? '<span class="tag ok">启用</span>' : '<span class="tag err">禁用</span>' },
      { title: "最后登录", render: r => Fmt.time(r.last_login_at) },
      { title: "操作", render: r => '<button class="sm" data-reset="' + r.id + '">重置密码</button>' },
    ], usersData.items || [], { empty: "无用户数据（可能无权限）" }));

    html("page-actions", '<button class="primary" id="btn-add-user">新建用户</button>' +
      '<button id="btn-self-check">一键自检</button>');
    el("btn-add-user").onclick = function () {
      Modal.open("新建用户",
        '<label class="field"><span>用户名</span><input id="u-name"></label>' +
        '<label class="field"><span>密码</span><input id="u-pwd" type="password"></label>' +
        '<label class="field"><span>角色</span><select id="u-role">' +
        roleList.map(r => '<option value="' + r.key + '">' + r.label + "</option>").join("") +
        "</select></label>" +
        '<label class="field"><span>显示名</span><input id="u-display"></label>',
        '<button data-close="1">取消</button><button class="primary" id="u-save">创建</button>');
      el("u-save").onclick = async function () {
        try {
          await API.post("/api/v1/users", {
            username: value("u-name"), password: value("u-pwd"),
            role: value("u-role"), display_name: value("u-display"),
          });
          Toast.ok("用户已创建");
          Modal.close();
          Pages.settings();
        } catch (error) { handleError(error); }
      };
    };
    el("btn-self-check").onclick = async function () {
      showSelfCheck((await API.post("/api/v1/system/self-check", {})).data);
    };
    document.querySelectorAll("[data-reset]").forEach(function (btn) {
      btn.onclick = function () {
        Modal.open("重置密码",
          '<label class="field"><span>新密码（至少 6 位）</span><input id="r-pwd" type="password"></label>',
          '<button data-close="1">取消</button><button class="primary" id="r-save">保存</button>');
        el("r-save").onclick = async function () {
          try {
            await API.post("/api/v1/users/" + btn.dataset.reset + "/reset-password",
              { new_password: value("r-pwd") });
            Toast.ok("密码已重置");
            Modal.close();
          } catch (error) { handleError(error); }
        };
      };
    });
  };

  function showSelfCheck(report) {
    if (!report) return;
    const s = report.summary || {};
    Modal.open("系统自检",
      UI.statusLine([["检查项", s.total], ["通过", s.passed], ["错误", s.errors], ["警告", s.warnings]]) +
      "<p>" + Fmt.escape(report.conclusion || "") + "</p>" +
      UI.table([
        { title: "检查项", key: "name" },
        { title: "结果", render: r => r.ok ? '<span class="tag ok">通过</span>'
          : '<span class="tag ' + (r.level === "warning" ? "warn" : "err") + '">' +
            (r.level === "warning" ? "警告" : "错误") + "</span>" },
        { title: "说明", key: "detail" },
      ], report.checks || [], { scroll: false }));
  }

  /* ================================================================
     日志与监控
     ================================================================ */
  Pages.monitor = async function () {
    html("filter-bar",
      '<div class="form-row">' +
      '<label class="field"><span>日志级别</span><select id="lg-level">' +
      '<option value="">全部</option><option value="INFO">INFO</option>' +
      '<option value="WARNING">WARNING</option><option value="ERROR">ERROR</option></select></label>' +
      '<label class="field"><span>关键字</span><input id="lg-key" placeholder="模块或内容"></label>' +
      '<label class="field"><span>&nbsp;</span><button class="primary" id="lg-apply">查询</button> ' +
      '<button id="lg-auto">自动刷新：开</button></label></div>');
    el("lg-apply").onclick = loadLogs;
    let auto = true;
    el("lg-auto").onclick = function () {
      auto = !auto;
      el("lg-auto").textContent = "自动刷新：" + (auto ? "开" : "关");
    };
    html("page-actions", '<button id="btn-self-check2">一键自检</button>' +
      '<button id="btn-metrics">刷新指标</button>');
    el("btn-self-check2").onclick = async function () {
      showSelfCheck((await API.post("/api/v1/system/self-check", {})).data);
    };
    el("btn-metrics").onclick = loadMetrics;

    async function loadLogs() {
      const d = await API.get("/api/v1/system/logs",
        { level: value("lg-level"), keyword: value("lg-key"), limit: 300 });
      const rows = (d.data || {}).items || [];
      html("log-view", rows.length ? rows.map(function (r) {
        return '<div class="log-line ' + r.level + '"><span class="lv">' + r.level + "</span>" +
          '<span class="muted">' + Fmt.escape(r.time) + "</span> " + Fmt.escape(r.message) + "</div>";
      }).join("") : '<div class="empty">暂无日志</div>');
    }

    async function loadMetrics() {
      const d = await API.get("/api/v1/system/metrics");
      const m = d.data || {};
      const cam = m.camera || {}, gpu = m.gpu || {}, queue = (m.queue || {}).counts || {};
      html("metric-row",
        UI.kpi("相机", cam.running ? "在线" : "离线", cam.backend || "-", cam.running ? "seg" : "danger") +
        UI.kpi("采集帧数", Fmt.int(cam.frames), "累计", "") +
        UI.kpi("实际帧率", Fmt.num(cam.fps_actual, 1), "fps", "") +
        UI.kpi("GPU 显存", gpu.mem_used_gb !== undefined ? gpu.mem_used_gb + " GB" : "未启用",
               gpu.mem_total_gb ? "共 " + gpu.mem_total_gb + " GB" : "", "accent") +
        UI.kpi("磁盘剩余", (m.disk ? m.disk.free_gb : "-") + " GB", "存储根分区", "range") +
        UI.kpi("任务队列", (queue.queued || 0) + " / " + (queue.running || 0), "排队 / 运行", ""));

      const dirs = (m.disk || {}).dirs || {};
      html("storage-row", UI.table([
        { title: "目录", key: "key" },
        { title: "文件数", num: true, render: r => Fmt.int(r.files) },
        { title: "占用", num: true, render: r => Fmt.escape(r.size) },
      ], Object.keys(dirs).map(k => ({ key: k, files: dirs[k].files, size: dirs[k].size })),
        { scroll: false }));
      const tail = await API.get("/api/v1/system/logs/tail", { lines: 40 });
      html("log-tail", "<pre class='mono' style='max-height:240px;overflow:auto;white-space:pre-wrap'>" +
        Fmt.escape((((tail.data || {}).lines) || []).join("\n")) + "</pre>");
    }

    await Promise.all([loadLogs(), loadMetrics()]);
    addPoller(poll(function () { if (auto) return loadLogs(); }, 4000));
    addPoller(poll(loadMetrics, 6000));
  };

  global.VL.Pages = Pages;
})(window);
