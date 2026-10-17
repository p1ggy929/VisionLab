/* ============================================================
   视界智研 —— API 客户端与通用工具
   ============================================================ */
((global) => {
  "use strict";

  const USER_KEY = "visionlab_user";
  const PROJECT_KEY = "visionlab_project";
  // 会话令牌只保存在内存中，持久化由服务端下发的 HttpOnly Cookie 承担，
  // 避免敏感凭据落入 localStorage 被脚本读取（G.WST.01）。
  const LEGACY_TOKEN_KEY = "visionlab_token";

  const State = {
    token: "",
    user: safeParse(localStorage.getItem(USER_KEY)) || null,
    projectId: localStorage.getItem(PROJECT_KEY) || "",
    dict: null,
    projects: [],
    socket: null,
  };

  // 清理旧版本遗留在 localStorage 中的令牌。
  try { localStorage.removeItem(LEGACY_TOKEN_KEY); } catch (e) { /* 存储不可用时忽略 */ }

  function safeParse(text) {
    try { return text ? JSON.parse(text) : null; } catch (e) { return null; }
  }

  function setToken(token, user) {
    // 令牌仅驻留内存；页面刷新后由 HttpOnly Cookie 恢复会话。
    State.token = token || "";
    if (user) {
      State.user = user;
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    }
  }

  function qs(params) {
    if (!params) return "";
    const parts = [];
    Object.keys(params).forEach((key) => {
      const value = params[key];
      if (value === undefined || value === null || value === "") return;
      if (Array.isArray(value)) {
        value.forEach((v) => { parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(v)); });
      } else {
        parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(value));
      }
    });
    return parts.length ? "?" + parts.join("&") : "";
  }

  async function request(method, path, options) {
    options = options || {};
    const headers = Object.assign({}, options.headers || {});
    if (State.token) headers.Authorization = "Bearer " + State.token;
    let body = options.body;
    if (body && !(body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(body);
    }
    const response = await fetch(path + qs(options.query), {
      method: method, headers: headers, body: body, credentials: "same-origin",
    });
    if (response.status === 401) {
      setToken("", null);
      if (!location.pathname.startsWith("/login")) {
        location.href = "/login?next=" + encodeURIComponent(location.pathname);
      }
      throw new Error("未登录");
    }
    const contentType = response.headers.get("content-type") || "";
    if (contentType.indexOf("application/json") >= 0) {
      const data = await response.json();
      if (data.code !== 0 && data.code !== undefined) {
        const error = new Error(data.message || "请求失败");
        error.code = data.code;
        error.detail = data.detail;
        throw error;
      }
      return data;
    }
    return response;
  }

  const API = {
    state: State,
    setToken: setToken,
    get: (p, q) => request("GET", p, { query: q }),
    post: (p, b, q) => request("POST", p, { body: b || {}, query: q }),
    patch: (p, b) => request("PATCH", p, { body: b || {} }),
    del: (p) => request("DELETE", p),
    upload: (p, formData) => request("POST", p, { body: formData }),
    raw: request,

    /* ------------- 便捷封装 ------------- */
    async bootstrap() {
      const tasks = [
        API.get("/api/v1/system/dict").then(d => { State.dict = d.data; }).catch(() => {}),
        API.get("/api/v1/projects", { page_size: 100 })
          .then(d => { State.projects = d.data || []; }).catch(() => {}),
      ];
      await Promise.all(tasks);
      if (!State.projectId && State.projects.length) State.projectId = State.projects[0].id;
      return State;
    },
    async me() {
      const d = await API.get("/api/v1/auth/me");
      State.user = d.data.user;
      localStorage.setItem(USER_KEY, JSON.stringify(State.user));
      return State.user;
    },
    projectId() { return State.projectId; },
    setProject(id) {
      State.projectId = id;
      localStorage.setItem(PROJECT_KEY, id || "");
    },
    dict() { return State.dict || {}; },
    label(list, key, fallback) {
      const items = API.dict()[list] || [];
      const found = items.filter(x => x.key === key)[0];
      return found ? found.label : (fallback || key || "-");
    },
  };

  /* ================= 格式化工具 ================= */
  const Fmt = {
    num(value, digits) {
      if (value === null || value === undefined || value === "" || isNaN(value)) return "-";
      return Number(value).toFixed(digits === undefined ? 2 : digits);
    },
    m(value, digits) {
      if (value === null || value === undefined || value === "" || isNaN(value)) return "-";
      return Number(value).toFixed(digits === undefined ? 3 : digits) + " m";
    },
    pct(value, digits) {
      if (value === null || value === undefined || value === "" || isNaN(value)) return "-";
      return (Number(value) * 100).toFixed(digits === undefined ? 1 : digits) + "%";
    },
    int(value) {
      if (value === null || value === undefined || value === "") return "-";
      return Number(value).toLocaleString("zh-CN");
    },
    time(value) {
      if (!value) return "-";
      const text = String(value).replace("T", " ");
      return text.length > 19 ? text.slice(0, 19) : text;
    },
    date(value) {
      if (!value) return "-";
      return String(value).slice(0, 10);
    },
    bytes(value) {
      if (!value) return "0 B";
      const units = ["B", "KB", "MB", "GB", "TB"];
      let v = Number(value);
      let i = 0;
      while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
      return v.toFixed(1) + " " + units[i];
    },
    status(value) {
      return API.dict().status_labels ? (API.dict().status_labels[value] || value) : value;
    },
    statusTag(value) {
      const map = { done: "ok", archived: "info", running: "seg", queued: "warn",
                    paused: "warn", failed: "err", canceled: "err", draft: "", configured: "info" };
      return '<span class="tag ' + (map[value] || "") + '">' + Fmt.status(value) + "</span>";
    },
    taskTag(value) {
      const map = { detection: "detect", segmentation: "seg", ranging: "warn", joint: "info" };
      const label = (API.dict().task_labels || {})[value] || value;
      return '<span class="tag ' + (map[value] || "") + '">' + label + "</span>";
    },
    sizeGroup(value) {
      const labels = { small: "小目标", large: "大目标" };
      return labels[value] || "-";
    },
    illum(value) {
      const items = API.dict().illuminations || [];
      const found = items.filter(x => x.key === value)[0];
      return found ? found.label : (value || "-");
    },
    escape(text) {
      return String(text === undefined || text === null ? "" : text)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    },
  };

  /* ================= 提示与弹窗 ================= */
  const Toast = {
    show(message, type, timeout) {
      const box = document.getElementById("toast-container");
      if (!box) return;
      const el = document.createElement("div");
      el.className = "toast " + (type || "");
      el.textContent = message;
      box.appendChild(el);
      setTimeout(() => {
        el.style.transition = "opacity .3s";
        el.style.opacity = "0";
        setTimeout(() => el.remove(), 320);
      }, timeout || 3200);
    },
    ok(m) { Toast.show(m, "ok"); },
    err(m) { Toast.show(m, "err", 5200); },
    warn(m) { Toast.show(m, "warn", 4200); },
    info(m) { Toast.show(m, "", 3000); },
  };

  function handleError(error, prefix) {
    console.error(error);
    const message = (prefix ? prefix + "：" : "") + (error && error.message ? error.message : "未知错误");
    Toast.err(message);
  }

  let pendingConfirm = null;

  function hideModalRoot() {
    const root = document.getElementById("modal-root");
    if (root) root.classList.add("hidden");
  }

  const Modal = {
    open(title, bodyHtml, footerHtml) {
      const root = document.getElementById("modal-root");
      document.getElementById("modal-title").textContent = title || "";
      document.getElementById("modal-body").innerHTML = bodyHtml || "";
      const footer = document.getElementById("modal-footer");
      footer.innerHTML = footerHtml || '<button data-close="1">关闭</button>';
      root.classList.remove("hidden");
      root.querySelectorAll("[data-close]").forEach((el) => {
        el.onclick = () => { Modal.close(); };
      });
    },
    close() {
      hideModalRoot();
      if (pendingConfirm) {
        const settle = pendingConfirm;
        pendingConfirm = null;
        settle(false);
      }
    },
    // 自建确认框：返回 Promise<boolean>，避免使用浏览器原生弹窗（AOD.04）。
    ask(message, title) {
      return new Promise((resolve) => {
        if (pendingConfirm) {
          const previous = pendingConfirm;
          pendingConfirm = null;
          previous(false);
        }
        Modal.open(title || "操作确认", '<p class="muted">' + Fmt.escape(message) + "</p>",
          '<button data-confirm="0">取消</button><button class="danger" data-confirm="1">确定</button>');
        pendingConfirm = resolve;
        document.getElementById("modal-footer").querySelectorAll("[data-confirm]").forEach((btn) => {
          btn.onclick = () => {
            const accepted = btn.dataset.confirm === "1";
            const settle = pendingConfirm;
            pendingConfirm = null;
            hideModalRoot();
            if (settle) settle(accepted);
          };
        });
      });
    },
    body() { return document.getElementById("modal-body"); },
    footer() { return document.getElementById("modal-footer"); },
  };

  /* ================= 轮询（WebSocket 不可用时降级） ================= */
  function poll(fn, intervalMs) {
    let stopped = false;
    let timer = null;
    async function tick() {
      if (stopped) return;
      try { await fn(); } catch (e) { /* 静默 */ }
      if (!stopped) timer = setTimeout(tick, intervalMs);
    }
    tick();
    return {
      stop() { stopped = true; if (timer) clearTimeout(timer); },
    };
  }

  /* ================= 实时通道 ================= */
  function connectSocket(onFrame) {
    if (!global.io || !window.VISIONLAB || !window.VISIONLAB.hasSocket) return null;
    try {
      const socket = global.io({ transports: ["websocket", "polling"] });
      socket.on("connect", () => { socket.emit("subscribe", { channel: "stream" }); });
      socket.on("frame", (payload) => { if (onFrame) onFrame(payload); });
      State.socket = socket;
      return socket;
    } catch (e) {
      console.warn("Socket.IO 连接失败，降级为轮询", e);
      return null;
    }
  }

  global.VL = global.VL || {};
  Object.assign(global.VL, { API: API, Fmt: Fmt, Toast: Toast, Modal: Modal,
                             handleError: handleError, poll: poll, connectSocket: connectSocket,
                             state: State });
})(window);
