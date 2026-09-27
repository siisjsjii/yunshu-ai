/* 认证(2026-09-27):**唯一**一处实现「401 怎么办」的地方。
 *
 * 两个页面(聊天页 / 工作台)共用它 —— 401 的处置在两页上必须是同一件事,
 * 各写一遍迟早漂移,而漂移的表现是「一页弹登录、另一页白屏」。
 *
 * ⚠️ token 存 localStorage(**与现有的 mewhelp.session_id 同一个地方**)。
 *    代价如实记账:XSS 能读走它。本仓没有 httpOnly cookie + CSRF 那一套,
 *    而需求原话是「后续请求就携带 jwt」(见 spec §7.4)。
 */
(() => {
  "use strict";

  const TOKEN_KEY = "mewhelp.jwt";

  function getToken() {
    try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
  }
  function setToken(t) {
    try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* 隐私模式 */ }
  }
  function clearToken() { setToken(null); }

  function authHeader() {
    const t = getToken();
    return t ? { Authorization: `Bearer ${t}` } : {};
  }

  // ── 登录浮层 ──────────────────────────────────────────────
  let overlay = null;

  function buildOverlay() {
    const box = document.createElement("div");
    box.id = "auth-overlay";
    box.innerHTML = `
      <div class="auth-card">
        <h2>登录 · 云枢客服</h2>
        <label>用户名<input id="auth-user" type="text" autocomplete="username"></label>
        <label>密码<input id="auth-pass" type="password" autocomplete="current-password"></label>
        <div class="auth-err" id="auth-err"></div>
        <button id="auth-go" type="button">登录</button>
        <div class="auth-hint">
          演示账号:<code>cinfly / 123456</code>、<code>demo-user / 123456</code>
        </div>
      </div>`;
    document.body.appendChild(box);
    return box;
  }

  /** 弹登录浮层。`onDone` 在**登录成功**后调用一次(调用方拿它重放请求)。 */
  function showLogin(onDone) {
    if (!overlay) overlay = buildOverlay();
    overlay.style.display = "flex";
    const err = overlay.querySelector("#auth-err");
    err.textContent = "";
    const go = overlay.querySelector("#auth-go");

    async function submit() {
      const username = overlay.querySelector("#auth-user").value.trim();
      const password = overlay.querySelector("#auth-pass").value;
      if (!username || !password) { err.textContent = "用户名与密码都要填"; return; }
      go.disabled = true;
      err.textContent = "登录中…";
      try {
        const r = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        const body = await r.json().catch(() => null);
        if (!r.ok) {
          // ⚠️ 原样显示服务端的文案(它是 401「用户名或密码不正确」)——
          //    前端**不许**自己编一句,那会与服务端漂移。
          err.textContent = body && body.detail ? body.detail : `HTTP ${r.status}`;
          return;
        }
        setToken(body.token);
        overlay.style.display = "none";
        overlay.querySelector("#auth-pass").value = "";
        if (onDone) await onDone();
      } catch (e) {
        err.textContent = `连不上服务(${e.message})`;
      } finally {
        go.disabled = false;
      }
    }

    go.onclick = submit;
    overlay.querySelector("#auth-pass").onkeydown = (e) => { if (e.key === "Enter") submit(); };
    overlay.querySelector("#auth-user").focus();
  }

  /**
   * 带 token 的 fetch。**401 ⇒ 清 token、弹登录、成功后重放那一次的调用**。
   *
   * ⚠️ **只重放一次**(`_retried`):token 坏了而服务端照回 401 时不重放,
   *    否则会变成「登录 → 401 → 弹登录」的死循环。
   * ⚠️ **流式请求不自动重放**(`opts.stream === true`):那要重发一条用户消息,
   *    比让他再点一次「发送」更糟(见 spec §7.3)。那种情况下只弹登录、不回放。
   */
  async function authFetch(path, opts = {}) {
    const init = { ...opts, headers: { ...(opts.headers || {}), ...authHeader() } };
    delete init.stream;
    let resp = await fetch(path, init);
    if (resp.status !== 401) return resp;

    clearToken();
    const retry = new Promise((resolve) => {
      showLogin(() => resolve(true));
    });
    if (opts.stream) return resp;          // 流式:不重放,把这次 401 原样交回调用方
    const ok = await retry;
    if (!ok) return resp;
    return fetch(path, { ...opts, headers: { ...(opts.headers || {}), ...authHeader() } });
  }

  /** 顶栏那个「当前用户 / 退出」。两个页面各自调用,`el` 是容器。 */
  async function mountUserBadge(el) {
    const r = await fetch("/api/auth/me", { headers: authHeader() });
    if (!r.ok) { el.textContent = "未登录"; return; }
    const me = await r.json();
    el.replaceChildren();
    const name = document.createElement("span");
    name.textContent = `${me.username}(${me.role})`;
    const out = document.createElement("a");
    out.href = "#"; out.textContent = "退出";
    out.onclick = (e) => { e.preventDefault(); clearToken(); location.reload(); };
    el.append(name, document.createTextNode(" · "), out);
  }

  window.authFetch = authFetch;
  window.showLogin = showLogin;
  window.getToken = getToken;
  window.setToken = setToken;
  window.clearToken = clearToken;
  window.mountUserBadge = mountUserBadge;
  //: 启动时:手里没 token 就直接弹(两页共用同一条路径)。
  window.authBoot = (onReady) => {
    if (getToken()) { onReady(); return; }
    showLogin(onReady);
  };
})();
