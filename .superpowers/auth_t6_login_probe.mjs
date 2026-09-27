/**
 * 认证 T6 · 前端 `auth.js` 的**数据半边**验证(Node 22 + 真服务端)。
 *
 * 本机没有 jsdom、没有 playwright ⇒ 那份「五个点的手工点击」**做不到**。
 * 但 `auth.js` 里真正会出错的那半边 —— 401 之后**到底发生了什么**
 * (弹没弹浮层?token 存没存?原来那次请求重放了没?流式那条有没有误伤?)——
 * 是**可以**验的:把 `app/static/auth.js` **原样当文件读进来**,配一个
 * 最小的 DOM / localStorage 替身,在 vm 里跑,`fetch` 打的
 * 是 **8000 上那个真服务端**。
 *
 * ⚠️ 替身的边界,写清楚免得被当成「测过了浏览器」:
 *   · 它**不做布局** —— 浮层盖没盖住、`position: fixed` 生不生效、z-index
 *     压不压得住侧栏、`box-shadow` 好不好看……**一个字都没验**;
 *   · 它**不做真实鼠标事件** —— 这里「点登录」是**直接调 `go.onclick()`**,
 *     不是派发一个 click 事件走浏览器的事件系统;
 *   · 它**不做 HTML 解析** —— 浮层那几行 DOM 是从 `auth.js` 自己的
 *     `innerHTML` 字符串里按 `id="…"` 抠出来的,`<label>` / `<code>` 的
 *     嵌套关系不存在;
 *   · 它**不跑两个页面的内联脚本** —— `index.html` / `admin.html` 里那把
 *     IIFE 在这条链路里只有两个入口被验到(`authBoot` 的闸 + `#who` 那个
 *     id 存在),「侧栏画出来长什么样」「五个标签页切得对不对」不在内。
 *   · `getElementById` / `querySelector` 对**不存在的 id 直接抛** ——
 *     于是「名字写错」在这里是**响亮的失败**,而不是一页静悄悄的白屏。
 *
 * 用法:`node .superpowers/auth_t6_login_probe.mjs`
 * 前置:客服服务起在 8000(脚本自己不清场 —— 见 CLAUDE.md 那条「先清残留进程」,
 *       本机实测残留进程会让你 curl 到旧代码,拿到假红)。启动要等 BGE-M3,约 15s。
 */

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const ROOT = path.resolve(import.meta.dirname, "..");
const BASE = "http://127.0.0.1:8000";
const AUTH_JS = path.join(ROOT, "app/static/auth.js");

let failures = 0;
function fail(msg) { console.error("  !!! " + msg); failures++; }
function ok(msg) { console.log("  ok  " + msg); }
function check(cond, msg) { cond ? ok(msg) : fail(msg); return cond; }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── 最小 DOM 替身 ────────────────────────────────────────────────
//: id → 节点。**只有 HTML/auth.js 自己声明过的 id 才会进来** ——
//: 查不到就抛,这是本替身最值钱的一条(名字写错不许静默)。
const registry = new Map();

function makeNode(id = "", tag = "div") {
  const node = {
    id, tagName: tag, textContent: "", value: "", disabled: false, href: "",
    style: {}, children: [], _byId: new Map(), _innerHTML: "",
    classList: { add() {}, remove() {}, toggle() {} },
    appendChild(c) { if (c && c.id) registry.set(c.id, c); this.children.push(c); return c; },
    append(...cs) { for (const c of cs) if (c && c.id) registry.set(c.id, c); this.children.push(...cs); },
    replaceChildren(...cs) { this.children = cs; },
    focus() { this._focused = true; },
    remove() {},
    querySelector(sel) {
      if (!sel.startsWith("#")) return null;
      const key = sel.slice(1);
      if (this._byId.has(key)) return this._byId.get(key);
      if (registry.has(key)) return registry.get(key);
      throw new Error(`querySelector("${sel}") 找不到任何节点 —— 这个 id 写错了`);
    },
    querySelectorAll() { return []; },
  };
  //: `auth.js` 的浮层是**一整段 innerHTML** 造出来的 ⇒ 替身必须把它里面
  //: 那些 `id="…"` 变成**真的节点**,名字从 auth.js 自己的字符串里来。
  //: 少一个 id,下面 `querySelector("#auth-xxx")` 就会抛 —— 正是我们要的。
  //: `auth.js` 的 `showLogin` 每调一次就 `go.onclick = submit` 一次 ⇒
  //: 「这个按钮被重新接线了几次」正好等于「此刻有几个调用方在等登录」。
  //: ⑧ 号观察靠它把「等三个 401 都回来了再点」变成**确定**的,不是靠 sleep 猜。
  Object.defineProperty(node, "onclick", {
    get() { return this._onclick; },
    set(fn) { this._onclick = fn; this.onclickWires = (this.onclickWires || 0) + 1; },
  });
  Object.defineProperty(node, "innerHTML", {
    get() { return this._innerHTML; },
    set(v) {
      this._innerHTML = v;
      for (const m of String(v).matchAll(/\bid="([^"]+)"/g)) {
        const child = makeNode(m[1], "input");
        this._byId.set(m[1], child);
        registry.set(m[1], child);
      }
    },
  });
  if (id) registry.set(id, node);
  return node;
}

const documentStub = {
  getElementById(id) {
    const n = registry.get(id);
    if (!n) throw new Error(`getElementById("${id}") —— 没有这个 id`);
    return n;
  },
  createElement: (tag) => makeNode("", tag),
  createTextNode: (t) => ({ textContent: t }),
  querySelectorAll: () => [],
  body: makeNode("body"),
};
registry.set("body", documentStub.body);

// ── localStorage 替身(浏览器的那个是同步的,这里也是)────────────
const store = new Map();
const localStorageStub = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
};

// ── fetch:浏览器替我们做的那件事(**补 origin**)+ 记流水 ─────────
//: 页面里的路径全是相对的(`/api/…`)。Node 的 `fetch` 会直接抛
//: `Failed to parse URL`,所以替身必须自己补 origin —— 否则每个用例都会
//: 报「连不上服务」,而那**不是 auth.js 的错**。
const calls = [];
function browserFetch(u, o = {}) {
  const url = typeof u === "string" && u.startsWith("/") ? BASE + u : String(u);
  const h = o.headers || {};
  calls.push({ url, method: (o.method || "GET").toUpperCase(), auth: h.Authorization ?? null,
               body: o.body ?? null });
  return fetch(url, o);
}
const hitCount = (p) => calls.filter((c) => c.url === BASE + p).length;
const hits = (p) => calls.filter((c) => c.url === BASE + p);

let reloads = 0;
const locationStub = { reload() { reloads++; } };

// ── 跑**真的** auth.js(按文件读,不是抄一份)──────────────────
const authSrc = fs.readFileSync(AUTH_JS, "utf8");
const sandbox = {
  document: documentStub, localStorage: localStorageStub, fetch: browserFetch,
  location: locationStub, console, setTimeout, clearTimeout, Promise, JSON, Math,
  Object, Array, Set, Map, Number, String, Boolean, isFinite, RegExp, Error, Date,
};
//: 浏览器里 `window === globalThis`。替身也必须这样 —— `auth.js` 用
//: `window.authBoot = …` 挂名字,下面要**不带 window 前缀**地全局调它们。
sandbox.window = sandbox;
const ctx = vm.createContext(sandbox);
try {
  vm.runInContext(authSrc, ctx, { filename: "app/static/auth.js" });
} catch (e) {
  fail("auth.js 在加载阶段就抛了:" + e.message);
  process.exit(1);
}
for (const name of ["authFetch", "showLogin", "getToken", "setToken", "clearToken",
                    "mountUserBadge", "authBoot"]) {
  if (typeof sandbox[name] !== "function") {
    fail(`auth.js 没有把 \`${name}\` 挂到 window 上 —— 页面里那个全局名是 undefined`);
    process.exit(1);
  }
}
console.log(`auth.js 已从 ${path.relative(ROOT, AUTH_JS)} 读入并在 vm 里跑起来`);

// ── 小工具 ──────────────────────────────────────────────────────
const overlayEl = () => registry.get("auth-overlay");
const visible = () => overlayEl()?.style.display === "flex";
const hidden = () => overlayEl()?.style.display === "none";
const errText = () => overlayEl()._byId.get("auth-err").textContent;
const storedToken = () => localStorageStub.getItem("mewhelp.jwt");

/** 「点一下登录按钮」——直接调 onclick(`submit` 是 async,回 Promise)。 */
async function clickLogin(username, password) {
  const o = overlayEl();
  o._byId.get("auth-user").value = username;
  o._byId.get("auth-pass").value = password;
  const go = o._byId.get("auth-go");
  if (!go || typeof go.onclick !== "function") throw new Error("登录按钮上没有 onclick");
  await go.onclick();
}

/** 盯一个 promise 有没有落定(不 await 它)。 */
function watch(p) {
  const w = { settled: false, status: null, err: null, bodyText: null };
  p.then(async (resp) => {
    w.settled = true; w.status = resp.status;
    try { w.bodyText = await resp.text(); } catch { /* 流已取走 */ }
  }, (e) => { w.settled = true; w.err = e; });
  return w;
}
async function waitFor(cond, ms = 5000, every = 25) {
  const until = Date.now() + ms;
  while (Date.now() < until) { if (cond()) return true; await sleep(every); }
  return cond();
}

const TOKEN_KEY = "mewhelp.jwt";
const reset = () => { localStorageStub.removeItem(TOKEN_KEY); };

console.log("\n================= 六条行为(真 fetch → 真服务端)=================");

// ── ① 没 token + 启动 ⇒ 弹浮层,而且**不往下走** ────────────────
console.log("\n① 没 token 时 authBoot:应弹浮层、且 onReady 不该被调用");
reset();
let booted = false;
sandbox.authBoot(() => { booted = true; });
await sleep(120);                       // 给「本不该发生的那条路」一点时间露头
check(visible(), `#auth-overlay 显示出来了(style.display=${JSON.stringify(overlayEl().style.display)})`);
check(booted === false, "onReady **没有**被调用(没 token 就不 boot —— 这条就是「不往下走」)");

// ── ② 密码错 ⇒ 原样显示**服务端的** detail,且不落 token ────────
console.log("\n② 密码错:浮层要显示服务端那句 detail,且 token 不许落库");
//: 先独立问一次服务端,拿到**它自己的**文案 —— 这样「前端显示的是服务端的
//: 那句话」才是被验的,而不是「前端显示的是我抄进断言的那句话」。
const srvBad = await (await fetch(`${BASE}/api/auth/login`, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ username: "cinfly", password: "definitely-wrong" }),
})).json();
console.log(`  服务端原文 = ${JSON.stringify(srvBad.detail)}`);
await clickLogin("cinfly", "definitely-wrong");
check(errText() === srvBad.detail,
      `浮层上是服务端那句(读到 ${JSON.stringify(errText())})`);
check(storedToken() === null, "localStorage 里没有 token");
check(visible(), "浮层还开着(登录没成功,用户还有机会再试)");

// ── ③ 密码对 ⇒ token 落库 + 浮层收起 + onReady 这才跑 ───────────
console.log("\n③ 密码对:token 要落 localStorage、浮层收起、onReady 这才被调用");
await clickLogin("cinfly", "123456");
const tk = storedToken();
check(typeof tk === "string" && tk.split(".").length === 3,
      `localStorage["${TOKEN_KEY}"] 是一个 JWT(三段点分,长度 ${tk ? tk.length : 0})`);
check(hidden(), `浮层收起了(style.display=${JSON.stringify(overlayEl().style.display)})`);
check(booted === true, "onReady 被调用了 —— 登录成功是**唯一**放行「往下走」的闸");

// ── ④ 受保护请求没 token ⇒ 401 ⇒ 弹浮层 ⇒ 登录后**重放** ────────
console.log("\n④ authFetch:401 之后要**重放原来那次调用**(这是整件事的承重点)");
reset();
const before4 = hitCount("/api/conversations");
const w4 = watch(sandbox.authFetch("/api/conversations"));
check(await waitFor(visible, 4000), "第一次请求 401 之后浮层弹出来了");
const firstLeg = hits("/api/conversations").slice(before4);
check(firstLeg.length === 1 && firstLeg[0].auth === null,
      "第一次那条**没带** Authorization(它就是这么 401 的)");
check(w4.settled === false, "这次调用**还挂着** —— 在等用户登录(没有自己放弃)");
await clickLogin("demo-user", "123456");
check(await waitFor(() => w4.settled, 5000), "登录之后那次调用回来了");
const legs4 = hits("/api/conversations").slice(before4);
check(legs4.length === 2, `一共发了 ${legs4.length} 次(1 次 401 + 1 次重放)`);
check(legs4[1]?.auth === `Bearer ${storedToken()}`,
      "重放那次带上了**新拿到**的 token");
check(w4.status === 200, `重放收回来的状态码 = ${w4.status}`);
let items4 = null;
try { items4 = JSON.parse(w4.bodyText).items; } catch { /* 下面那条会红 */ }
check(Array.isArray(items4) && items4.length > 0,
      `重放拿到的 body 是真数据(demo-user 名下 ${items4 ? items4.length : "?"} 条会话)`);

// ── ⑤ 垃圾 token ⇒ 下一次受保护请求 401 ⇒ 浮层再弹 ──────────────
console.log("\n⑤ 垃圾 token:下一次受保护请求要 401、浮层要再弹,且**不许自己重放**");
localStorageStub.setItem(TOKEN_KEY, "not-a-jwt");
const before5 = hitCount("/api/conversations");
const w5 = watch(sandbox.authFetch("/api/conversations"));
check(await waitFor(visible, 4000), "带着垃圾 token 的请求把浮层又弹出来了");
const legs5 = hits("/api/conversations").slice(before5);
check(legs5.length === 1, `此刻只发了 ${legs5.length} 次 —— 没有「登录→401→再弹」的自转`);
check(legs5[0]?.auth === "Bearer not-a-jwt", "那一次确实是拿着垃圾 token 发的");
check(storedToken() === null, "401 之后 token 被清掉了(垃圾 token 不会留在手里)");
await clickLogin("demo-user", "123456");
check(await waitFor(() => w5.settled, 5000), "重新登录之后这次调用也回来了");
check(w5.status === 200, `重放收回来的状态码 = ${w5.status}`);

// ── ⑥ stream:true ⇒ 401 只弹浮层、**不重放** ────────────────────
console.log("\n⑥ stream:true:流式那次 401 只弹浮层,**绝不重放**(重放=替用户再发一条消息)");
reset();
const before6 = hitCount("/api/chat/stream");
const w6 = watch(sandbox.authFetch("/api/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "你好" }),
  stream: true,
}));
check(await waitFor(() => w6.settled, 4000), "这次调用**当场就回来了**(没有挂在登录上)");
check(w6.status === 401, `交回调用方的就是那个 401(status=${w6.status})`);
check(JSON.parse(w6.bodyText || "{}").detail === "未认证",
      `body 是服务端的 401 文案(${JSON.stringify(w6.bodyText)})`);
check(visible(), "浮层还是弹了(用户被明确告知要登录,而不是静默失败)");
//: 关键的一条:登录之后**也不许**把那条流式请求补发出去。
await clickLogin("demo-user", "123456");
await sleep(600);
const legs6 = hits("/api/chat/stream").slice(before6);
check(legs6.length === 1, `登录之后 /api/chat/stream 仍然只被打了 ${legs6.length} 次`);

// ── ⑦(附加)`<script src>` 那条路径**真的取得到这个文件** ────────
//: brief 里写的是 `/static/auth.js`,而 `app/main.py` 把静态目录 `mount("/")`
//: 挂在了**根**上 ⇒ 正确路径是 `/auth.js`。这一条把它做成**读数**而不是说法。
console.log("\n⑦(附加)两个页面引的 auth.js 路径,在真服务端上取得到、且与磁盘上逐字节相同");
const diskBytes = fs.readFileSync(AUTH_JS);
for (const name of ["index.html", "admin.html"]) {
  const html = fs.readFileSync(path.join(ROOT, "app/static", name), "utf8");
  const m = html.match(/<script\s+src="([^"]+)"\s*>/);
  if (!m) { fail(`${name} 里没有 <script src=…> —— auth.js 根本没被引进来`); continue; }
  const r = await fetch(BASE + m[1]);
  const served = Buffer.from(await r.arrayBuffer());
  check(r.status === 200 && served.equals(diskBytes),
        `${name} 引的是 ${m[1]} ⇒ HTTP ${r.status},` +
        `${served.equals(diskBytes) ? "内容与磁盘逐字节相同" : "内容对不上"}`);
}

// ── ⑧(观察,不判红)并发 401 时,浮层的等待者会丢 ────────────────
//: `auth.js` 的浮层是**一个**(模块级 `overlay`),而 `showLogin(onDone)`
//: **每次调用都覆盖** `go.onclick` ⇒ 同一时刻有 N 个调用方在等登录时,
//: 只有**最后一次**那个 `onDone` 活下来,前 N-1 个的 promise **永远不落定**。
//: 这一段**只测量、不判红** —— brief 里的 auth.js 是逐字实现的,缺陷在
//: brief 里;把它量出来交给 controller 定夺,比在这里静悄悄改掉更诚实。
console.log("\n⑧(观察 · 不判红)三个并发 401 的调用,登录之后有几个回来了");
reset();
const before8 = hitCount("/api/conversations");
const goBtn = overlayEl()._byId.get("auth-go");
const wiresBefore = goBtn.onclickWires || 0;
const ws = [1, 2, 3].map(() => watch(sandbox.authFetch("/api/conversations")));
check(await waitFor(() => (goBtn.onclickWires || 0) - wiresBefore === 3, 5000),
      "三个 401 都回来了(登录按钮被重新接线 3 次 = 3 个调用方在等)");
await clickLogin("demo-user", "123456");
await waitFor(() => ws.some((w) => w.settled), 3000);
await sleep(800);
const cameBack = ws.filter((w) => w.settled).length;
console.log(`  读数:3 个并发调用里,登录之后回来了 ${cameBack} 个` +
            `(前 ${3 - cameBack} 个还在等 —— 它们的 promise 永远不会落定)`);
if (cameBack < 3) {
  console.log("  ⚠️ 这是 **brief 里那段 auth.js 的**性质,不是实现走样:`showLogin` 每次调用");
  console.log("     都把登录按钮的 `onclick` 换成新的 `submit`,于是只有最后一个等待者被唤醒。");
  console.log("     可达路径:`admin.html` 的「首页」一进去就 `renderHome()` —— 五张卡");
  console.log("     由 `Promise.allSettled(HOME_LOADERS.map(…))` **并发**打五条 `req()`。");
  console.log("     token **过期**时(不是「没有」:没有时 authBoot 会先拦下)那五条一起 401");
  console.log("     ⇒ 登录后**四张卡永远停在「读取中…」**,而页面不报任何错。");
}

// ── ⑨(附加)顶栏「谁 · 退出」:问得出当前用户,点退出要清 token ────
//: 这一段钉的是浮层**之外**那半个:`mountUserBadge` 真的从 `/api/auth/me`
//: 读到了用户名与角色(而不是把服务端返回的形状猜错、在页面上打出
//: `undefined(undefined)`),以及「退出」真的是**清 token + 重载**。
console.log("\n⑨(附加)顶栏徽标 + 退出:要读到登录的那个人,点退出要清 token 并重载");
//: ⑥ 那次登录留下的是 demo-user 的 token。
const who = makeNode("who");
//: ⚠️ 期望值**向服务端要**,不写死 —— 本仓那条「断言一个字段之前先去读它是怎么
//: 被赋值的」在这里的具体形态:两个演示账号的 `role` 都是 `admin`(见
//: `docs/superpowers/plans/2026-09-27-ecommerce-cs-auth.md` 里「`role` 都 admin」),
//: 写死一个 `(user)` 就是在替服务端编数据,而页面**是对的**。
const me = await (await fetch(`${BASE}/api/auth/me`, {
  headers: { Authorization: `Bearer ${storedToken()}` },
})).json();
await sandbox.mountUserBadge(who);
check(who.children.length === 3, `顶栏被填成了三个节点(用户名 / 分隔符 / 退出)`);
check(who.children[0]?.textContent === `${me.username}(${me.role})`,
      `用户名与角色来自 /api/auth/me(排面 ${JSON.stringify(who.children[0]?.textContent)},` +
      `服务端说 ${JSON.stringify(`${me.username}(${me.role})`)})`);
const outLink = who.children[2];
check(outLink?.textContent === "退出", "第三个节点是「退出」");
outLink.onclick({ preventDefault() {} });
check(storedToken() === null, "点「退出」之后 token 被清掉了");
check(reloads === 1, `而且触发了一次 location.reload(${reloads} 次)—— 重载之后 authBoot 才会重新弹浮层`);

// ── 汇总 ────────────────────────────────────────────────────────
console.log("\n================================================================");
console.log(failures ? `有 ${failures} 条失败(见上面的 !!!)`
                     : "brief 那六条行为逐条通过(外加 ⑦⑨ 两条附加断言)");
console.log("⚠️ ⑧ 那一段**不参与判红** —— 它是缺陷**读数**,不是回归判据。");
console.log("⚠️ 本探针**验不到**的:布局 / CSS(浮层的 position:fixed 与 z-index 压不压得住)、");
console.log("   真实的鼠标点击、两个页面的实际观感 —— 那些仍然要人眼看。");
process.exitCode = failures ? 1 : 0;
