/**
 * 首页五张卡「填出来是什么样」—— 拿**真的 JS + 真的服务端**跑一遍。
 *
 * 本机没有 jsdom,而这个页面的数据半边(五个 loader 到底能不能把服务端的东西
 * 填进那五行、会不会填出 `undefined`)是**可以**验的:那把 `<script>` 原样取出来,
 * 配一个**最小的 DOM 替身**在 Node 里跑。`fetch` 是 Node 22 自带的,打的是
 * **真服务端**(8000 上那个)。
 *
 * ⚠️ 替身的边界,写清楚免得被当成「测过了浏览器」:
 *   · 它**不做布局** —— 宽度、颜色、换行、六个目录等宽……**一个字都没验**;
 *   · 它**不做事件** —— 按钮点下去会发生什么,这里不验(`addEventListener` 是个空壳);
 *   · `getElementById` 对**不存在的 id 直接抛**(不是返回一个空壳节点)——
 *     于是「名字写错」在这里是**响亮的失败**,而不是一张停在「读取中…」的卡。
 *   它**验的是**:五行摘要的**文案**、徽标的**词**、以及五个 loader
 *   **全都走到了终态**(没有一个是 `读取中…`)。
 *
 * 用法:`node .superpowers/ch10f_home_render_probe.mjs`
 * 前置:客服服务起在 8000(脚本自己不清场,见 CLAUDE.md 那条「先清残留进程」)。
 */

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const ROOT = path.resolve(import.meta.dirname, "..");
const BASE = "http://127.0.0.1:8000";

function fail(msg) {
  console.error("!!! " + msg);
  process.exitCode = 1;
}

const html = fs.readFileSync(path.join(ROOT, "app/static/admin.html"), "utf8");

// ── 最小 DOM 替身 ────────────────────────────────────────────────
const nodes = new Map();
function makeNode(id, cls) {
  const n = {
    id: id || "", className: cls || "", textContent: "", innerHTML: "",
    style: {}, dataset: {}, children: [], value: "", disabled: false,
    classList: { toggle() {}, add() {}, remove() {} },
    appendChild(c) { this.children.push(c); return c; },
    append(...cs) { this.children.push(...cs); },
    replaceChildren(...cs) { this.children = cs; },
    prepend(c) { this.children.unshift(c); },
    addEventListener() {},
    // 只需要「有没有孩子」这件事(飞轮那段用),这里一律给 null
    querySelector() { return null; },
    querySelectorAll() { return []; },
    remove() {},
  };
  return n;
}

//: 从 HTML 里把**所有 id** 收进来 —— 于是「页面真的有这个元素」这件事
//: 与浏览器里一致;没写错的话下面一次都不会抛。
for (const m of html.matchAll(/\bid="([^"]+)"/g)) nodes.set(m[1], makeNode(m[1]));

//: `.tab` 与 `[data-goto]` 两组选择器按 HTML 里的顺序现造(不写死名字)。
const tabButtons = [...html.matchAll(/data-tab="([^"]+)"/g)].map((m) => {
  const n = makeNode("", "tab"); n.dataset = { tab: m[1] }; return n;
});
const gotoButtons = [...html.matchAll(/data-goto="([^"]+)"/g)].map((m) => {
  const n = makeNode("", "home-go"); n.dataset = { goto: m[1] }; return n;
});

const documentStub = {
  getElementById(id) {
    const n = nodes.get(id);
    if (!n) throw new Error(`getElementById("${id}") —— HTML 里没有这个 id`);
    return n;
  },
  createElement: () => makeNode(),
  createTextNode: (t) => ({ textContent: t }),
  querySelectorAll(sel) {
    if (sel === ".tab") return tabButtons;
    if (sel === "[data-goto]") return gotoButtons;
    return [];
  },
  body: makeNode("body"),
};

// ── 跑那把 script(与页面里逐字同一份)───────────────────────────
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
//: ⚠️ 页面里那些路径全是**相对**的(`/api/kb/stats`)—— 浏览器拿当前 origin
//: 补全,Node 的 `fetch` 会直接抛 `Failed to parse URL`。替身必须做**浏览器做的
//: 那一件事**(补 origin),否则五张卡会一起报「读取失败」,而那**不是页面的错**。
const browserFetch = (u, o) =>
  fetch(typeof u === "string" && u.startsWith("/") ? BASE + u : u, o);

const ctx = vm.createContext({
  document: documentStub, window: {}, fetch: browserFetch, setTimeout, clearTimeout,
  console, Promise, JSON, Object, Array, Math, Date, Set, Number, String, isFinite,
  URLSearchParams, encodeURIComponent,
});
try {
  vm.runInContext(script, ctx);
} catch (e) {
  fail("页面脚本在加载阶段就抛了:" + e.message);
}

// ── 先确认**脚本真的跑了**:`switchTab("首页")` 会把六个页的 display 摆好 ──
// ⚠️ 替身不做布局,但 `.style.display` 是 JS 自己写上去的赋值 —— 它是不是被写过,
//    正是「初始化那条路径到底有没有执行」的判据(少了这一步,下面五张全空
//    会被读成「服务端没数据」)。
const shown = nodes.get("tab-首页").style.display;
const hidden = nodes.get("tab-入库").style.display;
console.log(`switchTab("首页") 生效: tab-首页.display=${JSON.stringify(shown)} ` +
            `tab-入库.display=${JSON.stringify(hidden)}`);
if (shown !== "block" || hidden !== "none") {
  fail("初始化没跑到 switchTab(六个页的显示状态没被摆过)");
}

// ── 等五张卡走到终态 ─────────────────────────────────────────────
//: `homeSay` 一定会同时写正文与徽标 ⇒ 「两个都非空」就是终态;
//: 只等正文的话,一张卡写空串也会被当成「已经好了」。
const IDS = ["kb", "rv", "ev", "tp", "tr"];
const TITLES = { kb: "入库", rv: "待审", ev: "评测", tp: "主题分布", tr: "链路" };
const deadline = Date.now() + 30000;
while (Date.now() < deadline) {
  const pending = IDS.filter((i) =>
    !nodes.get(`home-${i}-line`).textContent || !nodes.get(`home-${i}-badge`).textContent);
  if (!pending.length) break;
  await new Promise((r) => setTimeout(r, 200));
}

console.log("========== 首页五张卡(真 JS + 真服务端)==========");
for (const i of IDS) {
  const badge = nodes.get(`home-${i}-badge`);
  const line = nodes.get(`home-${i}-line`);
  console.log(`\n【${TITLES[i]}】徽标 = ${JSON.stringify(badge.textContent)}` +
              ` (class=${JSON.stringify(badge.className)})`);
  console.log(`        正文 = ${line.textContent}`);
  if (line.textContent === "读取中…") fail(`${i} 那张卡**没走到终态**(停在「读取中…」)`);
  for (const bad of ["undefined", "NaN", "[object"]) {
    if (line.textContent.includes(bad) || badge.textContent.includes(bad)) {
      fail(`${i} 那张卡的文案里有 "${bad}" —— 字段名或形状对不上`);
    }
  }
}
console.log("\n================================================");
console.log(process.exitCode ? "有失败(见上面的 !!!)" : "五张卡都填出来了,无 undefined/NaN");
