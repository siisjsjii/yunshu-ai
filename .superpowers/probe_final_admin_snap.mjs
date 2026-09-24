/**
 * 最终修复轮的前端探针:把 `admin.html` 里**真的那段** `rawBlock` 抠出来跑一遍,
 * 看三种 `evidence_snapshot` 各渲染成哪一句。
 *
 * 为什么要抠而不是「看着对」:哨兵是一个 JSON **对象**,而那一支原先靠
 * `!Array.isArray(snap)` 与「空数组」共用一句话 —— 不跑一遍就分不清
 * 「改了文案」与「改了个走不到的分支」。
 *
 * 只搭最小 DOM 替身(`createElement` / `createTextNode` / `append` / `appendChild`),
 * 不引入 jsdom。**抠的是文件里的原文**,不是抄一份 —— 抄一份就等于测副本。
 *
 * 用法:`node .superpowers/probe_final_admin_snap.mjs`
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "..", "app", "static", "admin.html"), "utf8");

/** 抠出 `function rawBlock(q) { … }` 整段(靠大括号配平,不靠正则贪婪)。 */
function extractFn(name) {
  const start = html.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`admin.html 里找不到 function ${name}`);
  let i = html.indexOf("{", start);
  let depth = 0;
  for (let j = i; j < html.length; j += 1) {
    if (html[j] === "{") depth += 1;
    else if (html[j] === "}") {
      depth -= 1;
      if (depth === 0) return html.slice(start, j + 1);
    }
  }
  throw new Error(`${name} 的大括号没配平`);
}

class El {
  constructor(tag) {
    this.tag = tag;
    this.className = "";
    this.textContent = null;
    this.children = [];
  }
  appendChild(c) { this.children.push(c); return c; }
  append(...cs) { this.children.push(...cs); }
}

globalThis.document = {
  createElement: (tag) => new El(tag),
  createTextNode: (text) => ({ text }),
};

const src = `${extractFn("el")}\n${extractFn("rawBlock")}\nreturn rawBlock;`;
// eslint-disable-next-line no-new-func
const rawBlock = new Function("STATUS_LABEL", src)({});

/** 把一个元素树摊平成文本(判据取的是**人读到的那句话**)。 */
function flat(node) {
  if (node == null) return "";
  if (typeof node === "string") return node;
  if (node.text != null) return String(node.text);
  const own = node.textContent == null ? "" : String(node.textContent);
  return own + (node.children || []).map(flat).join("");
}

const CASES = [
  ["零召回(JSON null)", null, "召回片段快照:无"],
  ["回捞失败(哨兵对象)", { error: "recall_failed" }, "召回失败(检索服务不可用)"],
  // 真快照那一支**不写引导句**,它直接逐块画 `score · 章节 · chunk id`
  // (与 5b 那条对得上:审核页把分数原样 `String(c.score)` 打出来,
  //  所以两位写口必须都已经是四位的)。
  ["真快照(数组)", [{ chunk_id: 1, score: 0.1958, section_path: "s", answer: "a" }],
   "0.1958 · s · chunk 1"],
  ["空数组(第三个值,不该与上面两句混)", [], "零召回"],
];

let bad = 0;
for (const [name, snap, needle] of CASES) {
  const tree = rawBlock({
    question: "q", entry_point: "用户反馈", reject_reason: "r", evidence_snapshot: snap,
  });
  const text = flat(tree);
  const ok = text.includes(needle);
  if (!ok) bad += 1;
  console.log(`${ok ? "OK  " : "!!! "}${name}:${text}`);
}
// 哨兵那一句**必须**出现「失败」字样 —— 与零召回那句区分开(判据是这两句不相等)。
const failed = flat(rawBlock({ question: "q", evidence_snapshot: { error: "recall_failed" } }));
const zero = flat(rawBlock({ question: "q", evidence_snapshot: null }));
if (failed === zero || !failed.includes("失败")) {
  bad += 1;
  console.log("!!! 哨兵那一句与「零召回」那一句没能分开");
}
console.log(bad === 0 ? "全部通过" : `${bad} 处不符合`);
process.exit(bad === 0 ? 0 : 1);
