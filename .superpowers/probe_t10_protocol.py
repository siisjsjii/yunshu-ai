"""ch09 T10 探针:真机量协议违约率(spec §12.3-1 / 本章那条「没有硬保证」)。

连发 10 次**商品咨询**问题,统计:
  - `done` 帧的 trace 里有没有 `agent:protocol_violation`
  - `agent` 节点**真的被走到了**吗(`confidence_gate:pass` + `agent:converged`)
  - 回复是正常答案、兜底话术,还是 JSON 原文(后者 = 降级路径)
跑完打印一段可直接抄进 dev-notes 的汇总。

⚠️ SSE 的**帧名在 `event:` 那一行**,`data:` 里没有 `frame` 键(端点把
`payload["frame"]` 摘出来当事件名了)—— 首版探针只读 `data:` 里的 `frame`,
于是「一帧都没认出来」还静默继续,把 10 次空回复报成了「0 违约」。

⚠️ **含非 ASCII 的请求体一律走 httpx**(本仓栽过:MSYS2 会把 curl 的 argv
按 CP936 重编码,服务端只回 `error parsing the body`)。
"""

import json
import sys

import httpx

BASE = "http://127.0.0.1:8000"

#: 十条**商品咨询**问题,都取自知识库正文(问题太口语化会检索为空 ⇒ 被闸拦在
#: Agent 之前,那样这一轮压根走不到协议 —— 量出来的「0 违约」是假的)。
QUESTIONS = [
    "智能猫砂盆 Pro 容量多大",
    "智能猫砂盆 Lite 的尺寸是多少",
    "智能猫砂盆 Max 的重量",
    "自动饮水机的容量是多少",
    "自动喂食器基础版能装多少粮",
    "自动喂食器摄像头版的分辨率",
    "大型猫爬架有多高",
    "恒温加热垫的功率是多少",
    "大号三档加热垫有几个档位",
    "全景看护摄像头支持夜视吗",
]


def _parse(text: str) -> tuple[list[tuple[str, dict]], str]:
    """SSE → [(帧名, 载荷)],并拼回正文。帧名取 `event:` 那一行。"""
    frames: list[tuple[str, dict]] = []
    tokens: list[str] = []
    for block in text.split("\n\n"):
        name, payload = None, {}
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):].strip()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
        if name:
            frames.append((name, payload))
            if name == "token":
                tokens.append(payload.get("text") or "")
    return frames, "".join(tokens)


def main() -> int:
    rows = []
    with httpx.Client(timeout=180.0) as client:
        for i, question in enumerate(QUESTIONS, 1):
            resp = client.post(f"{BASE}/api/chat/stream",
                               json={"message": question, "session_id": f"t10-probe-{i}"})
            frames, reply = _parse(resp.text)
            done = next((p for n, p in frames if n == "done"), {})
            trace = done.get("trace") or []
            rows.append({
                "question": question,
                "code": resp.status_code,
                # **前提**:这一轮真的进了 Agent(否则协议压根没挂上去)
                "reached_agent": "agent:converged" in trace,
                "gate": done.get("gate_passed"),
                "intent": done.get("intent"),
                "violation": any("protocol_violation" in t for t in trace),
                "insufficient": any("self_assess_insufficient" in t for t in trace),
                "reply": reply,
            })

    print()
    print(f"{'#':>2}  {'码':>4}  {'进Agent':>7}  {'闸':>6}  {'违约':>4}  {'自评不足':>6}  问题")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}  {r['code']:>4}  {str(r['reached_agent']):>7}  {str(r['gate']):>6}"
              f"  {'是' if r['violation'] else '否':>4}"
              f"  {'是' if r['insufficient'] else '否':>6}  {r['question']}")

    reached = [r for r in rows if r["reached_agent"]]
    violations = [r for r in reached if r["violation"]]
    unusable = [r for r in reached if r["insufficient"]]
    print()
    print(f"共 {len(rows)} 次提问:进 Agent {len(reached)} 次;"
          f"其中违约 {len(violations)} 次,自评不足 {len(unusable)} 次")
    for i, r in enumerate(rows, 1):
        sys.stdout.buffer.write(f"  {i:>2} {r['question']} -> {r['reply'][:70]}\n"
                                .encode("utf-8"))
    print()
    print("意图/trace:")
    for i, r in enumerate(rows, 1):
        print(f"  {i:>2} intent={r['intent']} gate={r['gate']} "
              f"reached_agent={r['reached_agent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
