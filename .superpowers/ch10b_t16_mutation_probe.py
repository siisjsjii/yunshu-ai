#!/usr/bin/env python
"""ch10-B T16 的变异探针:把**产品**改坏 / 把**判词喂错**,验收必须变红。

```
.venv/Scripts/python.exe .superpowers/ch10b_t16_mutation_probe.py --list
.venv/Scripts/python.exe .superpowers/ch10b_t16_mutation_probe.py --product M1
.venv/Scripts/python.exe .superpowers/ch10b_t16_mutation_probe.py --judge J3
```

## 两类探针,问的是两个不同的问题

| 类 | 改什么 | 问什么 |
|---|---|---|
| `--product` | **产品代码 / 数据** | 「产品坏了,验收会不会红、**红在哪一条**」——一整轮验收(实测 **103–107 秒**;订正轮 1 把这里原先写的「约 4 分钟」改对了 —— 那是个没量过的数) |
| `--judge` | **喂给判词内核的输入** | 「某一条判据**是不是恒真**」——不跑服务、秒级 |

⚠️ 两类都要:**产品级**证明装置在真链路上有判别力;**判词级**把每一条判据单独拎出来
喂一个**错的输入**,证明它不是同义反复(本仓的头号风险是假绿测试,而验收脚本尤其危险:
一句 `grep -q "micro-F1"` 对一个**内容全错**的报告同样成立)。

## 三条装置规矩(都是本仓付过学费的)

1. **锚点锚在代码上**(不锚注释),且每次替换后**断言命中数恰好为 1** ——
   命中 0 就是「变异压根没生效」,而它与「断言无判别力」长得一模一样。
2. **还原一律字节读写**(`写入前先存 bytes,还原后核 sha256`)。⚠️ **不许用
   `git checkout --` / `cp`**:本仓 `core.autocrlf=true`,那两条会把行尾从 LF 改成
   CRLF —— 那不是还原(阶段 8 的复审亲身踩过)。
3. **跑验收要带 timeout**,而且必须是**本进程自己的** timeout(比外层工具的短):
   外层的 `SIGKILL` 打不到 `finally`,`变异会残留在源码里`(本仓今天已两次)。

用法之外只读:`$TEMP` 里那些是验收自己的产物;本脚本只碰它列在 `MUTATIONS` 里的文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# 探针自己也要 import `app.*`(J4/J5 要读权威标签表与产物)—— 与内核同款。
sys.path.insert(0, str(REPO))
ACCEPTANCE = "scripts/acceptance_ch10.sh"
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
# ⚠️⚠️ **`subprocess.run(["bash", …])` 拿到的是 WSL 的 bash,不是 Git Bash**(2026-09-27
# 实测,已写进验收脚本的装置自检):`CreateProcess` 的搜索顺序把 **System32 排在 PATH
# 之前** ⇒ 解析到 `C:\Windows\System32\bash.exe`(WSL)。WSL 里 `localhost:8000` 到不了
# Windows 上的服务、`/tmp` 也不是 Python 眼里那个 —— 第一次跑 M1 就是这个假红
# (`预检失败:客服服务起不来`,而真因是跑错了 shell)。
# ⇒ **钉到 Git Bash 的绝对路径,并用 `uname -s` 当场核**(MINGW*/MSYS 才对;
#    WSL 是 `Linux`)。核不过就停,不猜。
BASH = os.environ.get("CH10_BASH") or shutil.which("bash") or r"D:\kit\Git\usr\bin\bash.exe"
#: 判定内核**只有一份**(写在验收脚本的 heredoc 里)⇒ 这里从**那个 heredoc**
#: 现场抽出来跑,不留第二份副本(第二份就是漂移的开始)。抽到 `$WORK` 下 ——
#: 内核靠 `Path(__file__).parents[1]` 找仓库根,放到别处它会去找别人的仓库。
HELPER = REPO / ".ch10_acceptance" / "topic_helpers.py"


def extract_helper() -> Path:
    """把 `acceptance_ch10.sh` 里那段 heredoc 抽成文件(权威副本只有这一处)。"""
    text = (REPO / "scripts" / "acceptance_ch10.sh").read_text(encoding="utf-8")
    head = 'cat > "$WORK/topic_helpers.py" <<' + chr(39) + "PYEOF" + chr(39) + chr(10)
    start = text.index(head) + len(head)
    end = text.index(chr(10) + "PYEOF" + chr(10), start)
    HELPER.parent.mkdir(parents=True, exist_ok=True)
    HELPER.write_text(text[start:end], encoding="utf-8")
    return HELPER
EVID = REPO / ".superpowers"
TMP = Path(os.environ.get("TEMP") or tempfile.gettempdir())
#: 跑一轮验收的上界。**比外层工具的 timeout 短**,好让 `finally` 有机会跑
#: (见文件头第 3 条:外层 SIGKILL 打不到 finally)。
ACCEPTANCE_TIMEOUT = 420


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(text: str = "") -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


# ══════════════════════════════════════════════════════════════════════════
# 产品级变异
# ══════════════════════════════════════════════════════════════════════════

PRODUCT_MUTATIONS = {
    # ③ 的承重判据是「多诉求句 ≥2 个类目」。把逐标签过阈值改成 **argmax** 就退化回
    # 单标签 —— 这正是 `topic_service/model.py` docstring 里点名的那条。
    "M1": {
        "path": "topic_service/model.py",
        "why": "把「逐标签过阈值」改成 argmax(多标签退化成单标签)⇒ 验收 ③ 必须红",
        "expect": "③:多诉求句只命中 1 个类目 / 标签与阈值解码不一致",
        "edits": [(
            '                "labels": [label for label, p in scores.items() if p >= self.threshold],',
            '                "labels": [max(scores, key=scores.get)],',
        )],
    },
    # ② 的核心判据是「分布接口的每个数 == 另一条独立算法」。把「不同问题数」退回
    # 按 `low_confidence_question_id` 去重(列上有唯一键 ⇒ 与行数恒等)就是
    # 计划订正 17-A 修掉的那一版。
    "M2": {
        "path": "app/api/topics.py",
        "why": "把「不同问题数」退回按 low_confidence_question_id 去重(17-A 修掉的那一版)⇒ 验收 ② 必须红",
        "expect": "②:不同问题数 65 != 回池子按题面数的 33",
        "edits": [
            ("COUNT(DISTINCT q.question) AS distinct_n",
             "COUNT(DISTINCT t.low_confidence_question_id) AS distinct_n"),
            ("COUNT(DISTINCT q.question) AS distinct_questions",
             "COUNT(DISTINCT t.low_confidence_question_id) AS distinct_questions"),
        ],
    },
    # ★ **复审 F1 的原始破坏**:「跑批印 0 条空标签,而库里真的写了空标签」。
    # 判据此前读的是**脚本自己印的数** ⇒ 整节 54/0 全绿(库侧的独立证据就在同一份转录里:
    # `各 bucket 条数合计` 79→78)。订正轮 1 起判据拿**库里的实际行数**比。
    "M4": {
        "path": "scripts/classify_topics.py",
        "why": "跑批写库时把 labels 一律写成空(而印出来的空标签数仍是 0)⇒ 验收 ② 必须红",
        "expect": "②:跑批自报 0 条,而库里的空标签行数 != 0",
        "edits": [(
            '            "labels": list(row["labels"]),',
            '            "labels": [],',
        )],
    },
    # ★ **复审 F2 的原始破坏**:「落库变空操作」。日志照印「已写 20 行」,
    # 而库里一条新写都没有;其余判据看的都是表的**总量** ⇒ 池子早已全归过类时总量不变。
    "M5": {
        "path": "scripts/classify_topics.py",
        "why": "把 upsert 的 execute 摘掉(落库变空操作,而打印照旧)⇒ 验收 ② 必须红",
        "expect": "②:库里最新的 classified_at 早于本轮开始前的库时钟",
        "edits": [(
            "    await session.execute(stmt)\n    return len(values)",
            "    return len(values)",
        )],
    },
    # ① 的切分算术是「合计 = 输入 − 不可信靶子」。从 train 里抽掉一行 ⇒ 合计少 1,
    # 而那条「落进三份的 id 集合 == 靶子可信的输入集合」也会跟着报出丢了哪个 id。
    "M3": {
        "path": "evals/topic/train.jsonl",
        "why": "从 train.jsonl 里抽掉第一行 ⇒ 验收 ① 的切分算术必须红",
        "expect": "①:合计 1422 != 输入 1424 − 不可信靶子 1 / id 集合差 1 个",
        "drop_first_line": True,
    },
}


def check_bash() -> None:
    """**钉住 Git Bash**:`uname -s` 必须是 MINGW*/MSYS*/CYGWIN*(见 `BASH` 那段注释)。

    只核「路径里有 Git」不够 —— 得问那个 shell **自己是谁**。
    """
    out = subprocess.run([BASH, "-c", "uname -s; echo OSTYPE=$OSTYPE"],
                         capture_output=True, text=True, timeout=120).stdout
    first = out.splitlines()[0].strip() if out else ""
    if not first.startswith(("MINGW", "MSYS", "CYGWIN")):
        raise SystemExit("!!! 这个 bash 不是 Git Bash(uname -s = %r,%s)——"
                         "WSL 的 bash 里 localhost 与 /tmp 都不是 Windows 这边的,"
                         "跑出来的红全是假红。用 CH10_BASH=<git-bash 路径> 指一个。"
                         % (first, out.strip()))
    emit("   shell 自检:bash=%s / %s 中的 %s" % (BASH, first, out.splitlines()[1].strip()))


def kill_stale_ports() -> None:
    """跑完一轮之后,把四个端口上的残留进程收干净(下一轮的预检要求它们是空的)。"""
    for port in (8000, 8101, 8102, 8103):
        try:
            out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                                 timeout=60).stdout
        except subprocess.TimeoutExpired:
            return
        pids = {line.split()[-1] for line in out.splitlines()
                if re.search(r":%d\s+\S+\s+LISTENING" % port, line)}
        for pid in pids:
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            "Stop-Process -Id %s -Force" % pid],
                           capture_output=True, timeout=60)


def summarize(transcript: Path) -> None:
    """把转录里的判词摘出来:**红在哪一条**(带它属于哪一节)。"""
    text = transcript.read_text(encoding="utf-8", errors="replace")
    section = "(起环境)"
    reds = []
    for line in text.splitlines():
        if line.startswith("== 验收") or line.startswith("== 起环境"):
            section = line.strip("= ")
        elif re.match(r"^  (FAIL|!!!!) ", line):
            reds.append("%s  ← %s" % (line.strip(), section))
    count = re.search(r"通过 (\d+) / 失败 (\d+) / 未复现 (\d+)", text)
    emit("  判词:%s" % (count.group(0) if count else "(没有判词行 —— 脚本没跑到收尾)"))
    if reds:
        emit("  **红在这些行(%d 条)**:" % len(reds))
        for line in reds[:12]:
            emit("    " + line)
        if len(reds) > 12:
            emit("    ...(还有 %d 条,全文见 %s)" % (len(reds) - 12, transcript.name))
    else:
        emit("  **(一条红都没有 —— 这个变异没被判出来!)**")


def run_product(key: str) -> int:
    mutation = PRODUCT_MUTATIONS[key]
    path = REPO / mutation["path"]
    original = path.read_bytes()
    before = sha(path)
    backup = EVID / ("ch10b_t16_mut_backup_" + key + ".bak")
    backup.write_bytes(original)
    transcript = EVID / ("ch10b_t16_mut_" + key + ".txt")
    extract_helper()
    emit("== 产品级变异 %s ==" % key)
    emit("   改:%s" % mutation["path"])
    emit("   为什么:%s" % mutation["why"])
    emit("   期望红在:%s" % mutation["expect"])
    emit("   变异前 sha256=%s(备份 %s)" % (before[:16], backup.name))
    try:
        if "drop_first_line" in mutation:
            text = original.decode("utf-8")
            head, sep, rest = text.partition("\n")
            path.write_bytes(rest.encode("utf-8"))
        else:
            text = original.decode("utf-8")
            for old, new in mutation["edits"]:
                hits = text.count(old)
                if hits != 1:
                    raise SystemExit("!!! 锚点在 %s 里命中 %d 次(必须恰好 1 次)"
                                     "—— 命中 0 次与「断言无判别力」长得一样,不猜"
                                     % (mutation["path"], hits))
                text = text.replace(old, new, 1)
            path.write_bytes(text.encode("utf-8"))
        emit("   变异后 sha256=%s" % sha(path)[:16])

        check_bash()
        kill_stale_ports()
        t0 = time.time()
        with open(transcript, "wb") as fh:
            try:
                proc = subprocess.run([BASH, ACCEPTANCE], cwd=REPO, stdout=fh,
                                      stderr=subprocess.STDOUT,
                                      timeout=ACCEPTANCE_TIMEOUT)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = "TIMEOUT"
        emit("   验收退出码=%s,耗时 %.0fs,转录 %s" % (rc, time.time() - t0, transcript.name))
        summarize(transcript)
        return 0 if rc not in (0, "TIMEOUT") else 1
    finally:
        # ⚠️ **字节还原 + 当场核对 sha256**(不许 `git checkout --`:autocrlf 会把它
        #    改成 CRLF —— 那不是还原)。
        path.write_bytes(original)
        kill_stale_ports()
        now = sha(path)
        emit("   还原 sha256=%s %s" % (now[:16], "✅ 与变异前一致" if now == before
                                    else "❌ **与变异前不一致 —— 残留了!**"))


# ══════════════════════════════════════════════════════════════════════════
# 判词级探针(喂错的输入,问某一条判据是不是恒真)
# ══════════════════════════════════════════════════════════════════════════

def _scratch_report_dirs() -> tuple[Path, Path, Path]:
    """造一组「两次运行的五份产物」。**直接用入库的那五份当源** ——
    它们与一次全新运行逐字节相同(① 自己会断这件事),所以不必真跑两遍模型。"""
    base = TMP / "ch10_t16_judge"
    run, copy = base / "run", base / "copy"
    for d in (run, copy):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for name in ("report.md", "report.json", "matrix_confusion.csv",
                     "matrix_flow.csv", "misjudged.csv"):
            shutil.copyfile(REPO / "evals" / "topic" / name, d / name)
    return run, copy, TMP / "ch10_t16_judge" / "test_copy.jsonl"


def _run_helper(*args) -> tuple[int, str]:
    extract_helper()
    proc = subprocess.run([PY, str(HELPER), *args], cwd=REPO, capture_output=True)
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def _report_args(run: Path, copy: Path, test: Path):
    return ("report-check", "--run", str(run), "--copy", str(copy),
            "--committed", str(REPO / "evals" / "topic"),
            "--split-dir", str(REPO / "evals" / "topic"),
            "--prelabeled", str(REPO / "evals" / "topic" / "prelabeled.jsonl"),
            "--test", str(test))


def judge_report_identity() -> int:
    """J1:两次运行的产物差一个字节 ⇒ 「逐字节相同」那条**必须**判红。"""
    run, copy, test = _scratch_report_dirs()
    shutil.copyfile(REPO / "evals" / "topic" / "topic_test.jsonl", test)
    data = (run / "report.md").read_bytes()
    (run / "report.md").write_bytes(data.replace(b"micro-F1", b"micro-F2", 1))
    rc, out = _run_helper(*_report_args(run, copy, test))
    emit(out.strip())
    hit = "不是**逐字节相同:report.md" in out
    emit("==> %s" % ("✅ 判红在「两次运行逐字节相同:report.md」" if hit
                    else "❌ **没判出来!**"))
    return 0 if hit else 1


def judge_report_number_crosscheck() -> int:
    """J2:把 `report.json` 里的 micro-F1 改掉 ⇒ 「印出来的数 == 机器可读的数」必须红。

    这条问的是那个最危险的假绿形态:一份**内容全错**的报告照样含 `micro-F1` 四个字母
    ⇒ 只 grep 关键词是零判别力的。
    """
    run, copy, test = _scratch_report_dirs()
    shutil.copyfile(REPO / "evals" / "topic" / "topic_test.jsonl", test)
    payload = json.loads((run / "report.json").read_text(encoding="utf-8"))
    payload["grid"]["0.5"]["all"]["micro_f1"] = 0.9999
    (run / "report.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    rc, out = _run_helper(*_report_args(run, copy, test))
    emit(out.strip())
    hit = "对不上" in out and "FAIL" in out
    emit("==> %s" % ("✅ 判红在「报告里印的 micro-F1 与 report.json 对不上」" if hit
                    else "❌ **没判出来!**"))
    return 0 if hit else 1


def judge_report_test_sha() -> int:
    """J3:测试集文件变了一点 ⇒ 「报告绑定的测试集 == 盘上那份」必须红。"""
    run, copy, test = _scratch_report_dirs()
    data = (REPO / "evals" / "topic" / "topic_test.jsonl").read_bytes()
    test.write_bytes(data + data.splitlines(keepends=True)[0])   # 多出一行
    rc, out = _run_helper(*_report_args(run, copy, test))
    emit(out.strip())
    hit = "测试集变过" in out
    emit("==> %s" % ("✅ 判红在「测试集 sha256 与报告记录的不符」" if hit else "❌ **没判出来!**"))
    return 0 if hit else 1


def _fake_batch_log(limit: int = 20, total: int = 65, distinct: int = 33,
                    empty: int = 0) -> Path:
    path = TMP / "ch10_t16_judge" / "classify.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "服务 http://127.0.0.1:8103:labels=17 threshold=0.5 max_length=64\n"
        "model_version = v1\n"
        "池子里读到 %d 行(--limit %d)\n"
        "分批 1 次,归类 %d 行,其中**空标签 %d 条**\n"
        "已写 %d 行;本次空标签 %d 条\n"
        "分布页会看到:总行数 %d / 已归类的池子行数 %d\n"
        % (limit, limit, limit, empty, limit, empty, total, distinct),
        encoding="utf-8")
    return path


def judge_dist_semantics() -> int:
    """J4:喂一份「不同问题数 == total」的接口响应(17-A 那一版的样子)⇒ 必须红。"""
    base = TMP / "ch10_t16_judge"
    base.mkdir(parents=True, exist_ok=True)
    log = _fake_batch_log()
    # 用真实形状的响应,只把 distinct_questions 改成 total(那一版的行为)
    from app.topic.taxonomy import LABELS  # noqa: E402
    payload = {"total": 65, "distinct_questions": 65, "last_classified_at": "2026-09-27 08:00",
               "model_versions": ["v1"],
               "buckets": [{"label": lb, "count": 0, "distinct": 0} for lb in LABELS]}
    dist = base / "dist_fake.json"
    dist.write_text(json.dumps(payload), encoding="utf-8")
    rc, out = _run_helper("dist-check", "--batch-log", str(log), "--batch-rc", "0",
                          "--dist", str(dist), "--curl-rc", "0", "--limit", "20",
                          # 这一条探针只问「语义判据」——库时钟给个必然满足的值,
                          # 免得「这一轮真的写过库」那条也红(两条一起红就分不清是谁抓到的)。
                          "--db-start", "1970-01-01 00:00:00")
    emit(out.strip())
    hit = "没意义的数" in out
    emit("==> %s" % ("✅ 判红在「不同问题数退回按 low_confidence_question_id 去重」" if hit
                    else "❌ **没判出来!**"))
    return 0 if hit else 1


def judge_predict_multilabel() -> int:
    """J5:喂一份**只命中一类**的 /predict 响应(**scores 自洽**)⇒ ③ 的承重判据必须红。

    ⚠️ 关键在于**把 scores 也改自洽**:这样「标签 == 逐标签过阈值」那条仍然绿,
    红的是**唯一**该红的承重断言 —— 否则两个判据一起红,分不清是谁抓到的。
    """
    base = TMP / "ch10_t16_judge"
    base.mkdir(parents=True, exist_ok=True)
    from app.topic.model import load_artifacts  # noqa: E402
    from app.topic.taxonomy import LABELS  # noqa: E402
    art = load_artifacts(str(REPO / "models" / "topic-clf"))
    thr = float(art["threshold"])
    labels = list(LABELS)
    # 第一句:「退换货」「尺码」都过阈值 —— 这正是实测那一行
    scores = {lb: 0.01 for lb in labels}
    scores["退换货"], scores["尺码"] = 0.85, 0.81
    # 但把标签**只留一个**(模拟 argmax 退化),同时把另一个的概率压到阈值下面去,
    # 让「标签 == 逐标签过阈值」这一条仍然成立。
    scores["尺码"] = 0.01
    request = base / "predict_req.json"
    request.write_text(json.dumps({"texts": ["买大了想退"]}, ensure_ascii=False),
                       encoding="utf-8")
    predict = base / "predict_fake.json"
    predict.write_text(json.dumps({"results": [{"labels": ["退换货"], "scores": scores}]},
                                  ensure_ascii=False), encoding="utf-8")
    healthz = base / "healthz_fake.json"
    healthz.write_text(json.dumps({"status": "ok", "labels": labels, "num_labels": len(labels),
                                   "threshold": thr, "max_length": art["max_length"],
                                   "train_meta": art["meta"]}, ensure_ascii=False),
                       encoding="utf-8")
    rc, out = _run_helper("predict-check", "--predict", str(predict), "--healthz", str(healthz),
                          "--request", str(request), "--model-dir", str(REPO / "models" / "topic-clf"))
    emit(out.strip())
    hit = "只命中 1 个类目" in out and "FAIL" in out
    # 「标签 == 逐标签过阈值」这一条**不该**红(scores 是自洽的)——
    # 两个判据一起红就分不清是谁抓到的,所以这里把它也核一遍。
    isolated = out.count("FAIL ") == 1
    emit("==> %s%s" % ("✅ 判红在「多诉求句只命中 1 个类目(期望 >= 2)」" if hit
                      else "❌ **没判出来!**",
                      "(且只有这一条红)" if isolated else "(⚠️ 不止一条红,归因不干净)"))
    return 0 if hit else 1


JUDGES = {
    "J1": ("report-check:两次运行的产物差一字节", judge_report_identity),
    "J2": ("report-check:report.json 里的数被改掉", judge_report_number_crosscheck),
    "J3": ("report-check:测试集文件变了一点", judge_report_test_sha),
    "J4": ("dist-check:不同问题数退回旧语义", judge_dist_semantics),
    "J5": ("predict-check:只命中一类的响应(承重断言)", judge_predict_multilabel),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ch10-B T16 变异探针")
    ap.add_argument("--product", choices=sorted(PRODUCT_MUTATIONS))
    ap.add_argument("--judge", choices=sorted(JUDGES))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for key, m in PRODUCT_MUTATIONS.items():
            emit("产品 %s: %s  —— %s" % (key, m["path"], m["why"]))
        for key, (name, _) in JUDGES.items():
            emit("判词 %s: %s" % (key, name))
        return 0
    if args.product:
        return run_product(args.product)
    if args.judge:
        emit("== 判词级探针 %s:%s ==" % (args.judge, JUDGES[args.judge][0]))
        return JUDGES[args.judge][1]()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
