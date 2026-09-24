
---

# fix round 1(复审 4 条 Important + 5 条 Minor)

复审结论:**Spec ✅(附一个条件)、Quality 需修**。四条 Important 的处置与证据如下。

## I-1(已改)—— 补「按桶」视图,并且**默认就打**

**为什么这是功能缺陷而不是格式取舍**(复审用真跑数据算出来的):纯dense 的
`recall@10` 分桶值是 **A 0.817 / B 1.000 / C 0.933 / D 0.000 / E 0.983**,
头条值 **0.747**,而四个**可答**桶的平均是 **0.933**。`D_absent`(应拒答桶,n=60,
占 1/5)的 recall/mrr **结构性为 0**(它没有「正确章节」可召回)⇒ 它把每个头条值
**稀释约 1/5**:一个只在 A_policy 上发生的 **0.067** 真实回退,在头条上只动 **0.013**。
而且 `answer_ok`(拒答桶**唯一**有意义的指标)在头条里根本不显示 ⇒
**拒答行为的回退完全看不见**。需求原话是「哪个指标在下滑要一眼看出来」——
丢掉分桶就是把这个功能本身打掉。

**改法**:`render()` 现在打**两张表**(模块 docstring 里写了理由与那几个数):

1. `---- 每策略(跨桶加权;分桶见下)----` —— brief 草图那张,原样保留;
2. `---- 按桶 · recall@10 ----` / `---- 按桶 · mrr ----` / `---- 按桶 · answer_ok ----`
   —— 列是 **轮次 / 时间 / 条数 / 桶 / 各策略**,一行 = (轮次 × 桶),差值**内联**在格子里。
   `answer_ok` 比 spec §10.2 的「各桶 Recall@10 / MRR」多一列,是**刻意的**
   (`D_absent` 上只有它有意义)—— 这一处偏离已写进模块 docstring。

**默认开,不是开关**:`render()` 没有参数控制它;`--last N` 只截历史长度。
`tests/test_eval_trend.py::test_bucket_view_is_on_by_default_and_shows_the_dilution`
把「默认就打」与「稀释可见」一起钉住(M6 变异下必红)。

**实跑输出**(`.superpowers/t17/trend_fix1.txt`,命令
`.venv/Scripts/python.exe scripts/eval_trend.py`;为放进报告去掉了每行末尾的补齐空格,
逐字节原件在 `trend_fix1.txt`):

```
===== 评估趋势 =====
（↑/↓ = 相对上一轮的增减;`=` = 没变;第 1 轮没有可比对象、不打箭头。**条数或 top_k 与上一轮不同的两轮标「不可比」并整个不打箭头** —— 分母不同的两个分数之间的差不是「变化」。`-` = 该轮没有这个数(例如权重不在时跳过了「混合+Rerank」),**不是 0**。）

---- 每策略(跨桶加权;分桶见下)----
轮次  时间                 条数    纯dense                    纯BM25                     混合(RRF)                  混合+Rerank
  1   2026-09-24 14:07:25  300     recall@10=0.747 mrr=0.652  recall@10=0.710 mrr=0.587  recall@10=0.747 mrr=0.634  recall@10=0.747 mrr=0.671
  2   2026-09-24 14:08:11  5       recall@10=1.000 mrr=0.900  recall@10=1.000 mrr=0.850  recall@10=1.000 mrr=0.767  recall@10=1.000 mrr=0.800
                                   ≠ 与上一轮不可比（条数 300→5）:本条不打印增减
  3   2026-09-24 14:08:44  5       recall@10=1.000 mrr=0.900  recall@10=1.000 mrr=0.850  recall@10=1.000 mrr=0.767  recall@10=1.000 mrr=0.800
                                   = 0.000         = 0.000    = 0.000         = 0.000    = 0.000         = 0.000    = 0.000         = 0.000

---- 按桶 · recall@10 ----
轮次  时间                 条数    桶            纯dense         纯BM25          混合(RRF)       混合+Rerank
  1   2026-09-24 14:07:25  300     A_policy      0.817           0.817           0.817           0.817
                                                 B_model       1.000           1.000           1.000           1.000
                                                 C_colloquial  0.933           0.750           0.917           0.917
                                                 D_absent      0.000           0.000           0.000           0.000
                                                 E_multi       0.983           0.983           1.000           1.000
                                                 ≠ 与上一轮不可比（条数 300→5）:本轮不打印增减
  2   2026-09-24 14:08:11  5       A_policy      1.000           1.000           1.000           1.000
                                                 B_model       -               -               -               -
                                                 C_colloquial  -               -               -               -
                                                 D_absent      -               -               -               -
                                                 E_multi       -               -               -               -
  3   2026-09-24 14:08:44  5       A_policy      1.000 = 0.000   1.000 = 0.000   1.000 = 0.000   1.000 = 0.000
                                                 B_model       -               -               -               -
                                                 C_colloquial  -               -               -               -
                                                 D_absent      -               -               -               -
                                                 E_multi       -               -               -               -

---- 按桶 · mrr ----
轮次  时间                 条数    桶            纯dense         纯BM25          混合(RRF)       混合+Rerank
  1   2026-09-24 14:07:25  300     A_policy      0.747           0.701           0.724           0.728
                                                 B_model       0.924           0.992           0.967           0.992
                                                 C_colloquial  0.759           0.463           0.655           0.823
                                                 D_absent      0.000           0.000           0.000           0.000
                                                 E_multi       0.830           0.779           0.822           0.814
                                                 ≠ 与上一轮不可比（条数 300→5）:本轮不打印增减
  2   2026-09-24 14:08:11  5       A_policy      0.900           0.850           0.767           0.800
                                                 B_model       -               -               -               -
                                                 C_colloquial  -               -               -               -
                                                 D_absent      -               -               -               -
                                                 E_multi       -               -               -               -
  3   2026-09-24 14:08:44  5       A_policy      0.900 = 0.000   0.850 = 0.000   0.767 = 0.000   0.800 = 0.000
                                                 B_model       -               -               -               -
                                                 C_colloquial  -               -               -               -
                                                 D_absent      -               -               -               -
                                                 E_multi       -               -               -               -

---- 按桶 · answer_ok ----
轮次  时间                 条数    桶            纯dense         纯BM25          混合(RRF)       混合+Rerank
  1   2026-09-24 14:07:25  300     A_policy      0.817           0.817           0.817           0.817
                                                 B_model       1.000           1.000           1.000           1.000
                                                 C_colloquial  0.933           0.750           0.917           0.917
                                                 D_absent      1.000           1.000           1.000           1.000
                                                 E_multi       0.983           0.983           1.000           1.000
                                                 ≠ 与上一轮不可比（条数 300→5）:本轮不打印增减
  2   2026-09-24 14:08:11  5       A_policy      1.000           1.000           1.000           1.000
                                                 B_model       -               -               -               -
                                                 C_colloquial  -               -               -               -
                                                 D_absent      -               -               -               -
                                                 E_multi       -               -               -               -
  3   2026-09-24 14:08:44  5       A_policy      1.000 = 0.000   1.000 = 0.000   1.000 = 0.000   1.000 = 0.000
                                                 B_model       -               -               -               -
                                                 C_colloquial  -               -               -               -
                                                 D_absent      -               -               -               -
                                                 E_multi       -               -               -               -
```

**复审举的那几个数现在表上直接看得见**:`D_absent` 那一行 recall@10/mrr 是 `0.000`
而同表的 `answer_ok` 是 `1.000`;`A_policy` 是 `0.817`,头条却是 `0.747` ——
稀释不再是读者要自己算出来的东西。

## I-2(已改)—— 给 `render()` 补测试,并**撤回**「不可单测」

新增 `tests/test_eval_trend.py`(**无 db 标记**,不连库),7 条用例,覆盖复审点名的四条
+ 三条我加的:

| 用例 | 断的东西 |
|---|---|
| `test_first_round_has_no_arrow` | 第一轮不打箭头,但那几个数确实打出来了 |
| `test_legend_itself_would_fool_a_naive_grep` | **图例那行自己含 ↑/↓/=/`-`** ⇒ 对整份输出 grep 是恒真的(把 I-4 的坑钉在测试里) |
| `test_comparable_rounds_print_signed_deltas` | 同规模同 top_k:头条差值 `↓ -0.020` / `↑ +0.010`,分桶格子里 `0.700 ↓ -0.100` / `0.950 ↑ +0.050`(全是手算得出的具体数) |
| `test_incomparable_case_count_prints_marker_and_no_arrows` | `≠ …条数 300→60` 出现,且 **body 里一个箭头都没有** |
| `test_incomparable_top_k_prints_marker_and_no_arrows` | `top_k 10→5` 同样判不可比 |
| `test_missing_strategy_renders_dash_not_zero` | 缺席的那一格是 `-`,**不是 `0.000`** |
| `test_bucket_view_is_on_by_default_and_shows_the_dilution` | 分桶表默认存在;A_policy `0.800` vs 头条 `0.740`;D_absent recall@10 `0.000` 而 answer_ok `1.000` |

断言走的是**真** `_row_view`(喂 `SimpleNamespace` 假行)+ 真 `render()`,
不是另写一份视图构造。`_body()` 先滤掉图例行再断言 —— 这就是 I-4 那条的落地。

**「不可单测」这句撤回**:错因值得记 —— **我当时的判据是「它是个格式化报告」,
而真正相关的是「它是不是纯函数」**。`scripts.eval_trend` 不连库也能 import
(`get_engine()` 只在 `main()` 里调),`render()` 只吃 dict ⇒ 约 30 行就测完了。
代价当场就显出来了:**M4(符号翻转)在初版只有「输出变了」而没有红** ——
按本仓的变异规矩,「变异后没红」只有两种解释,这里就是**没有断言存在**。

## I-3(已改)—— 证据文件进版本控制

`git ls-files .superpowers` 原本 **11** 个(本章前面的任务刻意跟踪的探针),
而 `.superpowers/t17/` 一个都没跟踪;`.superpowers/sdd/` 是被 gitignore 的
(`.superpowers/sdd/.gitignore` 里一行 `*`),`.superpowers/` 本身**不是** ⇒
不提交就是「分支一合、承重证据全没」。
⇒ 本次 commit 把 `.superpowers/t17/` **整目录**加进去(见 `git show --stat`)。
§7 里「全部不进仓库」那句已订正。

## I-4(已改)—— 给 T18 的验收锚点建议是恒真的假断言

原文建议 `grep '↓|↑|='`。**表头那行图例自己就含全部四个字形**
(`eval_trend.py` 的 `LEGEND`),所以那个 grep 在「一条增量行都没打出来」时**照样匹配**。
⇒ 已改成**按行形状判**(§6 第 2 条给了可抄的 Python 片段:`ln.lstrip()[:1] in ("↓","↑","=","≠")`),
并且把这条陷阱本身写成了一个测试(`test_legend_itself_would_fool_a_naive_grep`)。

## Minor

- **M-1(已改)**:`run_eval.py` 的模块 docstring 补上前置 —— **`eval_runs` 必须已存在**
  (`init_db.py` 跑过,或 `db/ch09.sql` 应用过),并写明漏掉的失败形态:
  全量跑一两分钟、`latest.json` 写完、对照表打完,**然后插库时 1146 炸** ⇒
  一次实际成功的评估拿到非零退出码。**刻意不改成「只警告不报错」**(那就违反 H2)。
- **M-3(已改)**:图例补上 `-` 的含义;新增 `--last N`(默认 0 = 全部)。
  **语义写清楚了**:它是**截断历史**,截后的第一轮按「没有上一轮」处理、不打箭头 ——
  这样**表上每个箭头都能用表上的数字手算复核**(H1 不退让);想看那一轮的增减就把 N 调大。
  实跑:`--last 2` 只剩两轮且重新编号 1/2(见 `trend_last2.txt`)。
- **M-5(已改)**:§2.2「两轮输出逐字节相同」→ 订正为「**结果**逐字节相同,输出不是
  (含 `6.9s` vs `6.6s`)」;§2.3「三个策略、六个指标」→ 订正为「四个策略、八个指标」。
  两处都留了痕(不静默改)。
- **M-2(记账,不改)**:`_width` 只把 `W`/`F` 算 2 格,`↓`/`↑`/`⚠`(East-Asian
  **Ambiguous**)按 1 格算 ⇒ 在把 Ambiguous 渲染成宽字符的终端上每个箭头后错一列。
  ⚠️ **并且「列对齐是核过的」这句话要打折**:核验用的就是产者那份 `_width`,是**自指**的,
  查不出这一类。已写进模块 docstring 的「已知限制」,属外观级。
- **M-4(记账,不改)**:`_incomparable_reason` 只看 `case_count`/`top_k` ⇒
  同规模同 top_k 但**标尺换了**(用例文件被就地改过、知识库变了)会被判成可比并打箭头。
  要真闭掉得往 `metrics` 里再加一个「用例集摘要」键(§10.3 之外的新键,没做);
  今天暴露面有限(`--limit` 总是取**前 N** 条,不换样本)。已写进模块 docstring。

## 五条变异重跑(现在 6 条,**全部有红**)

`.superpowers/t17/mutate.py`(基线先各跑一遍确认全绿:`test_eval_runs_write.py` 2 passed、
`test_eval_trend.py` 7 passed),输出全文 `mutate_out2.txt`:

| 变异 | 锚点命中 | 判定 | 具体红了哪条 |
|---|---|---|---|
| M1 顶层 `case_count` 取全量 | 1 | **RED** | `test_eval_run_row_carries_case_count_separately`(`assert 300 == 5`) |
| M2 `metrics.case_count` 取全量 | 1 | **RED** | 同上(嵌套键那一条断言) |
| M3 先落库、再评估 | 1 | **RED** | `test_failed_round_writes_no_row` |
| M4 差值符号取反 | 1 | **RED**(这一轮才有的) | `test_comparable_rounds_print_signed_deltas` |
| M5 删掉「不可比」分支 | 1 | **RED** | `test_incomparable_case_count_…` + `test_incomparable_top_k_…` |
| M6 分桶表不再默认渲染 | 1 | **RED** | `test_comparable_rounds_print_signed_deltas` + `test_bucket_view_is_on_by_default_and_shows_the_dilution` |

**M4 的对比最说明问题**:初版它是「输出变了、没有红」,现在是
`1 failed, 6 passed` 且失败点指名道姓。M5/M6 是新加的渲染变异,守住的是本轮新增的两条性质。

## 这一轮的测试真数

- `tests/test_eval_trend.py` → **7 passed**(新增)
- `tests/test_eval_runs_write.py` → **2 passed**
- 全量 `pytest` → **963 passed**(956 + 7),见 `fullsuite_fix1.txt`
