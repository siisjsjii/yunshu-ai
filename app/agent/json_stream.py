"""增量 JSON 解码器 —— 边收 token 边解出 ``useful`` / ``confidence`` / ``answer``。

ch09 spec §5.3 / §5.4。**纯状态机**:输入 ``str`` 片段序列,输出事件序列,
零 IO、零 await、零随机、零第三方依赖 ⇒ 可以用 chunk 边界 fuzz **穷举钉死**。

**为什么自写而不用 partial-json-parser / ijson / json-stream**:
本章只需要两件事 —— ①从头部取出 ``useful``;②把 ``answer`` 的值边收边吐。
而 ``answer`` 的值**必然含中文**,必然出现 ``\\"`` ``\\\\`` ``\\n`` ``\\uXXXX``,
而**转义序列跨 chunk 边界**恰恰是这类通用库最难验的地方(一个 ``中`` 被切成
``\\u4e`` + ``2d`` 是**必现**的,不是边角)。自写的东西能被穷举切点 fuzz 钉死,
通用库我们只能信它。

三态(spec §5.4):

``lead``
    还没定形态。首个**非空白**字符是 ``{`` ⇒ ``protocol``;不是 ⇒ ``plain``。
    **全空白/空片段留在 lead**(边界 13),什么都不 emit(不变量 1)。
``protocol``
    在解协议对象。首键不是 ``useful`` ⇒ ``violation``(调用方据此重试一次)。
``plain``
    **今天的行为**(ch08 的纯文本流):每个片段原样透出。它存在的意义是让协议
    违规**不产生任何回归** —— 这是本章的兜底不变量。

两条不变量(§5.4,都有测试):
    1. ``lead`` 态下什么都不 emit —— 用户此刻看不到任何字。
    2. ``useful=false`` 之后,一个 ``answer_delta`` 都不许再出去。**在解出
       ``useful`` 为 ``False`` 的那一刻就停止消费**(``_finish_scalar`` 里置
       ``_done``),后续 ``feed`` 直接返回空表 —— 这是违规流(边界 11)唯一
       被守住的理由,因为正常路径下 ``answer`` 本来就是空串,**没有可吐的东西**。
"""

from dataclasses import dataclass

#: 三态的取值,别写成裸字符串(调用方与测试都要引用)
LEAD = "lead"
PROTOCOL = "protocol"
PLAIN = "plain"

#: JSON 的简转义(``\uXXXX`` 不走这张表,它要攒够 4 位十六进制)
_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

_HEX_DIGITS = "0123456789abcdefABCDEF"


def _value_kind(ch: str) -> str:
    """只看**值的首字符**给违规文案起个类型名(够诊断用,不求严谨)。"""
    if ch == "{":
        return "object"
    if ch == "[":
        return "array"
    if ch in "-0123456789":
        return "number"
    if ch in "tf":
        return "bool"
    if ch == "n":
        return "null"
    return "non-string"

# ---- protocol 态下字符级扫描的四个位置(字符串内另由 _in_string 表示)----
_EXPECT_KEY = "expect_key"      # 对象内:等键名的开引号,或对象收尾的 `}`
_EXPECT_COLON = "expect_colon"  # 键名读完,等 `:`
_EXPECT_VALUE = "expect_value"  # 等值的第一个字符
_IN_SCALAR = "in_scalar"        # 非字符串值(数字 / true / false / null)收集中


@dataclass(frozen=True)
class Event:
    kind: str            # useful | confidence | answer_delta | done | violation
    value: object = None


class JsonAnswerDecoder:
    """⚠️ **追加**语义:``feed`` 按到达顺序喂,不是替换。

    ``answer_delta`` 的 ``value`` 是**本次新增**的那段文本,不是累计。
    累计全文看 ``answer``。

    ``Event("done")`` 在两种情况下发出,与 ``done`` 属性完全同步:
    ①协议对象收到收尾的 ``}``;②``useful`` 解出为 ``False`` 的那一刻
    (此后一个 delta 都不再出去)。调用方用 ``useful`` 区分这两支。
    """

    def __init__(self) -> None:
        self._mode = LEAD
        self._raw = ""               # 收到过的全部原文
        self._lead = ""              # lead 态攒下的片段(定形态前不吐)
        self._useful: bool | None = None
        self._confidence: float | None = None
        self._answer = ""
        self._done = False
        self._violation: str | None = None
        # ---- 字符级扫描进度 ----
        self._state = _EXPECT_KEY
        self._in_string = False
        self._is_key = False         # 当前这段字符串是键名还是值
        self._string_buf = ""        # 键名 / 非 answer 的字符串值的字符
        self._scalar_buf = ""        # 非字符串值的原文
        self._cur_key: str | None = None   # 当前这个值属于哪个键
        self._saw_key = False        # 协议对象里是否已出现第一个键
        self._depth = 0              # 花括号深度:只有回落到 0 的那个 `}` 才收尾
        self._pending_high: str | None = None   # 扣住的高代理,等可能跟着的低代理
        self._escape = False         # 上一个字符是字符串内的 `\`
        self._unicode_hex: str | None = None   # 非 None = 正在攒 \uXXXX
        self._out: list[Event] = []  # 本次 feed 的事件出口

    # ---- 只读视图 ---------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def useful(self) -> bool | None:
        return self._useful

    @property
    def confidence(self) -> float | None:
        return self._confidence

    @property
    def answer(self) -> str:
        return self._answer

    @property
    def done(self) -> bool:
        return self._done

    @property
    def raw(self) -> str:
        return self._raw

    @property
    def violation(self) -> str | None:
        return self._violation

    # ---- 主入口 -----------------------------------------------------------
    def feed(self, fragment: str) -> list[Event]:
        # 原文无条件记账(它的语义是「收到过的全部」,不受停机影响)
        self._raw += fragment
        if not fragment or self._done or self._violation:
            return []
        if self._mode == PLAIN:
            return [Event("answer_delta", fragment)]
        if self._mode == LEAD:
            return self._feed_lead(fragment)
        return self._feed_protocol(fragment)

    # ---- lead -------------------------------------------------------------
    def _feed_lead(self, fragment: str) -> list[Event]:
        self._lead += fragment
        stripped = self._lead.lstrip()
        if not stripped:
            return []                     # 全是空白:留在 lead,不判定
        if stripped[0] != "{":
            # 不是协议对象(`好的,` / ```json 围栏 / 任何前言)⇒ 降级成今天的行为。
            # 把**攒下的**与**这一片**作为一段一起吐出去,顺序与原文一致。
            self._mode = PLAIN
            buffered, self._lead = self._lead, ""
            return [Event("answer_delta", buffered)]
        # 首个非空白是 `{` ⇒ 是协议对象。缓冲区里 `{` 之前只可能是空白,
        # 没有需要 flush 的字节。
        self._mode = PROTOCOL
        pending, self._lead = self._lead, ""
        return self._feed_protocol(pending)

    # ---- protocol ---------------------------------------------------------
    def _feed_protocol(self, fragment: str) -> list[Event]:
        self._out = []
        for ch in fragment:
            self._consume(ch)
            if self._done or self._violation:
                break
        return self._out

    def _consume(self, ch: str) -> None:
        # ---------- 字符串内(键名 / 字符串值,共用一套转义处理)----------
        if self._in_string:
            self._consume_in_string(ch)
            return

        # ---------- 非字符串值收集中 ----------
        if self._state == _IN_SCALAR:
            if ch.isspace():
                self._finish_scalar()
                self._state = _EXPECT_KEY
                return
            if ch in ",}":
                self._finish_scalar()
                self._state = _EXPECT_KEY
                # ⚠️ `_finish_scalar` 可能已经收过尾(`useful=false` 或**违规**)⇒
                # 直接 return:`done` 至多发一次,违规之后也不许再收尾。
                if self._done or self._violation:
                    return
                # 不 return:让 `,` / `}` 落到下面被同一轮处理
                # (`}` 那一路会先过**深度闸**,见文末那一段)
            elif ch == "{":
                # 标量里冒出 `{`:畸形。丢掉半截标量,把它当结构字符处理。
                self._scalar_buf = ""
                self._state = _EXPECT_KEY
            else:
                self._scalar_buf += ch
                return

        # ---------- 字符串外:按位置分派 ----------
        if self._state == _EXPECT_KEY:
            if ch == '"':
                self._begin_string(is_key=True)
                return
            if ch == "," or ch.isspace():
                return
            # `{` / `}`(以及别的杂字符)落到下面统一处理
        elif self._state == _EXPECT_COLON:
            if ch == ":":
                self._state = _EXPECT_VALUE
            return
        else:                                    # _EXPECT_VALUE
            if ch == '"':
                self._begin_string(is_key=False)
                return
            if ch.isspace():
                return
            if self._cur_key == "answer":
                # ⚠️ **`answer` 的值必须是字符串。** 不是 ⇒ 协议不合 ⇒ 违规(fail-open)。
                # 放行的话调用方拿到的是 `useful=True` + `answer=""` + **无 violation**
                # ⇒ **什么都不发** —— 正是 Critical 那条里被认定「比任何一种兜底都糟」
                # 的形态。**「模型真没给字符串答案」与「模型给了、我们解错了」在调用方
                # 眼里长得一模一样**,所以两者都必须走同一条可预测的路(裁定 ④ 同理)。
                # 文案写清是**类型**不是**键**。
                self._violation = f"answer_type={_value_kind(ch)}"
                self._out.append(Event("violation", self._violation))
                return
            if ch not in "{}":
                self._scalar_buf = ch
                self._state = _IN_SCALAR
                return
            # 非 answer 的键:值整个是个对象/数组,落到下面按结构字符处理

        # ---------- 花括号:**结构字符,与位置无关;只有深度 0 的 `}` 才收尾** ----------
        # ⚠️ 这里是本轮修掉的那个 Critical 的所在。原先标量后遇到的 `}` 会被
        # `_EXPECT_KEY` 当成**根对象的收尾括号** ⇒ 嵌套对象的值(`"f0": {}`)
        # **静默吞掉整段回答**:`useful` 已解出、`done=True`、`answer=""`,
        # 于是调用方按 §7 的契约**什么都不发、也不走兜底** —— 一条**静默空回复**,
        # 比任何一种兜底都糟,而且调用方测不出来。
        if ch == "{":
            self._depth += 1
            self._state = _EXPECT_KEY            # 进了对象:开始等键名
            return
        if ch == "}":
            self._depth -= 1
            if self._depth > 0:
                self._state = _EXPECT_KEY        # 嵌套对象收尾:回外层继续等键名
                return
            # 深度 0:这才是根对象的收尾。`useful` 没解出也照样收尾 ——
            # 那种流由调用方按「流结束仍没 useful」处理(§5.6),解码器不替它下结论。
            # 「`done` 至多发一次」的守卫**不在这里**,在 `_IN_SCALAR` 那条
            # 「收过尾就别再让分隔符落下来」的早返回上 —— 守卫只该有一处。
            #
            # ⚠️ 「`useful=true` 却一个字没答」这颗雷**只在这一个时刻**判:
            # **判据是 `done` 由假变真(对象闭合),不是流的任意中途。**
            # 截断在答案中途(`{"useful": true, "answer": "半截`)是 **fail-open 的应有
            # 之义** —— 用户看到已生成的那半截,**不许**算违规。
            # **区分就在 `done`:对象闭合 vs 流被截断。**
            if self._useful is True and not self._answer and self._violation is None:
                # prompt 的语义是**双支的**(spec §5.2 第 3 条):证据不足 ⇒ `useful=false`
                # **且** `answer` 为空;反过来足够 ⇒ `useful=true` **且正常作答**。
                # 「说答得了、实际一个字没给」**自相矛盾**,而且正是本章入口②要抓的
                # 那件事(模型嘴上说答得了、实际一个字没给)。
                # 放行的话用户看到的是**零字节**:没有回答、没有兜底、没有 trace ——
                # 与「`answer` 类型不对」那条是**同一个形态**,只是触发条件从
                # 「值的类型不对」换成「值是空串」。
                self._violation = "empty_answer"
                self._out.append(Event("violation", self._violation))
            self._done = True          # 对象确实闭合了 —— 这条与中途违规(不改 done)不同
            self._out.append(Event("done"))
            return

    # ---- 字符串 -----------------------------------------------------------
    def _begin_string(self, *, is_key: bool) -> None:
        self._in_string = True
        self._is_key = is_key
        self._escape = False
        self._unicode_hex = None
        self._string_buf = ""

    def _consume_in_string(self, ch: str) -> None:
        # ⚠️ 这一支必须**先于** `_escape` 判:`\` 与 `u` 是**两个**字符,
        # `u` 一到 `_escape` 就被清掉了,后续的十六进制位不能再靠 `_escape` 认。
        # 把它们写成「靠 `_unicode_hex` 是不是 None 认」,转义才真正跨得了 chunk。
        if self._unicode_hex is not None:
            if ch in _HEX_DIGITS:
                self._unicode_hex += ch
                # **攒够 4 位才解码**(`\u4e` + `2d` 是必现的,不是边角)
                if len(self._unicode_hex) == 4:
                    self._emit_text(chr(int(self._unicode_hex, 16)))
                    self._unicode_hex = None
                return
            self._unicode_hex = None   # 畸形 \u:放弃收集,ch 按普通字符走
        if self._escape:
            self._escape = False
            if ch == "u":
                self._unicode_hex = ""
                return
            self._emit_text(_SIMPLE_ESCAPES.get(ch, ch))
            return
        if ch == "\\":
            self._escape = True
            return
        if ch == '"':
            self._in_string = False
            self._finish_string()
            return
        self._emit_text(ch)

    def _emit_text(self, ch: str) -> None:
        """字符串里解出的一个字符(可能是 ``\\uXXXX`` 解出来的)。

        ⚠️ **非 BMP 字符(`😀` 这类)是按 UTF-16 代理对发的**(`\\ud83d\\ude00`),
        而 `ensure_ascii=True` 的网关**正是这么发中文以外的一切非 ASCII**。
        逐个 `chr(int(hex,16))` 解出来是**两个孤立代理**,于是::

            decoder.answer.encode("utf-8")  →  UnicodeEncodeError: surrogates not allowed

        T10 要把 `answer_delta.value` 放进 **UTF-8 的 SSE `token` 帧**、并进 `log_turn`
        的 JSON ⇒ 那会变成**流中途一条 `error` 帧**,而不是一句回答。所以这里必须拼回去。
        """
        high = self._pending_high
        if high is not None:
            self._pending_high = None
            if "\udc00" <= ch <= "\udfff":
                # 拼成一个码点:非 BMP 平面 = 0x10000 + 20 位
                ch = chr(0x10000 + (ord(high) - 0xD800) * 0x400 + (ord(ch) - 0xDC00))
            else:
                # **孤立高代理**(后面不是低代理)⇒ 用 U+FFFD 顶掉。
                # 刻意与 `json.loads` 不同:后者原样给回一个孤立代理,而那正是上面
                # 那条 UnicodeEncodeError 的形态。**可预测 + 一定编得出来** > 忠实。
                self._flush_text("�")
        if "\ud800" <= ch <= "\udbff":
            self._pending_high = ch        # 先扣住,等可能跟着的后半个
            return
        if "\udc00" <= ch <= "\udfff":
            self._flush_text("�")     # **孤立低代理**:同上
            return
        self._flush_text(ch)

    def _flush_text(self, ch: str) -> None:
        """把一个**已经合法可编码**的字符送去它该去的地方(键名 / 普通值 / answer)。"""
        if not self._is_key and self._cur_key == "answer":
            self._answer += ch
            self._out.append(Event("answer_delta", ch))
            return
        self._string_buf += ch

    def _finish_string(self) -> None:
        # 字符串在**高代理之后**就收了尾(如 `"...\ud83d"`)⇒ 那个代理永远等不到
        # 后半个,收尾时按孤立代理顶掉。**不能留到下一个字符串**去配对。
        if self._pending_high is not None:
            self._pending_high = None
            self._flush_text("�")
        if self._is_key:
            key, self._string_buf = self._string_buf, ""
            self._is_key = False
            self._cur_key = key
            if not self._saw_key:
                self._saw_key = True
                if key != "useful":
                    # §5.6 全章**唯一**一处「我们确信这是作答轮、且还没吐过任何
                    # 东西」的时刻 —— 调用方据此重试一次。此处必须零 emit。
                    self._violation = f"first_key={key}"
                    self._out.append(Event("violation", self._violation))
                    return
            self._state = _EXPECT_COLON
            return
        # 字符串值读完。`answer` 的每个字符已经在 _emit_text 里吐过了。
        if self._cur_key == "useful":
            # `"useful": "true"` —— 字符串**不是**布尔,与 `1` / `nul` 同一类
            # 「协议不合」,走同一条 fail-open 的路(理由见 `_finish_useful`)。
            text, self._string_buf = self._string_buf, ""
            self._violation = f"useful_value={text!r}"
            self._out.append(Event("violation", self._violation))
            return
        self._state = _EXPECT_KEY

    # ---- useful 的值 ------------------------------------------------------
    def _finish_useful(self, raw: str) -> None:
        """⚠️ **严格只认字面 `true` / `false`。** 别的值一律 `violation`,**不是** `False`。

        spec §5.6 对「协议不合」的处置是 **fail-open**,与闸的 fail-closed 方向相反,
        理由原文:闸拦下的是「**还没生成**的回答」,代价是再看一次兜底话术;这里面对的
        是「**已经生成完、只是包装不合协议**」的回答,按 `useful=false` 处理等于
        **把一段可能完全正确的回答扔掉**并落一条 spec 点名过的**假池记录**。

        `1` / `"true"` / `nul` / `TRUE` 都属这一类 ⇒ 设 `violation`(**说明是「值」不是
        「键」**)、`useful` 保持 `None`、冻结后续一切输出。**调用方**按它既有的
        「`useful is None` ⇒ 降级成纯文本」那条路走 —— 解码器**不造第三种结局**。

        「在解出 `useful` 之前一个 delta 都不许出去」这条不变量,不因为值畸形而破。
        """
        if raw == "true":
            self._useful = True
        elif raw == "false":
            self._useful = False
        else:
            self._violation = f"useful_value={raw}"
            self._out.append(Event("violation", self._violation))
            return
        self._out.append(Event("useful", self._useful))
        if self._useful is False:
            # **不变量 2**:解出 false 的那一刻停,**后续一个 delta 都不吐**。
            # 边界 11 的违规流(useful=false 但 answer 非空)完全靠这一行守住。
            self._done = True
            self._out.append(Event("done"))

    # ---- 非字符串值 -------------------------------------------------------
    def _finish_scalar(self) -> None:
        raw, self._scalar_buf = self._scalar_buf.strip(), ""
        if self._cur_key == "useful":
            self._finish_useful(raw)
            return
        if not raw:
            return
        if self._cur_key == "confidence":
            try:
                self._confidence = float(raw)
            except ValueError:
                self._confidence = None
            self._out.append(Event("confidence", self._confidence))
