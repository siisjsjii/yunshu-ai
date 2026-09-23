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
                # 不 return:让 `,` / `}` 落到下面被同一轮处理
            else:
                self._scalar_buf += ch
                return

        # ---------- 字符串外 ----------
        if self._state == _EXPECT_KEY:
            if ch == '"':
                self._begin_string(is_key=True)
            elif ch == "}":
                # 对象收尾。`useful` 没解出也照样收尾 —— 那种流由调用方
                # 按「流结束仍没 useful」处理(§5.6),解码器不替它下结论。
                # ⚠️ `done` **至多发一次**:`{"useful": false}` 这种标量直接在 `}`
                # 上结束的流,`_finish_scalar` 已经收过尾了,别再收一次。
                if not self._done:
                    self._done = True
                    self._out.append(Event("done"))
            # `{` / `,` / 空白 / 其他杂字符:忽略
            return

        if self._state == _EXPECT_COLON:
            if ch == ":":
                self._state = _EXPECT_VALUE
            return

        # _EXPECT_VALUE
        if ch == '"':
            self._begin_string(is_key=False)
        elif not ch.isspace():
            self._scalar_buf = ch
            self._state = _IN_SCALAR

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
        """字符串里解出的一个字符:键名与普通值只收着,``answer`` 的值边收边吐。"""
        if not self._is_key and self._cur_key == "answer":
            self._answer += ch
            self._out.append(Event("answer_delta", ch))
            return
        self._string_buf += ch

    def _finish_string(self) -> None:
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
        self._state = _EXPECT_KEY

    # ---- 非字符串值 -------------------------------------------------------
    def _finish_scalar(self) -> None:
        raw, self._scalar_buf = self._scalar_buf.strip(), ""
        if not raw:
            return
        if self._cur_key == "useful":
            # 非 `true` 一律当 False 处理:解析不出「证据够用」时,宁可当不够用。
            self._useful = raw == "true"
            self._out.append(Event("useful", self._useful))
            if self._useful is False:
                # **不变量 2**:解出 false 的那一刻停,**后续一个 delta 都不吐**。
                # 边界 11 的违规流(useful=false 但 answer 非空)完全靠这一行守住。
                self._done = True
                self._out.append(Event("done"))
            return
        if self._cur_key == "confidence":
            try:
                self._confidence = float(raw)
            except ValueError:
                self._confidence = None
            self._out.append(Event("confidence", self._confidence))
