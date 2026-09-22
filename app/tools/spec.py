"""`ToolSpec` —— 一条工具登记项 + 全章**唯一**的 JSON Schema 校验器。

内置与 MCP 来的工具**一视同仁**:名 / 用途描述 / JSON Schema 参数定义三样齐备。
"""

from dataclasses import dataclass

import jsonschema

READ = "read"
WRITE = "write"


@dataclass(frozen=True)
class ToolSpec:
    """一条工具登记项。

    `input_schema` 是**原始 JSON Schema**,不是 pydantic 模型:
    - 内置那份从 `tool.args_schema.model_json_schema()` 派生;
    - MCP 那份从 Server 的 `inputSchema` **原样取**(spec §3.3)。

    后者刻意**不经 adapters 的 pydantic 转换** —— 转换会削平
    `minimum` / `maxLength` / `enum` 这类约束,于是「统一按 JSON Schema 校验」
    退化成「只查必填和类型」,闸看起来在工作、实际漏掉一半。

    `tool` 是绑给模型的那份 `BaseTool`;执行时才用它。
    """

    name: str
    description: str
    input_schema: dict
    kind: str          # READ | WRITE
    source: str        # "builtin" | "mcp:logistics" | "mcp:aftersales"
    tool: object       # BaseTool(此处不 import LangChain,见 app/tools/registry.py)


def _readable(error: jsonschema.ValidationError) -> str:
    """一条 `ValidationError` → **给模型看**的一句话。

    ⚠️ 文案是给**模型**看的,不是给人看的调试信息:它会被当作工具结果回灌,
    模型据此**追问用户**或**重新组织调用**。所以:
    - 点名到字段(`order_id`),不要 `$['order_id']` 这种取值路径;
    - 说清楚**为什么**(必填未提供 / 类型不对 / 取值越界);
    - **不带** pydantic 或 jsonschema 的堆栈式原文 ——
      `test_problems_are_human_readable_not_pydantic_dumps` 钉着这条。
    """
    path = ".".join(str(p) for p in error.absolute_path) or "(根)"
    if error.validator == "required":
        missing = error.message.split("'")[1] if "'" in error.message else error.message
        return f"缺少必填字段 `{missing}`"
    if error.validator == "type":
        return f"`{path}` 类型不对:{error.message}"
    if error.validator == "enum":
        allowed = "、".join(str(v) for v in error.validator_value)
        return f"`{path}` 取值必须是 {allowed} 之一"
    if error.validator in ("minimum", "maximum", "minLength", "maxLength"):
        return f"`{path}` 超出取值范围:{error.message}"
    return f"`{path}` 不合法:{error.message}"


def validate_args(spec: ToolSpec, args: dict) -> list[str]:
    """按 `spec.input_schema` 校验 `args`。返回**给模型看**的问题列表,空 = 通过。

    **空 schema 是合法的**(等价于「什么参数都行」),不是「校验失败」——
    `jsonschema.validate` 对 `{}` 一律放行,本函数不额外加限制。

    **刻意不禁止多余字段**:JSON Schema 的默认语义就是「额外的键不校验」。
    加 `additionalProperties: false` 的话,模型多传一个它自己编的字段就会被
    拦下,而那是个**无害**行为,拦它只会白白浪费一轮对话。

    ⚠️ **不抛异常**:校验失败是**可恢复**的,由执行器包成一条工具结果回灌给模型。
    抛异常会让它走 `except Exception` → `ToolInfrastructureError` → 502,
    把「模型参数写错了」伪装成「服务挂了」。
    """
    if not spec.input_schema:
        return []
    validator_cls = jsonschema.validators.validator_for(spec.input_schema)
    # schema 本身是坏的(我们的 bug,不是模型的)就**响亮地抛**,由执行器归到
    # 基础设施那一支。静默返回 [] 等于把闸关掉而没人知道。
    validator_cls.check_schema(spec.input_schema)
    validator = validator_cls(spec.input_schema)
    return [_readable(e) for e in validator.iter_errors(args)]
