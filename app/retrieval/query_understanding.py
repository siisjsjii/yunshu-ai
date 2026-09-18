"""Query 理解:口语模糊问法 → 标准问法 + 同义词扩展(仅检索侧)。

ch04 spec §5.4。json_mode 结构化输出;同义词只在检索侧扩展,不在入库侧拆存
多份。解析失败不中断检索 —— 退化为原始 query(检索链路比改写失败更重要)。
"""

from dataclasses import dataclass

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class QueryRewriteResult(BaseModel):
    normalized: str = Field(description="改写成标准问法,用于检索")
    synonyms: list[str] = Field(default_factory=list, description="同义变体,仅检索侧扩展")


QUERY_UNDERSTANDING_PROMPT = """你是电商客服的查询改写助手。
把用户的口语化、模糊的问法改写成更标准、更适合检索的问法,并给出 1-2 个同义变体。

输出一个 JSON 对象,含两个字段:
1. normalized:字符串。把用户原话改写成标准问法,保留关键信息(型号、品类、诉求),去掉口语冗余。
2. synonyms:字符串数组。给出 1-2 个语义相近的检索变体(换个说法或同义词),没有则给空数组。

不要输出 JSON 以外的任何内容。"""

_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QUERY_UNDERSTANDING_PROMPT), ("human", "{query}")]
)


@dataclass(frozen=True)
class QueryVariant:
    normalized: str
    synonyms: list[str]


class QueryUnderstanding:
    def __init__(self, model):
        self._model = model

    async def rewrite(self, query: str) -> QueryVariant:
        chain = self._model.with_structured_output(QueryRewriteResult, method="json_mode")
        try:
            result = await chain.ainvoke(_QUERY_PROMPT.format_messages(query=query))
        except OutputParserException:
            return QueryVariant(normalized=query, synonyms=[])
        return QueryVariant(
            normalized=result.normalized.strip() or query,
            synonyms=[s.strip() for s in (result.synonyms or []) if s.strip()],
        )
