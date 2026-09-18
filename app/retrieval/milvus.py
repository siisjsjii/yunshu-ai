"""Milvus 向量库封装(离线写入与在线检索共用)。

ch04 起集合多存两列:**text**(BM25 全文检索的原文,`category+questions+answer`
拼接)与 **category**(元数据过滤)。原文仍以 MySQL 为权威源,Milvus 的 text 只
服务 BM25 分词,检索命中后仍回 MySQL 取原文。

懒连接两段式:**构造不 import pymilvus、不建连接**,首次调用才连。

这里**不做**错误翻译:Milvus 抛什么就抛什么,由 retrieval/search.py 统一翻成
`ToolInfrastructureError`。
"""

from functools import lru_cache

VECTOR_DIM = 1024
_ID_FIELD = "id"
_TEXT_FIELD = "text"
_CATEGORY_FIELD = "category"
_VECTOR_FIELD = "vector"
_BM25_FIELD = "text_bm25"
_PK_MAX_LENGTH = 64
_TEXT_MAX_LENGTH = 4096
_CATEGORY_MAX_LENGTH = 255


class MilvusVectorStore:
    """MilvusClient 的薄封装。`client` 是测试注入缝。"""

    def __init__(self, uri: str, collection_name: str, dim: int = VECTOR_DIM, client=None):
        self._uri = uri
        self._collection = collection_name
        self._dim = dim
        self._client = client

    def _ensure_client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self._uri)
        return self._client

    def ensure_collection(self) -> None:
        """幂等建集合:存在即跳过,不存在则建 schema(text/category/vector + BM25 函数)。

        建 schema 时不会自动 load(T0 实测),故建完手动 load。
        """
        client = self._ensure_client()
        if client.has_collection(self._collection):
            return
        from pymilvus import (
            CollectionSchema,
            DataType,
            FieldSchema,
            Function,
            FunctionType,
        )

        schema = CollectionSchema(
            fields=[
                FieldSchema(name=_ID_FIELD, dtype=DataType.VARCHAR,
                            is_primary=True, max_length=_PK_MAX_LENGTH),
                FieldSchema(name=_TEXT_FIELD, dtype=DataType.VARCHAR,
                            max_length=_TEXT_MAX_LENGTH, enable_analyzer=True,
                            analyzer_params={"tokenizer": "jieba"}),
                FieldSchema(name=_CATEGORY_FIELD, dtype=DataType.VARCHAR,
                            max_length=_CATEGORY_MAX_LENGTH),
                FieldSchema(name=_VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR,
                            dim=self._dim),
                # BM25 输出字段必须显式声明为 SPARSE_FLOAT_VECTOR(T0 实测),
                # 否则报「Function output field not found in collection schema」。
                FieldSchema(name=_BM25_FIELD, dtype=DataType.SPARSE_FLOAT_VECTOR),
            ],
            enable_dynamic_field=True,
        )
        schema.add_function(Function(
            name=_BM25_FIELD, function_type=FunctionType.BM25,
            input_field_names=[_TEXT_FIELD], output_field_names=[_BM25_FIELD],
        ))
        client.create_collection(self._collection, schema=schema)
        # dense 索引(IP)+ BM25 稀疏索引(metric 必须是 BM25,T0 实测)。
        client.create_index(
            self._collection,
            index_params=client.prepare_index_params(
                field_name=_VECTOR_FIELD, index_type="AUTOINDEX", metric_type="IP"),
        )
        client.create_index(
            self._collection,
            index_params=client.prepare_index_params(
                field_name=_BM25_FIELD, index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25"),
        )
        client.load_collection(self._collection)

    def drop_collection(self) -> None:
        client = self._ensure_client()
        if client.has_collection(self._collection):
            client.drop_collection(self._collection)

    def upsert(self, ids: list, texts: list, categories: list, vectors: list) -> None:
        """按 pk 覆盖写入(text/category/vector)。写完必须 flush 才立查(T0 实测)。"""
        if not ids:
            return
        rows = [
            {"id": str(i), "text": t, "category": c, "vector": list(v)}
            for i, t, c, v in zip(ids, texts, categories, vectors)
        ]
        self._ensure_client().upsert(collection_name=self._collection, data=rows)
        self.flush()

    def flush(self) -> None:
        self._ensure_client().flush(self._collection)

    @staticmethod
    def _category_expr(category: str | None) -> str | None:
        return f'{_CATEGORY_FIELD} == "{category}"' if category else None

    def search(self, vector: list, top_k: int, category: str | None = None) -> list[tuple[str, float]]:
        """dense 单路(纯向量策略)。"""
        res = self._ensure_client().search(
            collection_name=self._collection,
            data=[list(vector)],
            filter=self._category_expr(category) or "",
            limit=top_k,
            output_fields=[_ID_FIELD],
            search_params={"metric_type": "IP"},
            # 集合里现有 dense + sparse 两个向量字段,必须显式指 anns_field(T0 实测)。
            anns_field=_VECTOR_FIELD,
        )
        return self._flatten(res)

    @staticmethod
    def _flatten(res) -> list[tuple[str, float]]:
        if not res:
            return []
        return [(str(hit[_ID_FIELD]), float(hit["distance"])) for hit in res[0]]

    def bm25_search(self, text: str, top_k: int, category: str | None = None) -> list[tuple[str, float]]:
        """BM25 单路(纯关键词策略)。BM25 腿 data 是原始文本(T0 实测)。"""
        from pymilvus import AnnSearchRequest, RRFRanker

        return self._hybrid(
            [AnnSearchRequest(
                data=[text], anns_field=_BM25_FIELD, param={}, limit=top_k,
                expr=self._category_expr(category))],
            top_k,
        )

    def hybrid_search(self, vector: list, text: str, top_k: int,
                      category: str | None = None) -> list[tuple[str, float]]:
        """dense + BM25 双路,RRF(k=60) 融合。"""
        from pymilvus import AnnSearchRequest, RRFRanker

        return self._hybrid(
            [
                AnnSearchRequest(
                    data=[list(vector)], anns_field=_VECTOR_FIELD,
                    param={"metric_type": "IP"}, limit=top_k,
                    expr=self._category_expr(category)),
                AnnSearchRequest(
                    data=[text], anns_field=_BM25_FIELD, param={}, limit=top_k,
                    expr=self._category_expr(category)),
            ],
            top_k,
        )

    def _hybrid(self, reqs, top_k: int) -> list[tuple[str, float]]:
        from pymilvus import RRFRanker

        res = self._ensure_client().hybrid_search(
            collection_name=self._collection, reqs=reqs,
            ranker=RRFRanker(k=60), limit=top_k, output_fields=[_ID_FIELD],
        )
        return self._flatten(res)

    def count(self) -> int:
        """集合真实行数(用 count(*),stats 的 row_count 未扣 delete 不可信)。"""
        res = self._ensure_client().query(
            collection_name=self._collection, filter="", output_fields=["count(*)"]
        )
        return int(res[0]["count(*)"]) if res else 0


@lru_cache(maxsize=1)
def get_vector_store(uri: str, collection_name: str) -> MilvusVectorStore:
    """进程内单例(同 get_embedder 模式)。"""
    return MilvusVectorStore(uri, collection_name)
