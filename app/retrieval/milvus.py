"""Milvus 向量库封装(离线写入与在线检索共用)。

**Milvus 只当索引,不存文本**(spec §6.1):集合里只有两个字段 ——
`id`(VARCHAR,值 = `str(MySQL knowledge_chunks.id)`)和 `vector`
(FLOAT_VECTOR 1024)。命中后拿 id 回 MySQL 查原文,所以 Milvus 可以随时
drop 重建(全表回到 pending → 重跑补齐),不存在双份真相漂移。

懒连接两段式(同 embedder):**构造不 import pymilvus、不建连接**,
首次真正调用才连。单测与不碰检索的服务启动路径不该背上 pymilvus 的
导入与网络开销。

这里**不做**错误翻译:Milvus 抛什么就抛什么,由 retrieval/search.py 这个
边界统一翻成 `ToolInfrastructureError`(spec §6.7),离线脚本则直接崩
—— 三处的处置不同,翻译留在各自的那一层。
"""

from functools import lru_cache

VECTOR_DIM = 1024
_ID_FIELD = "id"
_VECTOR_FIELD = "vector"
#: VARCHAR 主键的长度上限。Milvus **强制要求** VARCHAR 字段声明 max_length,
#: 不传直接报 1101「type param(max_length) should be specified」—— 而 pymilvus
#: 的快捷建法(get_collection_stats 那套 create_collection 不带 schema)默认
#: 不给主键补这个参数,必须自己传。pk 值 = str(MySQL id),BIGINT UNSIGNED 的
#: 十进制上限是 20 位,64 留足余量。
_PK_MAX_LENGTH = 64


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
        """幂等建集合:存在即跳过,不存在则建(VARCHAR 主键 + 1024 维 + IP)。

        用 pymilvus 的快捷建法(不给 schema)会自动带 AUTOINDEX 并 load。
        """
        client = self._ensure_client()
        if client.has_collection(self._collection):
            return
        client.create_collection(
            collection_name=self._collection,
            dimension=self._dim,
            primary_field_name=_ID_FIELD,
            # pk 是 str(MySQL id):用整数 pk 会在 id 一旦超过 int64 或需要
            # 前缀化时逼着改 schema,而字符串 pk 与「Milvus 是纯索引、
            # MySQL 才是权威源」的定位一致。
            id_type="string",
            max_length=_PK_MAX_LENGTH,
            vector_field_name=_VECTOR_FIELD,
            # dense 向量已 L2 归一化(BGE-M3 输出),IP 与余弦在此等价。
            metric_type="IP",
            auto_id=False,
        )

    def upsert(self, ids: list, vectors: list) -> None:
        """按 pk 覆盖写入。同 pk 重复 upsert 是幂等的(T0 实证)。

        写完**必须 flush** 才能立查 —— 默认 Bounded 一致性下,不 flush 的
        新数据搜不到(T0 实测,spec §12);离线路径每批都写,索性每批都 flush。
        """
        if not ids:
            return
        rows = [
            {"id": str(i), "vector": list(v)} for i, v in zip(ids, vectors)
        ]
        self._ensure_client().upsert(collection_name=self._collection, data=rows)
        self.flush()

    def flush(self) -> None:
        self._ensure_client().flush(self._collection)

    def search(self, vector: list, top_k: int) -> list[tuple[str, float]]:
        """单条查询向量 → [(id, score)],按 score 降序(由 Milvus 保证)。

        返回的是 `res[0]`:pymilvus 的返回形状是「每条查询向量一个命中列表」,
        我们一次只查一条。无命中时是空列表,不是异常 —— 集合不存在才是异常,
        那条路径必须让它抛出去(search.py 翻成 502),不能在这里兜成空结果,
        否则「Milvus 挂了」会被伪装成「这条知识没收录」。
        """
        res = self._ensure_client().search(
            collection_name=self._collection,
            data=[list(vector)],
            limit=top_k,
            output_fields=[_ID_FIELD],
        )
        if not res:
            return []
        return [(str(hit[_ID_FIELD]), float(hit["distance"])) for hit in res[0]]

    def count(self) -> int:
        """集合真实行数。

        **不能用 get_collection_stats**:它的 row_count 数的是 insert 操作、
        未扣 delete(同 pk upsert 三遍报 9 而真值 3),compaction 前不可信
        —— 验收 6 的「Milvus 数 == MySQL done 数」必须走 count(*)。
        """
        res = self._ensure_client().query(
            collection_name=self._collection, filter="", output_fields=["count(*)"]
        )
        return int(res[0]["count(*)"]) if res else 0


@lru_cache(maxsize=1)
def get_vector_store(uri: str, collection_name: str) -> MilvusVectorStore:
    """进程内单例(同 get_embedder 模式)。

    单例在这里不只是省内存:MilvusClient 每次构造都是一条新的 gRPC 通道,
    每请求一条会把连接数打在 Milvus 上 —— 在线检索每个请求都要走这条路。
    """
    return MilvusVectorStore(uri, collection_name)
