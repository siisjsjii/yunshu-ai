"""鉴权内核:`app/auth.py` 的纯函数部分 + 两个 FastAPI 依赖。

**不打 `pytest.mark.db`** —— 这一份一个库都不碰(用户名与角色就在 token 里,
`require_user` **不查库**)。查库的只有登录端点(`tests/test_api_auth.py`)。

## 为什么每条都要断「它到底拒了没有」

这一份的性质是**拒绝**:一个「永远返回 True」的 `verify_password`、或一个
「任何 token 都解出 admin」的 `decode_token`,在**别的所有测试里**都不会红
(它们走 conftest 那个默认已登录的替身)。⇒ 这里每一条都必须有一个
**必须被拒**的输入,而且那条输入要真的构造得出来。
"""

import time

import jwt
import pytest

from app.auth import (
    ADMIN, USER, AuthError, AuthenticatedUser, create_token, decode_token,
    hash_password, require_admin, require_user, verify_password,
)
from app.config import Settings

_REQUIRED = dict(
    openai_base_url="https://example.invalid/v1",
    openai_api_key="sk-auth-test-KEY",
    openai_model="test-model",
    database_url="mysql+asyncmy://u:p@h:3306/db",
)

#: ⚠️ 非默认的密钥 —— 用默认值的话「密钥真的被用上了吗」区分不出来
#: (随机密钥与这个值下的行为长得一样)。
SECRET = "unit-test-secret"


def _settings(**over) -> Settings:
    return Settings(_env_file=None, jwt_secret=SECRET, **{**_REQUIRED, **over})


def test_hash_is_salted_so_two_hashes_of_the_same_password_differ():
    a, b = hash_password("123456"), hash_password("123456")
    assert a != b, "两次哈希相同 ⇒ 盐没生效(固定盐的库一次泄露全泄露)"
    assert a.startswith("scrypt$"), f"自描述串的形状变了:{a[:20]}"


def test_verify_accepts_the_right_password_and_rejects_a_wrong_one():
    stored = hash_password("123456")
    assert verify_password("123456", stored) is True
    assert verify_password("1234567", stored) is False, "错密码必须拒"
    assert verify_password("", stored) is False


def test_verify_rejects_a_malformed_stored_string_instead_of_raising():
    """坏串(手工改库 / 换过算法)必须**回 False**,不能抛。

    抛的话它会从端点里冒成 500 —— 而「这条记录的哈希读不懂」对用户是
    「密码不对」这一件事,不是服务端故障。
    """
    for bad in ("", "明文密码", "scrypt$16384$8$1$notbase64$xxx", "bcrypt$aa$bb"):
        assert verify_password("123456", bad) is False, f"{bad!r} 应当回 False"


def test_token_round_trip_carries_username_and_role():
    tok = create_token(username="cinfly", role=ADMIN, settings=_settings())
    u = decode_token(tok, _settings())
    assert u == AuthenticatedUser(username="cinfly", role=ADMIN)


def test_expired_token_is_rejected():
    """有效期靠 `exp`,由 PyJWT 自己校验 —— 这条钉住「真的设了 exp」。

    `jwt_expire_minutes` 传**负数**造一个出生即过期的 token(比 sleep 快且不骰子)。

    ⚠️ **负数必须走 `model_copy`,不能当构造参数传**:配置项上那个 `gt=0` 是
    **启动期防呆**(负有效期 ⇒ 每个 token 出生即过期,而**没有任何东西报错**),
    所以 `Settings(jwt_expire_minutes=-1)` 会先被 pydantic 拒掉,红在
    `ValidationError` 上 —— 那样这条用例**根本没验到过期**。
    `model_copy` 绕过校验,与计划里 `/api/auth/me` 那条过期用例同一形状。

    密钥取自同一份 `_settings()` ⇒ 拒绝的理由**只可能是过期**,
    不会退化成「签名不匹配」(那是 `decode_token` 的另一条分支)。
    """
    expired = _settings().model_copy(update={"jwt_expire_minutes": -1})
    tok = create_token(username="cinfly", role=ADMIN, settings=expired)
    with pytest.raises(AuthError):
        decode_token(tok, _settings())


def test_signature_tampering_is_rejected():
    """换一个密钥签的 token 必须被拒(签名真的在校验)。"""
    other = Settings(_env_file=None, jwt_secret="a-completely-different-secret",
                     **_REQUIRED)
    tok = create_token(username="cinfly", role=ADMIN, settings=other)
    with pytest.raises(AuthError):
        decode_token(tok, _settings())


def test_alg_none_token_is_rejected():
    """`alg: none` 的经典攻击:不签名的 token 必须被拒。

    这条守的是 `jwt.decode(..., algorithms=["HS256"])` 里那个**显式**列表 ——
    去掉它 PyJWT 会抛(它自己的防混淆机制),写错成 `algorithms=None` 才是真漏洞。
    """
    forged = jwt.encode({"sub": "cinfly", "role": ADMIN, "exp": int(time.time()) + 3600},
                        key="", algorithm="none")
    with pytest.raises(AuthError):
        decode_token(forged, _settings())


def test_token_without_sub_is_rejected():
    """**缺 `sub`** 的 token 不许当成匿名用户放过去。

    ⚠️ **两条 claim 必须分开测**(计划初稿把 `sub` 与 `role` **一起省掉** ——
    那样**删掉任何一条守卫它都照样绿**,本仓「假绿形态」里最经典的一种)。
    这里只**少给 `sub`**,`role` 给对的。
    """
    bare = jwt.encode({"role": ADMIN, "exp": int(time.time()) + 3600},
                      SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        decode_token(bare, _settings())


def test_token_with_an_unknown_role_is_rejected():
    """**角色不认识**的 token 也不许放过去(只少给 `role` 那一半)。"""
    bare = jwt.encode({"sub": "cinfly", "role": "root", "exp": int(time.time()) + 3600},
                      SECRET, algorithm="HS256")
    with pytest.raises(AuthError):
        decode_token(bare, _settings())


def test_empty_secret_generates_a_random_one_and_still_round_trips():
    """`.env` 没配时:同一个进程内签发/校验仍然自洽(重启后失效是**设计**)。

    ⚠️ `jwt_secret=""` **必须显式传**:`_env_file=None` 只关掉 `.env` 这个**文件**,
    pydantic-settings **照读 `os.environ`** —— 跑测试的 shell 里一旦 export 过
    `JWT_SECRET`,这条就会拿到那个值,而它断的正是「空配置」那条路。
    """
    s = Settings(_env_file=None, jwt_secret="", **_REQUIRED)
    assert s.jwt_secret == ""
    u = decode_token(create_token(username="demo-user", role=ADMIN, settings=s), s)
    assert u.username == "demo-user"


def test_require_admin_rejects_a_plain_user():
    with pytest.raises(Exception) as ei:
        require_admin(AuthenticatedUser(username="bob", role=USER))
    assert getattr(ei.value, "status_code", None) == 403, (
        "非 admin 打工作台必须是 403(不是 401 —— 前端只对 401 弹登录)")


def test_require_user_has_no_header_and_raises_401():
    """**缺 header 必须是 401**,且带 `WWW-Authenticate`。

    ⚠️ 这条是 Task 5 那条结构性测试的**行为**对应物:前者断「依赖挂上了」,
    这条断「挂上之后行为对」。
    """
    with pytest.raises(Exception) as ei:
        require_user(None, _settings())
    assert getattr(ei.value, "status_code", None) == 401, "缺 header 必须 401"
    assert ei.value.headers.get("WWW-Authenticate") == "Bearer"


def test_require_user_rejects_a_garbage_token_with_401_not_500():
    from fastapi.security import HTTPAuthorizationCredentials
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not-a-jwt")
    with pytest.raises(Exception) as ei:
        require_user(creds, _settings())
    assert getattr(ei.value, "status_code", None) == 401
