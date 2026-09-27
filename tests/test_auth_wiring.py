"""接线:**每一个**端点都挂了守卫吗。

## 为什么非有这一条不可

`tests/conftest.py` 那个 autouse 的「默认已登录」装置让**整套测试**都不带 token
也能打端点 —— 那是为了让 79 处既有调用不用改。但它的副作用是:
**一个忘了挂 `require_user` 的 router 在整套测试里全绿**
(本仓编目过的形态 ⑦「替身替被测对象完成了语义」)。

⇒ 这条**刻意不经过那个装置**:它不请求任何 fixture,直接看
`app.routes` 里每条路由身上挂了什么。

## 白名单

只有 `POST /api/auth/login` —— 它是**拿 token 的地方**,自己不能要 token。
"""

from fastapi.routing import APIRoute

from app.auth import require_admin, require_user
from app.main import app

#: 唯一允许不带守卫的端点。**加一条都要在这里写明理由。**
PUBLIC = {("POST", "/api/auth/login")}

#: 守卫的词汇表(本仓只有这两个,`app/auth.py` 是**唯一**的鉴权边界)。
#: 两条用例都读它 —— 不是死代码。
GUARDS = {"require_user", "require_admin"}

# ⚠️ 这里**不要**再留一个「只要求 require_user」的集合:plan 初稿写过 `USER_ONLY`,
# 而它是**死代码**(下面那条用例自己就把 `me` 点名了)。定义一个没人读的常量
# 正是复审要拦的那类东西。


def _guard_names(route: APIRoute) -> set[str]:
    names = set()
    for dep in route.dependencies:                # APIRouter(dependencies=[...])
        call = getattr(dep, "dependency", None)
        if call is not None:
            names.add(call.__name__)
    # 端点签名里那一个也要算(有些端点只把它写在参数里)
    for sub in getattr(route.dependant, "dependencies", []):
        names.add(sub.call.__name__)
    return names


def _iter_api_routes():
    """遍历**所有**真实端点(**递归** —— 理由见下面那条护栏的 docstring)。

    ⚠️ **fastapi 0.141.1 实测**(2026-09-27):`include_router` 往 `app.routes` 里放的是
    一个 `_IncludedRouter` **包装对象**,**不是** `APIRoute`。所以
    `for r in app.routes: if isinstance(r, APIRoute)` **一条都遍历不到**。
    真正的路由挂在包装的 `.original_router.routes` 上(再往下一层还是包装就继续递归)。
    """
    def walk(routes):
        for r in routes:
            if isinstance(r, APIRoute):
                yield r
            else:
                inner = getattr(r, "original_router", None)
                if inner is not None:
                    yield from walk(inner.routes)

    for r in walk(app.routes):
        if r.path.startswith("/api/"):
            yield r


def test_the_scan_actually_finds_routes():
    """⚠️ **上面那条扫描器的护栏 —— 没有它,这一整个文件是一条空绿。**

    `assert not unguarded` 在**空列表**上恒真:一个「一条都没遍历到」的扫描器
    (比如 fastapi 换了个包装类型、或者写成了非递归)会**安静地通过**,
    而它一个端点都没检查过 —— 这正是本文件要防的那种假绿,只不过换到了它自己身上。

    27 = spec §6.4 的操作数(25 既有 + login + me)。**只多不少** ——
    将来加端点时这个数字不会假红(留了余量),但「遍历塌成 0」一定会。
    """
    found = [(sorted(r.methods)[0], r.path) for r in _iter_api_routes()]
    assert len(found) >= 27, (
        f"只扫到 {len(found)} 条端点 —— 扫描器塌了(不是端点少了):{found}"
    )


def test_every_api_route_is_guarded_or_whitelisted():
    unguarded = []
    for r in _iter_api_routes():
        key = (sorted(r.methods)[0], r.path)
        if key in PUBLIC:
            continue
        if not (_guard_names(r) & {"require_user", "require_admin"}):
            unguarded.append(key)
    assert not unguarded, (
        f"这些端点**没有**挂守卫:{unguarded}\n"
        f"(忘了挂的 router 在整套测试里是绿的 —— 那条 autouse 装置把它们盖住了)"
    )


def test_workbench_routes_require_admin():
    """工作台的四个 router 必须是 `require_admin` —— 挂成 `require_user` 的话
    任何登录用户都能核准知识,而**所有测试照样绿**(默认装置给的是 admin)。"""
    workbench = ("/api/kb/", "/api/review/", "/api/topics/", "/api/traces/")
    wrong = []
    for r in _iter_api_routes():
        if not r.path.startswith(workbench):
            continue
        if "require_admin" not in _guard_names(r):
            wrong.append((sorted(r.methods)[0], r.path, _guard_names(r)))
    assert not wrong, f"这些工作台端点不是 require_admin:{wrong}"


def test_login_is_public_and_me_is_user_only():
    by_key = {(sorted(r.methods)[0], r.path): r for r in _iter_api_routes()}
    # ⚠️ **这里不能写成 `_guard_names(...) == set()`**(计划初稿就是那么写的)
    # —— 那条断言**恒假**,而红的原因与「挂没挂守卫」毫无关系:登录端点**合法地**
    # 依赖 `get_session`(要查 `users` 表)与 `get_settings`(要读 JWT 配置),
    # 而 `_guard_names` 收的是**签名里全部**依赖的名字。
    # (实测:未改任何实现时它给出的差集正是 `{'get_session', 'get_settings'}`。)
    # 要断的性质是「**没有**守卫」,所以与守卫词汇表取交集。
    assert not (_guard_names(by_key[("POST", "/api/auth/login")]) & GUARDS), \
        "登录端点不能要 token(否则永远登不上)"
    # `me` **必须是** `require_user`:改成 `require_admin` 的话普通用户验不了 token,
    # 而 T3 那 7 条用例**一条都不会红**(它们只断状态码,不看依赖是谁)。
    assert "require_user" in _guard_names(by_key[("GET", "/api/auth/me")])
    assert "require_admin" not in _guard_names(by_key[("GET", "/api/auth/me")])
