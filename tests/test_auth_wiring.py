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

from app.main import app

#: 唯一允许不带守卫的端点。**加一条都要在这里写明理由。**
#: ⚠️ 它是 `EXPECTED_GUARDS` 里值为 `None` 那一条的**同一件事的第二处写法**。
#: 没有并进来是**刻意**的:漂移不会静默 —— 这里多一条会在
#: `test_the_guard_matrix_matches_spec_64_exactly` 上红(端点集合不等),少一条会在
#: `test_every_api_route_is_guarded_or_whitelisted` 上红。
#: (它另有一个读者:`tests/test_api_auth.py` 的运行时 401 扫描按它跳过公开端点。)
PUBLIC = {("POST", "/api/auth/login")}

#: 守卫的词汇表(本仓只有这两个,`app/auth.py` 是**唯一**的鉴权边界)。
GUARDS = {"require_user", "require_admin"}

#: spec §6.4 的权限矩阵 —— **逐行**照抄(27 行,一行都不许抽样)。
#:
#: ⚠️ **为什么非得是这张表,而不是「至少挂了某一个守卫」那种粗断言**:
#: 复审给过反例 —— 把 `app/api/extract.py` 的守卫换成 `require_admin`,
#: 「有没有守卫」那类断言**全绿**(`require_admin` 在守卫词表里、
#: `test_workbench_routes_require_admin` 按前缀跳过它、运行时无 token 照样 401,
#: 而 `conftest.py` 把两个守卫**都**替成同一个 admin 假用户)
#: ⇒ `/api/extract` 悄悄变成「只有管理员能用」,而**没有一条测试红**。
#: 判据必须细到「**挂的是哪一个**」—— 那正是 spec §6.4 那句话的内容。
#: (同一个形状的第二个例子:`me` 挂成 `require_admin`,见
#: `test_login_is_public_and_me_is_user_only`。)
#:
#: `None` = 公开(只有 login 一处)。`{conversation_id}` / `{name}` / `{job_id}` /
#: `{review_id}` 这几个占位符名是**从 `r.path` 读出来的**(不是照文档手抄的)。
EXPECTED_GUARDS = {
    ("POST", "/api/auth/login"): None,                 # 公开:拿 token 的地方
    ("GET", "/api/auth/me"): "require_user",
    ("POST", "/api/chat/stream"): "require_user",
    ("POST", "/api/ticket"): "require_user",
    ("GET", "/api/conversations"): "require_user",
    ("GET", "/api/conversations/{conversation_id}/messages"): "require_user",
    ("POST", "/api/extract"): "require_user",
    ("POST", "/api/feedback"): "require_user",
    ("POST", "/api/refund"): "require_user",
    # ---- 工作台 18 个:require_admin ----
    ("GET", "/api/kb/stats"): "require_admin",
    ("GET", "/api/kb/documents"): "require_admin",
    ("GET", "/api/kb/documents/{name}"): "require_admin",
    ("POST", "/api/kb/documents"): "require_admin",
    ("GET", "/api/kb/search"): "require_admin",
    ("GET", "/api/kb/eval"): "require_admin",
    ("POST", "/api/kb/jobs/vectorize"): "require_admin",
    ("POST", "/api/kb/jobs/mine"): "require_admin",
    ("POST", "/api/kb/jobs/flywheel"): "require_admin",
    ("POST", "/api/kb/jobs/eval"): "require_admin",
    ("GET", "/api/kb/jobs"): "require_admin",
    ("GET", "/api/kb/jobs/{job_id}"): "require_admin",
    ("GET", "/api/review/queue"): "require_admin",
    ("GET", "/api/review/{review_id}"): "require_admin",
    ("POST", "/api/review/{review_id}/approve"): "require_admin",
    ("POST", "/api/review/{review_id}/reject"): "require_admin",
    ("GET", "/api/topics/distribution"): "require_admin",
    ("GET", "/api/traces/recent"): "require_admin",
}

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
        # 只收 `/api/*`:spec §6.4 那张矩阵说的就是这个前缀下的操作,
        # 而这个 App 另外还有「不是 API」的几页 —— `mount("/")` 的静态 catch-all
        # 与 FastAPI 自带的 `/docs` / `/redoc` / `/openapi.json`
        # (⚠️ 实测:今天这一行**一条都没挡掉** —— 那几页是
        #  `starlette.routing.Route` 与 `Mount`,**不是** `APIRoute`,
        #  上面的 `walk` 本来就遍历不到它们。留着是因为扫描器哪天改成
        #  「凡是带 path 的都收」时,得有个地方把非 API 的挡在外面。)
        # ⚠️ 下面那条非空性护栏(`>= 27`)数的是**过滤之后**这个列表 ——
        # 别把「扫描器没塌」读成「App 里每一条路由都被检查过了」。
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
        if not (_guard_names(r) & GUARDS):
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


def test_the_guard_matrix_matches_spec_64_exactly():
    """**逐行**钉住 spec §6.4 —— 27 行,**类型也要对**。

    这条与上面那几条的分工:上面问的是「**有没有**挂守卫」(粗),
    这条问的是「挂的**是哪一个**」(细)。两者都要 —— 粗的那条在「整行被删掉」时
    给的信息更直白,细的那条才管得住「挂错了类型」那一类(见 `EXPECTED_GUARDS`
    的说明:换成 `require_admin` 时上面几条**全绿**)。

    **两端都断**:端点集合相等(**多一条、少一条都红**)+ 每一行的守卫集合相等。
    只断「期望的每一行都对」的话,一个**新加进来却没写进矩阵**的端点会漏过去;
    只断集合的话,类型挂错了看不出来。

    `me` 那一行是**有意重复**的:它在上面的
    `test_login_is_public_and_me_is_user_only` 里被点过一次名 —— 那条留下来是因为
    它把「为什么 `me` 必须是 `require_user`」写在了断言旁边(前端启动时验 token,
    普通用户也要能过)。这一条是**整张表**的机器可读版本。两条不冲突。
    """
    actual = {
        (sorted(r.methods)[0], r.path): (_guard_names(r) & GUARDS)
        for r in _iter_api_routes()
    }
    assert set(actual) == set(EXPECTED_GUARDS), (
        "端点集合与 spec §6.4 不一致:"
        f"多出 {sorted(set(actual) - set(EXPECTED_GUARDS))},"
        f"缺少 {sorted(set(EXPECTED_GUARDS) - set(actual))}"
    )
    wrong = {
        k: (sorted(v), [EXPECTED_GUARDS[k]] if EXPECTED_GUARDS[k] else [])
        for k, v in actual.items()
        if v != ({EXPECTED_GUARDS[k]} if EXPECTED_GUARDS[k] else set())
    }
    assert not wrong, f"守卫挂错(实际 vs 期望):{wrong}"


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
