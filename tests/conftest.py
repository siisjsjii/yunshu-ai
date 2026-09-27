"""全局测试装置。

## 1. anyio 后端

`pytest.ini` 没钉后端,这里显式给 asyncio(与 `pytest.mark.anyio` 配套,
不用 pytest-asyncio)。

## 2. 飞轮那三处 fire-and-forget 钩子**一律打成 no-op**(ch09 T14)

`app/agent/nodes.py`(置信度闸 / 生成自评)与 `app/api/feedback.py`(👎 落池)
在**落池之后**各调一次 `start_flywheel_job_safely(settings)` —— 它**不 await、
也不抛**,内部是「起一条守护线程跑一批飞轮」。

⇒ 不打桩的话,**任何**走到落池那一步的用例(不是只有 T14 那份文件:闸的
`tests/test_agent_gate*.py`、自评的 `tests/test_agent_protocol.py`、反馈的
`tests/test_api_feedback.py` 全在内)都会**真的起一条线程**,而那条线程会:
拿用例那份 settings 自建 engine 去连库、调 `create_extract_model`、再跑
`run_flywheel`。

**实测到的后果**(`.superpowers/t14_hook_probe.py`,不经过 pytest 因而没有这份守卫;
流水线被换成 no-op,所以那次探测本身是安全的):

  ① 一条 `flywheel-job-*` 线程真的起来;
  ② **进程级 `JobStore` 里多出一条 `flywheel` 任务** —— 而它是**单槽**的
     (`is_busy`/`start` 只看那一个 `_running_id`),于是「上一个用例的飞轮还在跑」
     会让**别的**用例的 `start_flywheel_job` 拿到 `None`(端点 → 409、T14 那两条
     用例 → 直接红),红在一个与被测代码毫不相干的地方;
  ③ 线程**从单测里发真实的 TCP 连接**(那是「非 DB 测试绝不碰网络」那条硬约束)。

**还有一条今天没被踩到、但机制上成立的**:`_fresh_factory(settings)` 用的是
**用例那份 settings 的 `database_url`**。今天是靠「池子相关的用例全都注入假 URL」
才没吃到库里的东西;哪天有人加一条用**真 `.env`** 走到落池的用例,那条线程就会
拿真 URL 连上开发库、跑真 `run_flywheel`,而 `WHERE matched_review_id IS NULL`
**正是它的待处理谓词** —— 池子里那 30 行前几章的旧数据会被**静默吃掉**
(`matched_review_id` 一写就退出待处理,`occurrences` 还会跟着涨),
**没有任何断言会红**。这份守卫让那条路**结构上**不可达。

### 为什么打在这一层、而不是打在 `app.flywheel.tasks` 上

打在 `tasks` 上会连带把 `tests/test_flywheel_task.py` 自己**要验的东西**
一起打成空话(T14 的全部价值就是那三件事:自建 engine、专用线程、dispose)。
所以这里只换掉**钩子模块命名空间里的那一个名字**(`from ... import X` 之后,
调用处是 `LOAD_GLOBAL`,换命名空间的绑定即生效),`app.flywheel.tasks` 本身
**一个字都不动**。

### 为什么是「记录器」而不是 `lambda: None`

替身要**替得出证据**:`flywheel_hooks.calls` 是那三处的调用流水,用例靠它断
「落池之后**确实**起了飞轮,且传的是**这一份** settings」(brief Step 3 的
「在落池的用例里断被调用了一次」)。换成一个纯 no-op 的话,「钩子整行被删掉」
与「钩子好好的」在测试里长得**一模一样**。

⚠️ **`calls` 记的是 settings 对象本身,不是 `True`** —— 只记「被调过」的话,
一个把**别的** settings(或常量)传进去的实现照样绿。
"""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FlywheelHookCalls:
    """落在飞轮三处钩子位置上的替身:记下**每一次**调用拿到的 settings。"""

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, settings):
        self.calls.append(settings)
        return None


@pytest.fixture
def flywheel_hooks(monkeypatch) -> FlywheelHookCalls:
    """把三处 fire-and-forget 钩子换成**记录器**(全局 autouse 的底座)。

    用例可以显式请求它来读 `calls`;不请求的用例也照样受它保护(见
    `_no_flywheel_threads`)。
    """
    from app.agent import nodes as agent_nodes
    from app.api import feedback as feedback_api

    recorder = FlywheelHookCalls()
    monkeypatch.setattr(agent_nodes, "start_flywheel_job_safely", recorder)
    monkeypatch.setattr(feedback_api, "start_flywheel_job_safely", recorder)
    return recorder


@pytest.fixture(autouse=True)
def _no_flywheel_threads(flywheel_hooks):
    """**每条用例**都自动生效:不接受 `flywheel_hooks` 的用例也拿不到真线程。"""
    return flywheel_hooks


@pytest.fixture
def _no_default_login():
    """**关掉**「默认已登录」装置的开关。

    只有认证自己的测试请求它 —— 那几条测的就是「没登录时会怎样」,
    被默认装置一盖,断言的「401」会变成 200 而**红得莫名其妙**。

    ⚠️ 这里只是**占个名字**:真正的开关在 **Task 4**(计划裁定 R9 —— 原先排在
    Task 5,后来跟着 `Depends(require_user)` 一起挪到了 T4)加的 autouse 装置里,
    它靠 `"_no_default_login" in request.fixturenames` 认这个名字。
    ⇒ **本任务不许改它的形状**(名字即契约),那个装置才加它的逻辑。
    """
    return None


@pytest.fixture(autouse=True)
def _default_login(request):
    """**每条用例**默认带一个已登录的 admin。

    ## 为什么要有它

    认证一挂,`app/api/` 下 25 个操作全部要 token ⇒ 16 个既有测试文件里
    **约 79 处**端点调用会集体变红,而它们**测的都不是认证**。
    在**一处**注入默认身份,那 79 处一个字都不用改。

    ## 它与「假绿」的关系(必须配套 `tests/test_auth_wiring.py`)

    它让「这个端点受不受保护」在测试里**恒真** —— 少挂一个守卫照样全绿。
    那件事由 `tests/test_auth_wiring.py` **不经过本装置**地钉住。
    **两条必须同时在**,少一条就是本仓编目过的形态 ⑦。

    ## 为什么是 admin 而不是 user

    默认给 `user` 的话,工作台那 18 个端点在既有测试里会**集体 403**;
    而 admin 是**超集**(能打用户面也能打工作台)—— 与 spec §6.4 的语义一致。

    ## ⚠️ 开关为什么用 `request.fixturenames` 而**不是参数**

    写成 `def _default_login(_no_default_login)` 的话,那个名字**永远**在
    `fixturenames` 里 ⇒ 默认登录**永远**被跳过 ⇒ 那 79 处调用**集体 401**。
    (实测过:临时 conftest + 两条用例,只有这一版对。)
    """
    if "_no_default_login" in request.fixturenames:
        yield None
        return

    from app.auth import ADMIN, AuthenticatedUser, require_admin, require_user
    from app.main import app as fastapi_app

    fake = AuthenticatedUser(username="tests-default-user", role=ADMIN)
    fastapi_app.dependency_overrides[require_user] = lambda: fake
    fastapi_app.dependency_overrides[require_admin] = lambda: fake
    yield
    fastapi_app.dependency_overrides.pop(require_user, None)
    fastapi_app.dependency_overrides.pop(require_admin, None)
