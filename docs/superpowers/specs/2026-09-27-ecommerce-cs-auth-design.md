# 电商智能客服系统 · 认证(JWT 登录)—— 设计文档

> **这不是一章**。它是 ch10 之后的一次**功能追加**,命名沿用
> `db/followup_messages_citations.sql` 那条先例(不带章号的产物)。用户 2026-09-27
> 口头点名要求,原话:「给我们云枢客服加一个登录功能吧,这样可以根据 userid 查
> conversationid,然后前端渲染会话栏预览,而且我刚才看会话点击不了了有个 bug,
> 登录就用 jwt 吧,然后后续请求就携带 jwt,没有就报 401,前端弹未认证」。

## 1. 目标

1. **按用户隔离会话**:侧栏列表与历史回载**只回当前登录用户自己的**会话。
   今天全部会话挂在字面量 `demo-user` 上(`app/api/conversations.py:DEMO_USER`),
   而这个字面量同时是「无认证」这个前提的产物。
2. **JWT 登录**:`POST /api/auth/login` 换 token;后续请求带
   `Authorization: Bearer <token>`;**没有 / 坏了 / 过期 ⇒ 401**。
3. **前端「弹未认证」**:任何请求拿到 401 ⇒ 清 token、弹登录浮层、登录成功后
   **重放那条原请求**。
4. 顺带修掉一个**真窟窿**:`ChatRequest.user_id` 今天让**客户端自称是谁**。

## 2. 非目标(YAGNI,一条都不做)

- **注册**(用户明确:「不用注册功能」)。账号由一份种子脚本预置。
- **刷新 token / 续期 / 登出黑名单**。token 到期就再登一次。
- **改密码 / 找回密码 / 账号管理页**。
- **多用户并发下的配额、限流、审计**(认证只回答「你是谁」)。
- **给 `topic_service`(8103)或两个 MCP Server(8101/8102)加鉴权** ——
  它们不在浏览器能碰到的路径上,是**服务端进程之间**的调用。
- **Langfuse trace 上补 user_id**。观测面本章不动。

## 3. 一条被推翻的既定非目标(必须记账)

`CLAUDE.md` 的「**全程不做**:多轮 Agent Loop、**认证**」里那条「认证」,
由**用户在 2026-09-27 明确要求**推翻。这不是实现者的自作主张:

- `CLAUDE.md` 那一行要改(改成「认证:2026-09-27 由用户要求补上,见 spec」);
- 本文件就是那条推翻的凭据;
- `dev-notes/ch10.md` 记一条**用户关键原话**(上面那段逐字)。

## 4. 用户拍板的四件事(逐字)

| # | 决定 | 出处 |
|---|---|---|
| 1 | **两个预置账号**:`cinfly` / `123456`、`demo-user` / `123456`,**不做注册** | 「用户名为cinfly,密码为123456,走个流程就行,以及demo-user,密码也是123456,不用注册功能」 |
| 2 | **JWT 密钥用 `itcinfly`** | 「jwtsecret用itcinfly」 |
| 3 | **两个账号都是超级用户**(能聊天**也能**看后台) | 「demo-user和cinfly都是超级用户(能聊天也能看后台)」 |
| 4 | 目的就是「**按用户查所有会话历史**」 | 「就是模拟个登录,然后能根据用户查所有会话历史就行」 |

⚠️ **决定 3 的连带事实(会被误读成 bug,要主动说)**:
存量的 581 条会话**全部** `user='demo-user'`。⇒ **用 `cinfly` 登录时侧栏是空的**
(它自己一条历史都没有);那 581 条只有 `demo-user` 看得见。
因为 `demo-user` 这个账号名**与存量字面量逐字相同**,**不需要任何数据迁移**。

## 5. 数据模型

### 5.1 新表 `users`(`db/auth.sql`,**刻意不幂等**,同其余几份 DDL)

```
id            BIGINT PK AUTO_INCREMENT
username      VARCHAR(128) NOT NULL UNIQUE      -- 与 conversations.user 同宽
password_hash VARCHAR(255) NOT NULL             -- scrypt 自描述串,见 §6.1
role          VARCHAR(16)  NOT NULL             -- 'user' | 'admin'
created_at    DATETIME     NOT NULL
```

- **不给 `conversations.user` 加外键**:那一列是 `String(128)` 且库里已有 581 行
  值,加 FK 要一条迁移,而收益只是「写错的 user 被数据库拦住」——
  写入方只有一个端点,且值来自 token。**如实记账:这是一处刻意的取舍。**
- `db/auth.sql` 放的是**建表**,种账号**不放在 SQL 里**(见 5.2)。

### 5.2 种子账号 `scripts/seed_users.py`

**幂等 upsert**(按 `username`),两个账号都写 `role='admin'`(决定 3)。
密码走 `app/auth.py:hash_password`,**每次运行生成新的随机盐** ⇒ 重跑 = 覆盖,
不是追加。

> **为什么不把密码哈希写进 `db/auth.sql`**:scrypt 的盐是随机的,写进 SQL 就
> 得先算一次再把**那一份**盐焊死在文件里;而一条 SQL 里的哈希**无法自证**
> (改密码要人来重算)。脚本每次现算,且**可重跑**。

## 6. 后端

### 6.1 `app/auth.py` —— **唯一**的鉴权边界

写法对齐 `app/observability.py`(本仓「全章唯一一个边界」的既有形状):
别的模块只认四个名字,换实现只改这一个文件。

| 名字 | 职责 |
|---|---|
| `hash_password(pw) -> str` | `hashlib.scrypt`(stdlib,**零新依赖**)+ 16 字节随机盐,存成自描述串 `scrypt$n$r$p$<b64盐>$<b64哈希>` |
| `verify_password(pw, stored) -> bool` | 解析上面那串,**常数时间比较**(`hmac.compare_digest`) |
| `create_token(*, username, role, settings) -> str` | HS256,claims = `sub` / `role` / `iat` / `exp` |
| `decode_token(token, settings) -> dict` | **显式 `algorithms=["HS256"]`**(不给的话 PyJWT 抛 —— 这是防 alg 混淆的机制,别绕过);失败一律翻译成「未认证」 |
| `require_user(creds, settings) -> AuthenticatedUser` | FastAPI 依赖:**任何登录用户** |
| `require_admin(user) -> AuthenticatedUser` | FastAPI 依赖:**`role == 'admin'`**,否则 **403** |

> 依赖只有**这两个**名字(全仓统一:`require_*`,与 `get_session` / `get_settings`
> 同一种命名)。⚠️ 本文档早先草稿里同一个东西写过 `current_user` 与 `require_user`
> 两个名字 —— **已经统一成 `require_user`**,实现时以这里为准。

`AuthenticatedUser` = `app/auth.py` 里一个 `@dataclass(frozen=True)`,
只有两个字段:`username` / `role`。**不返回 ORM 对象** —— 依赖不该把
一个活着的 `AsyncSession` 绑到一个跨函数的返回值上(本仓那条「读库的值要用新
session」的同族考虑),而且 `frozen` 让「端点悄悄改了自己的身份」不可能。

⚠️ **必须用 `HTTPBearer(auto_error=False)` 自己抛 401**(实现时用 Context7 核一遍):
FastAPI 的 `HTTPBearer` 在**缺 header 时默认抛 403**,而本需求要的是 401
(「没有就报401」)。两者对前端是两件事:401 ⇒ 去登录;403 ⇒ 登录了但没权限。
⇒ 依赖里显式 `raise HTTPException(401, "未认证", headers={"WWW-Authenticate": "Bearer"})`。

**401 与 403 的分工(前端只对 401 弹登录)**:

| 情况 | 状态码 |
|---|---|
| 没带 header / header 形状不对 / 签名错 / 过期 / 查无此人 | **401 未认证** |
| 是合法用户但 `role != admin`,打工作台端点 | **403 权限不足** |

### 6.2 `app/api/auth.py`(新 router)

| 端点 | 权限 | 说明 |
|---|---|---|
| `POST /api/auth/login` | **公开** | body `{username, password}`;成功 ⇒ `{token, username, role, expires_at}`;失败 ⇒ **401**,文案**不区分**「用户不存在」与「密码错」(不给枚举用户名留口子) |
| `GET /api/auth/me` | 已登录 | 回 `{username, role}` —— 前端启动时用它验一下手里的 token 还有效没 |

### 6.3 挂依赖:用 **router 级** `dependencies=[...]`

九个 router 各改**一行**:

```python
router = APIRouter(dependencies=[Depends(require_user)])     # 用户面 5 个
router = APIRouter(dependencies=[Depends(require_admin)])    # 工作台 4 个
```

需要**用到**用户身份的端点(chat / conversations / feedback / refund / ticket)
在签名里再加一个 `user: AuthenticatedUser = Depends(require_user)` ——
FastAPI 的依赖在**同一请求内按 callable 缓存**,所以那个函数**不会跑两遍**。

> **为什么用 router 级而不是每个端点各写一遍**:① 9 行 vs 25 处,少 16 个
> 「调用点自己记得做」的机会(本仓那条「不变量要放在唯一写口上」);
> ② 它让 §8 那条**结构性测试**变成一次 `for route in app.routes` 的可断言事实。

### 6.4 权限矩阵(**27** 个操作,逐条列出)

> 数一遍:改动前 `app/api/` 下共 **25** 个操作(chat 2 / conversations 2 /
> extract 1 / feedback 1 / kb 12 / refund 1 / review 4 / topics 1 / traces 1),
> 本次新增 login 与 me 各 1 ⇒ **27**。
> 下面三张表加起来必须是 27 条:**1 + 7 + 18 + 1**。

**公开(1)**:`POST /api/auth/login`

**用户面 `require_user`(7)**

| 端点 | 文件 |
|---|---|
| `POST /api/chat/stream` | `app/api/chat.py` |
| `POST /api/ticket` | `app/api/chat.py` |
| `GET /api/conversations` | `app/api/conversations.py` |
| `GET /api/conversations/{id}/messages` | `app/api/conversations.py` |
| `POST /api/extract` | `app/api/extract.py` |
| `POST /api/feedback` | `app/api/feedback.py` |
| `POST /api/refund` | `app/api/refund.py` |

**工作台 `require_admin`(18)**

| 端点 | 文件 |
|---|---|
| `GET /api/kb/stats` / `documents` / `documents/{name}` / `search` / `eval` / `jobs` / `jobs/{id}` | `app/api/kb.py` |
| `POST /api/kb/documents` / `jobs/vectorize` / `jobs/mine` / `jobs/flywheel` / `jobs/eval` | `app/api/kb.py` |
| `GET /api/review/queue` / `{id}`、`POST /api/review/{id}/approve` / `reject` | `app/api/review.py` |
| `GET /api/topics/distribution` | `app/api/topics.py` |
| `GET /api/traces/recent` | `app/api/traces.py` |

**已登录(不区分 role,1)**:`GET /api/auth/me`

⚠️ **聊天页也在打工作台端点**(引用弹层的「查看原文」两处:`/api/kb/documents`
与 `/api/kb/documents/{name}`,见 `app/static/index.html:787` / `:859`)。
⇒ 如果哪天新增一个 `role='user'` 的账号,他在聊天页点「查看原文」会拿到 **403**。
今天两个账号都是 admin,所以不可达;**这条写在这里,免得后来人把它当新 bug**。

### 6.5 删掉 `user_id`:客户端再也不能自称是谁

`app/schemas.py` 里 `ChatRequest.user_id`(第 94 行)与同族的请求字段**一律删除**,
`user_id` 改由 token 的 `sub` 提供。实测依据:7 个验收脚本**一个都不发**这个字段
(`grep -rn "user_id" scripts/*.sh` 为空),所以删它是安全的。

连带要改的**注释**(本仓那条「本次改动别让别处的注释变假」):
- `app/schemas.py:70` 那句「`user_id` 的上限 128 与 `conversations.user` 的列宽一致」
  —— 字段删了,这句话要跟着走(列宽那条事实挪到 §5.1)。
- `app/api/conversations.py:19-21` 的 `DEMO_USER` 与它那段「与会话端点同一个字面量」
  的说明 —— 字面量没了,过滤值改成 token。

## 7. 前端

### 7.1 新增 `app/static/auth.js`(两个页面共用)

它是**唯一**一处实现「401 怎么办」的地方:

- `authFetch(path, opts)` —— 自动带 `Authorization`;**401 ⇒ 清 token → 弹登录
  浮层 → 登录成功后重放原请求**(`index.html` 里 `fetch` 的语义是流式的,
  重放只对非流式请求做,见 7.3)。
- `showLogin()` —— 浮层:用户名 / 密码 / 「登录」;失败时显示服务端那句 401 文案。
  **账号直接印在浮层上**(`cinfly / 123456`、`demo-user / 123456`)—— 演示用,
  用户明确说「走个流程就行」。
- `getToken()/setToken()/clearToken()` —— 存 `localStorage["mewhelp.jwt"]`,
  与现有的 `mewhelp.session_id` 同一个地方。
- `currentUser()` / 「退出」 —— 退出 = 清 token + 弹回登录浮层。

### 7.2 两页的接入点

| 页面 | 改动 |
|---|---|
| `index.html` | 8 处 `fetch(` 全部换 `authFetch`(逐条:feedback / chat/stream / kb/documents ×2 / refund / ticket / conversations ×2) |
| `admin.html` | **只有 1 处要改**:`req()`(`:441`)—— 全部调用都收口在它身上 |

### 7.3 一处**刻意不做**的事:`chat/stream` 的自动重放

`/api/chat/stream` 是 SSE 流。401 会在**流开始之前**返回(端点是普通
`async def` 手工构造 `EventSourceResponse` —— 预算校验的既有形状),所以
**首次请求**拿 401 时弹登录、登录成功后再让用户点一次「发送」是安全的。
**不做自动重放**:一次用户消息已经出现在屏幕上、又被后台重发一遍,是比
「再点一次」更糟的失败。⇒ 重放只覆盖**幂等的 GET 类调用**。

### 7.4 token 存放的取舍(如实记账)

放 `localStorage` ⇒ **XSS 能读走它**。本仓没有 httpOnly cookie + CSRF 那一套
(单页、无构建工具链、演示规模),而需求原话是「后续请求就携带 jwt」。
⇒ 采用 `localStorage` + `Authorization` 头,**把代价写在这里**。

## 8. 测试

### 8.1 既有测试怎么不被冲垮(16 个文件、约 79 处端点调用)

`tests/conftest.py` 加一个 **autouse fixture**:默认给
`app.dependency_overrides` 装上「一个已登录的 admin 用户」,测试结束清掉。
⇒ 那 79 处调用**一个字都不用改**。

### 8.2 但那个 fixture 会造出**假绿**,所以要配一条结构性测试

⚠️ 全局 override 让「这个端点到底受不受保护」在测试里**恒真** ——
正是本仓编目过的形态 ⑦「替身替被测对象完成了语义」:一个**忘了挂依赖**的
router 在整套测试里全绿。⇒ 必须**另有一条不经过 override 的断言**:

**`tests/test_auth_wiring.py`**:遍历 `app.routes` 里的每一条 `APIRoute`,
断言它的 `dependencies` 里含 `require_user` 或 `require_admin`。
**白名单只有一条**:`POST /api/auth/login`(理由写在文件里)。
再加一条反向断言:`/api/auth/me` 必须挂 `require_user` 而**不**要求 admin
(它自己 `require_user` 的实现里就调了 `decode_token`,所以「me 有效」= 「token 有效」)。

### 8.3 新增测试

| 文件 | 覆盖 |
|---|---|
| `tests/test_auth.py` | 哈希(同密码两次盐不同)、`verify_password` 对错密码/坏串、签发与解码、**过期**、**篡改签名**、**alg 混淆**(把 header 改成 `none` 必须被拒) |
| `tests/test_api_auth.py` | 登录成功 / 错密码 401 / 用户不存在 401(**文案与错密码逐字相同**)/ 缺 header 401(**不是 403**)/ 过期 401 / 非 admin 打工作台 403 / `me` 回正确的 username 与 role |

### 8.4 一条要**在真库上**验的(db)

`test_api_conversations_db.py` 补一条:**两个用户的会话在侧栏里互不可见**
(造两条不同 `user` 的会话,分别用两个 token 拉,断言各自只看见自己的)。
理由与 ch07 那条同款:过滤是 **SQL 里**的事,替身替不出这个语义。

## 9. 配置

| 设置项 | 默认 | 说明 |
|---|---|---|
| `jwt_secret` | `""` | **不写死在代码里**。为空 ⇒ 每次进程启动**随机生成一个**并打一条 WARNING(重启后旧 token 全失效)。`.env` 与 `.env.example` 里都写 `JWT_SECRET=itcinfly`(用户拍板的值) |
| `jwt_expire_minutes` | `720`(12 小时) | 一次登录够一个工作日;一句话回退:`.env` 里调这个数 |

> **为什么默认是空而不是 `itcinfly`**:密钥进**源码**与进**配置文件**是两件事 ——
> 后者可以换、可以不入库(`.env` 实测**已被 gitignore**)。留空 + 随机是为了
> 「谁 clone 下来不配也能跑」,而 `.env.example` 里给出演示值。
> ⚠️ `.env.example` 是**入库**的 ⇒ `itcinfly` 因此是**公开值**,
> 只能用于本机演示。**这条必须写在 `.env.example` 的注释里。**

`requirements.txt` **显式加 `PyJWT==2.14.0`**:它今天只是 `mcp` 的传递依赖
(`pip show PyJWT` ⇒ `Required by: mcp`),mcp 哪天不带它了,登录会**突然崩**,
而报错指向 import。理由写进那一行的注释。

## 10. 验收脚本(7 个,100 处调用)

每个脚本**开头加三行**(放在服务起来之后):

```bash
TOKEN=$(curl -s -X POST "http://127.0.0.1:$PORT/api/auth/login" \
          -H 'Content-Type: application/json' \
          --data-binary '{"username":"cinfly","password":"123456"}' \
        | .venv/Scripts/python.exe -c "import json,sys;print(json.load(sys.stdin)['token'])")
# ⚠️ 遮蔽 curl:让下面**一百处**调用一个字都不用改。用 cinfly(admin)一个 token
#    覆盖用户面 + 工作台两侧。
curl() { command curl -H "Authorization: Bearer $TOKEN" "$@"; }
```

⚠️ 三个坑(实现时要核):
1. **函数遮蔽 vs 子 shell**:`curl()` 只在**当前 shell** 生效;脚本里若有
   `( ... )` 子 shell 或 `bash -c`,那些调用**不受影响**(在那些地方补 header)。
2. **`--data-binary @-` 与 stdin 共存**:多个脚本用 heredoc 喂 JSON,
   `curl` 的函数包装**不能**吃掉 stdin(用了 `"$@"` 转发,stdin 原样透传 —— 但要实测)。
3. **`ch08` / `ch09` 自己起服务**:登录必须排在**服务就绪之后**(它们各有
   「等端口」的既有做法,跟着走)。

**验收脚本改完必须重跑一遍**(不是「改完就算」):README/CLAUDE.md 那条
「先清残留进程」照旧,否则 curl 到旧服务 ⇒ 那一类假红。

## 11. 实施顺序(交给 writing-plans 展开)

1. `db/auth.sql` + `scripts/seed_users.py` + `app/auth.py` + `tests/test_auth.py`
   —— **纯函数内核先立住**,此时还没有任何端点被保护。
2. `app/api/auth.py`(登录 / me)+ `tests/test_api_auth.py` + `app/main.py` 挂路由。
3. **删 `user_id`**(schemas + chat/refund/conversations 的接线)+ 既有测试跟改。
4. **挂依赖**(9 个 router 各一行)+ conftest 的 autouse fixture +
   `tests/test_auth_wiring.py`;跑全量,把「79 处调用」那一批的回归收干净。
5. `app/static/auth.js` + 两个页面的接入(纯 UI,按项目规矩走 Vibe Coding)。
6. 7 个验收脚本 + 逐个重跑。
7. 文档:CLAUDE.md(改「全程不做:认证」那一行 + 架构段 + 高频命令)、
   `.env.example`、`dev-notes/ch10.md`(用户原话、关键产出、翻车)。

## 12. 实现订正

(实现期间发现与本文档的偏离写在这里,注明「为什么」——本仓每份 spec 的既有做法。)
