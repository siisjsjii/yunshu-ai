"""静态页的「接线」自检:JS 里写的那些 id 与目录,在 HTML 里**真的存在**吗。

为什么需要这样一份测试(而不是「看一眼就知道了」):本项目最近三次前端缺陷
里**有两次是名字对不上**那一族的 —— 一次是引用弹层的 × 绑在会被 `innerHTML`
替换掉的元素上(#113),一次是历史回载按角色画出错(#114)。这类缺陷在 pytest 里
**一条都抓不到**,因为静态页没有可导入的被测对象;而它们的症状又都长得像
「页面坏了」而不是「代码报错」。

**这里不跑浏览器**(本机没有 jsdom),只做源码级检查:
  ① JS 里每个**字面量** id(`$("x")` / `getElementById("x")`)在 HTML 里都有;
  ② 每个 `data-tab` 都有一个 `id="tab-<它>"` 的块;
  ③ `switchTab` 的 `TAB_NAMES` 与 HTML 里那串 `data-tab` **逐位相同**(顺序也算);
  ④ 每个 `data-goto` 都落在 `TAB_NAMES` 里。

它**证明不了**「页面在第 X 步之后仍然可用」——那要真浏览器点击。
它能证明的是**名字没有对不上**,而那恰好是这一族缺陷最常见的形状。
"""

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"

#: `$("x")` 与 `getElementById("x")` 两种写法。**只认字面量**:像
#: `$("tab-" + t)` 这种拼出来的 id 不在此列(正则要求引号紧跟右括号)。
_ID_REF_RE = re.compile(r'\$\("([^"]+)"\)|getElementById\("([^"]+)"\)')
_HTML_ID_RE = re.compile(r'\bid="([^"]+)"')
_DATA_TAB_RE = re.compile(r'data-tab="([^"]+)"')
_DATA_GOTO_RE = re.compile(r'data-goto="([^"]+)"')
_TAB_NAMES_RE = re.compile(r"const TAB_NAMES = \[([^\]]*)\]")


def _js(text: str) -> str:
    """所有 `<script>` 块拼起来(页面里现在只有一块,但不假设只有一块)。"""
    return "\n".join(re.findall(r"<script>(.*?)</script>", text, re.S))


def _refs(js: str) -> set[str]:
    return {a or b for a, b in _ID_REF_RE.findall(js)}


def _ids(html: str) -> set[str]:
    # ⚠️ 同一份文本里既有 HTML 也有 JS —— JS 里没有 `id="…"`,所以直接扫全文即可;
    #    但**不能**把 `data-tab="…"` 当成 id(它不带 `id=` 前缀,正则不会误伤)。
    return set(_HTML_ID_RE.findall(html))


@pytest.mark.parametrize("name", ["index.html", "admin.html"])
def test_every_referenced_id_exists(name):
    """`$("…")` 指错一个字母 ⇒ `document.getElementById` 回 `None` ⇒
    `None.textContent` 抛在**事件回调里**(页面上只表现为「点了没反应」)。
    """
    text = (STATIC / name).read_text(encoding="utf-8")
    missing = sorted(_refs(_js(text)) - _ids(text))
    assert not missing, (
        f"{name}: JS 里引用了 HTML 中不存在的 id:{missing}\n"
        f"(这一族的症状是「点了没反应」,不是报错 —— 正是这份测试要拦的)"
    )


def test_tabs_and_panels_match():
    """六个目录 ↔ 六个页,以及 `TAB_NAMES` 与 HTML 的那串**逐位相同**。

    ⚠️ `TAB_NAMES` 少一个名字的后果**不是报错**:`switchTab` 里那个循环遍历的是
    `TAB_NAMES`,漏掉的那个页**永远不会被隐藏**(它会在别的页上一直显示着),
    而目录按钮的 active 仍会跟着切。这一条就是钉它的。
    """
    text = (STATIC / "admin.html").read_text(encoding="utf-8")
    ids = _ids(text)
    tabs = _DATA_TAB_RE.findall(text)

    assert len(tabs) == len(set(tabs)), f"目录里有重复的 data-tab:{tabs}"
    for t in tabs:
        assert f"tab-{t}" in ids, f"目录「{t}」没有对应的 `id=\"tab-{t}\"` 块"
    # 反向:每个 tab-<X> 块都要有目录按钮(否则那一页进不去)
    for i in sorted(ids):
        if i.startswith("tab-"):
            assert i[len("tab-"):] in tabs, f"`{i}` 这一页没有目录按钮,进不去"

    m = _TAB_NAMES_RE.search(_js(text))
    assert m, "admin.html 里找不到 `const TAB_NAMES = [...]`"
    names = re.findall(r'"([^"]+)"', m.group(1))
    assert names == tabs, (
        f"TAB_NAMES 与 HTML 的 data-tab 必须逐位相同(顺序也算):\n"
        f"  TAB_NAMES = {names}\n  data-tab  = {tabs}"
    )


def test_home_cards_wire_up_to_existing_elements():
    """首页五张卡:每个 loader 的短名都要有一套真元素。

    ⚠️ 这一条**单独存在**的理由:`homeSay` 里那两个 id 是**拼出来的**
    (`"home-" + id + "-line"`),上面那条「字面量 id 必须存在」的正则**照不到它们**
    —— 而「五张卡里有一张写错了 id」在这页上的表现是**那张卡永远停在『读取中…』**,
    没有报错、另外四张完全正常。
    """
    text = (STATIC / "admin.html").read_text(encoding="utf-8")
    ids = _ids(text)
    block = re.search(r"const HOME_LOADERS = \[(.*?)\];", _js(text), re.S)
    assert block, "admin.html 里找不到 `const HOME_LOADERS = […]`"
    names = re.findall(r'\["([^"]+)",', block.group(1))
    assert len(names) == 5, f"首页该有五张卡,实际 {names}"
    for n in names:
        for suffix in ("-line", "-badge"):
            assert f"home-{n}{suffix}" in ids, (
                f"`home-{n}{suffix}` 在 HTML 里不存在 —— 这张卡会永远停在「读取中…」"
            )


def test_home_goto_buttons_point_at_real_tabs():
    """首页那张卡上的「进入 X」不许指到一个不存在的页。

    `switchTab("打错的字")` 的表现是:**什么都没发生**(所有页都被显示成 none,
    屏幕一片空白),而控制台不报错。
    """
    text = (STATIC / "admin.html").read_text(encoding="utf-8")
    tabs = set(_DATA_TAB_RE.findall(text))
    gotos = _DATA_GOTO_RE.findall(text)
    assert gotos, "首页一张卡都没有 `data-goto` ⇒ 入口按钮全断了"
    bad = sorted(set(gotos) - tabs)
    assert not bad, f"data-goto 指向不存在的页:{bad}(可用的:{sorted(tabs)})"
