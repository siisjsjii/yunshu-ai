"""T10 变异:读一次、替一次、写一次(上一版写成 `open(w).write(open(r).read().replace(...))`
—— 写打开先截断了文件,读到的就是 0 字节,把 nodes.py 清空了)。"""
import io, sys

path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = io.open(path, encoding="utf-8").read()
n = s.count(old)
print(f"锚点命中 {n} 次", flush=True)
assert n == 1, f"锚点命中 {n} 次(必须是 1)"
with io.open(path, "w", encoding="utf-8", newline="") as fh:
    fh.write(s.replace(old, new))
