from collections import Counter
from itertools import combinations

class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None

def build_fp_tree(transactions, order):
    header = {it: None for it in order}
    root = FPNode(None, 1, None)
    for items in transactions.values():
        node = root
        for it in order:
            if it in items:
                if it in node.children:
                    node.children[it].count += 1
                    node = node.children[it]
                else:
                    new = FPNode(it, 1, node)
                    node.children[it] = new
                    if header[it] is None:
                        header[it] = new
                    else:
                        cur = header[it]
                        while cur.link:
                            cur = cur.link
                        cur.link = new
                    node = new
    return root, header

def extract_prefix_paths(header, item):
    paths = []
    node = header[item]
    while node:
        cnt = node.count
        path = []
        p = node.parent
        while p and p.item:
            path.append(p.item)
            p = p.parent
        if path:
            paths.append((list(reversed(path)), cnt))
        node = node.link
    return paths

def solve_fpgrowth(transactions, minsup):
    freq = Counter(it for items in transactions.values() for it in items)
    L = [(it, freq[it]) for it in freq if freq[it] >= minsup]
    L.sort(key=lambda x: -x[1])
    order = [it for it, _ in L]
    steps = []
    steps.append(rf"\mathbf{{(a)\ Frequent\ pattern\ set\ }}L(\mathrm{{freq}}\ge {minsup})")
    for it, sup in L:
        steps.append(f"{it} : {sup}")
    steps.append(r"\mathbf{(b)\ Ordered‐item\ sets}")
    for tid, items in transactions.items():
        o = [it for it in order if it in items]
        steps.append(f"{tid} : \\{{{','.join(o)}\\}}")
    steps.append(r"\mathbf{(c)\ Build\ FP\ tree\ (header\ table)}")
    _, header = build_fp_tree(transactions, order)
    for it in order:
        total = 0
        node = header[it]
        while node:
            total += node.count
            node = node.link
        steps.append(f"{it} : total\\ count={total}")
    steps.append(r"\mathbf{(d)\ Conditional\ pattern\ bases\ and\ trees}")
    rows = []
    for it, _ in reversed(L):
        base = extract_prefix_paths(header, it)
        base_str = r"\{" + ", ".join(f"\\{{{','.join(p)}\\}}:{c}" for p, c in base) + r"\}"
        agg = Counter()
        for path, cnt in base:
            for x in path:
                agg[x] += cnt
        tree = {k: v for k, v in agg.items() if v >= minsup}
        tree_str = r"\{" + ", ".join(f"{k}:{v}" for k, v in tree.items()) + r"\}"
        rows.append((it, base_str, tree_str))
    tab = [r"\begin{array}{c|c|c}",
           r"Item & Conditional Pattern Base & Conditional FP-Tree \\ \hline"]
    for it, b, t in rows:
        tab.append(f"{it} & {b} & {t}\\\\")
    tab.append(r"\end{array}")
    steps.append("\n".join(tab))
    steps.append(r"\mathbf{(e)\ Frequent\ patterns\ mined}")
    final = []
    for r in range(2, len(order) + 1):
        for combo in combinations(order, r):
            sup = sum(1 for items in transactions.values() if set(combo).issubset(items))
            if sup >= minsup:
                final.append((combo, sup))
    for combo, sup in final:
        steps.append(f"\\{{{','.join(combo)}\\}} : {sup}")
    return steps
