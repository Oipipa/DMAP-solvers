from itertools import combinations

def solve_eclat(transactions, minsup, maxk=None, minconf=0):
    tidlists = {}
    for tid, items in transactions.items():
        for it in items:
            tidlists.setdefault(it, set()).add(tid)
    steps = ["\\mathbf{(a) Initial TID-lists}"]
    for it in sorted(tidlists):
        tl = sorted(tidlists[it])
        steps.append(f"\\{{{it}\\}} : \\{{{','.join(tl)}\\}}")
    L1 = {it: tl for it, tl in tidlists.items() if len(tl) >= minsup}
    freq = {frozenset([it]): len(tl) for it, tl in L1.items()}
    steps.append(f"\\mathbf{{(b) Frequent 1-itemsets (support ≥ {minsup})}}")
    for it, tl in sorted(L1.items()):
        steps.append(f"\\{{{it}\\}} : {len(tl)}")
    steps.append("\\mathbf{(c) Recursive intersections}")
    def recurse(prefix, prefix_tl, items):
        for idx, it in enumerate(items):
            newset = prefix | {it}
            newtl = (prefix_tl & tidlists[it]) if prefix else tidlists[it]
            if len(newtl) >= minsup:
                fs = frozenset(newset)
                freq[fs] = len(newtl)
                steps.append(f"\\{{{','.join(sorted(newset))}\\}} : support={len(newtl)}")
                if maxk is None or len(newset) < maxk:
                    recurse(newset, newtl, items[idx+1:])
    all_tids = set(transactions)
    recurse(set(), all_tids, sorted(L1))
    steps.append(f"\\mathbf{{(d) Association rules (conf ≥ {minconf:.2f})}}")
    for itemset in sorted(freq, key=lambda x: (len(x), sorted(x))):
        if len(itemset) >= 2:
            sup = freq[itemset]
            for r in range(1, len(itemset)):
                for ant in combinations(sorted(itemset), r):
                    antset = frozenset(ant)
                    conf = sup / freq[antset]
                    if conf >= minconf:
                        cons = itemset - antset
                        steps.append(
                            f"\\{{{','.join(sorted(antset))}\\}}\\to\\{{{','.join(sorted(cons))}\\}} : conf={conf:.2f}"
                        )
    return steps
