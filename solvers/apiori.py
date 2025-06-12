from itertools import combinations

def solve_apiori(transactions_raw, min_sup_count, min_conf):
    # parse transactions as sets of strings
    transactions = [set(t) for t in transactions_raw]

    steps = []
    # (a) Minimum support
    steps.append(rf"\textbf{{Minimum support count}} = {min_sup_count}")

    # (b) Frequent 1-itemsets
    c1 = {}
    for t in transactions:
        for item in t:
            c1[item] = c1.get(item, 0) + 1
    L1 = {frozenset([i]): c for i, c in c1.items() if c >= min_sup_count}
    steps.append(r"\textbf{Frequent 1-itemsets}")
    for s, c in sorted(L1.items(), key=lambda x: (-x[1], sorted(x[0]))):
        items = ",".join(sorted(s))
        steps.append(rf"\{{{items}\}} : {c}")

    layers = [L1]
    support = dict(L1)
    k = 2
    prev_L = L1

    # (c) Generate & prune k-itemsets
    while prev_L:
        steps.append(rf"\textbf{{Generate candidate {k}-itemsets from L{k-1}}}")
        candidates = set()
        prev_keys = list(prev_L)
        for i in range(len(prev_keys)):
            for j in range(i+1, len(prev_keys)):
                union = prev_keys[i] | prev_keys[j]
                if len(union) == k and all(frozenset(sub) in prev_L
                                           for sub in combinations(union, k-1)):
                    candidates.add(union)
        steps.append(rf"\text{{Number of candidates}} = {len(candidates)}")

        counts = {}
        for t in transactions:
            for cand in candidates:
                if cand.issubset(t):
                    counts[cand] = counts.get(cand, 0) + 1

        Lk = {s: c for s, c in counts.items() if c >= min_sup_count}
        if not Lk:
            break

        steps.append(rf"\textbf{{Frequent {k}-itemsets}}")
        for s, c in sorted(Lk.items(), key=lambda x: (-x[1], sorted(x[0]))):
            items = ",".join(sorted(s))
            steps.append(rf"\{{{items}\}} : {c}")

        layers.append(Lk)
        support.update(Lk)
        prev_L = Lk
        k += 1

    # (d) Strong rules
    steps.append(rf"\textbf{{Minimum confidence}} = {min_conf:.2f}")
    steps.append(r"\textbf{Strong rules}")
    found = False
    for Lk in layers[1:]:
        for itemset in Lk:
            for r in range(1, len(itemset)):
                for ant in combinations(itemset, r):
                    ant = frozenset(ant)
                    cons = itemset - ant
                    conf = support[itemset] / support[ant]
                    if conf >= min_conf:
                        ant_items = ",".join(sorted(ant))
                        cons_items = ",".join(sorted(cons))
                        steps.append(
                            rf"\{{{ant_items}\}} \to \{{{cons_items}\}} : \mathrm{{conf}}={conf:.2f}"
                        )
                        found = True
    if not found:
        steps.append(r"\text{none}")

    return steps
