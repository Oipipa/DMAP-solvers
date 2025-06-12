from itertools import combinations

def solve_purity(true_labels, pred_labels):
    table = {}
    classes = sorted(set(true_labels), key=true_labels.index)
    clusters = sorted(set(pred_labels), key=pred_labels.index)
    for t in classes:
        table[t] = {c:0 for c in clusters}
    for t, p in zip(true_labels, pred_labels):
        table[t][p] += 1

    steps = []
    # (a) contingency table
    colspec = "c|" + "r"*len(clusters)
    header = " & ".join([""] + clusters)
    rows = []
    for t in classes:
        row = [t] + [str(table[t][c]) for c in clusters]
        rows.append(" & ".join(row))
    array = (
        r"\begin{array}{" + colspec + "}\n"
        + header + r" \\ \hline" + "\n"
        + "\n".join(r + r" \\" for r in rows) + "\n"
        + r"\end{array}"
    )
    steps.append(r"\mathbf{(a)\ Contingency\ table}")
    steps.append(rf"D = {array}")

    # (b) purity
    N = len(true_labels)
    best = sum(max(table[t].values()) for t in classes)
    pur = best / N
    steps.append(r"\mathbf{(b)\ Purity}")
    steps.append(rf"\text{{Purity}} = \frac{{{best}}}{{{N}}} = {pur:.2f}")

    # (c) Rand Index
    tp = fp = fn = 0
    for i, j in combinations(range(N), 2):
        same_t = true_labels[i] == true_labels[j]
        same_p = pred_labels[i] == pred_labels[j]
        if same_t and same_p:
            tp += 1
        elif same_t and not same_p:
            fn += 1
        elif not same_t and same_p:
            fp += 1
    tn = (N*(N-1)//2) - tp - fp - fn
    ri = (tp + tn) / (tp + fp + fn + tn)
    steps.append(r"\mathbf{(c)\ Rand\ Index}")
    steps.append(rf"TP={tp},\ FP={fp},\ FN={fn},\ TN={tn}")
    steps.append(rf"\text{{Rand Index}} = \frac{{{tp}+{tn}}}{{{tp}+{fp}+{fn}+{tn}}} = {ri:.4f}")

    return steps
