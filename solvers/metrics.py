# solvers/metrics.py
from collections import Counter
from sympy import Matrix, latex

def solve_metrics(y_true, y_pred):
    """
    y_true, y_pred : lists of labels (strings or ints)
    returns        : list[str] – LaTeX steps
    """
    steps = []
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths differ")

    # ── (a) Label set
    labels = sorted(set(y_true) | set(y_pred))
    L = len(labels)
    steps.append(r"\mathbf{(a)\ Classes}: " + ", ".join(map(str, labels)))

    # ── (b) Confusion matrix counts
    counts = Counter(zip(y_true, y_pred))
    M = Matrix([[counts.get((a,p), 0) for p in labels] for a in labels])

    # pretty table
    header = " & ".join(map(str, labels))
    body = r" \\ ".join(
        f"{labels[i]} & " + " & ".join(map(str, M.row(i)))
        for i in range(L)
    )
    steps.append(r"\mathbf{(b)\ Confusion\ Matrix}")
    steps.append(
        r"\begin{array}{|c|" + "c|"*L + r"}\hline "
        r"& " + header + r"\\\hline "
        + body + r"\\\hline\end{array}"
    )

    total = sum(M)
    correct = sum(M[i,i] for i in range(L))
    acc = correct/total
    steps.append(rf"\mathbf{{(c)\ Accuracy}} = "
                 rf"\frac{{{correct}}}{{{total}}} = {acc:.4f}")

    # ── per-class metrics
    precs, recs, f1s = [], [], []
    steps.append(r"\mathbf{(d)\ Per\!-\!class\ Metrics}")
    for i, cls in enumerate(labels):
        TP = M[i,i]
        FP = sum(M.row(j)[i] for j in range(L)) - TP
        FN = sum(M.row(i)) - TP
        prec = TP/(TP+FP) if TP+FP else 0
        rec  = TP/(TP+FN) if TP+FN else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        precs.append(prec); recs.append(rec); f1s.append(f1)
        steps.append(
            rf"{cls}: \; "
            rf"Prec={prec:.4f},\; Rec={rec:.4f},\; F1={f1:.4f}"
        )

    steps.append(r"\mathbf{(e)\ Macro\ Averages}")
    steps.append(
        rf"Precision={sum(precs)/L:.4f},\; "
        rf"Recall={sum(recs)/L:.4f},\; "
        rf"F1={sum(f1s)/L:.4f}"
    )

    return steps
