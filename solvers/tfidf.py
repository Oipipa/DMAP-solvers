# solvers/tfidf.py  — refined output
import math
from sympy import Rational, latex

def _esc(txt):
    return txt.replace('_', r'\_')

def _fmt(num, places=4):
    """Return 'exact (decimal)' for a float / Rational."""
    return rf"{latex(Rational(num).limit_denominator())}~({num:.{places}f})"

def solve_tfidf(docs, queries):
    """
    docs    : list[str]
    queries : list[(term, docIdx)]   docIdx is 1-based
    """
    N     = len(docs)
    steps = []

    # ─── (a) Corpus overview ──────────────────────────────────────────
    steps.append(r"\mathbf{(a)\ Corpus\ }(N="+str(N)+")")
    tokenised = []
    for i, d in enumerate(docs, 1):
        words = d.lower().split()
        tokenised.append(words)
        steps.append(rf"D_{i}:~``\text{{{_esc(d)}}}`` \; [{len(words)}\ \text{{tokens}}]")

    # ─── Build TF counts & DF table ───────────────────────────────────
    tf_counts = []
    df = {}
    for words in tokenised:
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        tf_counts.append((counts, len(words)))
        for w in counts:
            df[w] = df.get(w, 0) + 1

    # Pretty DF table
    steps.append(r"\mathbf{(b)\ Document\ Frequency\ table}")
    df_header = " & ".join(_esc(w) for w in df)
    df_values = " & ".join(str(df[w]) for w in df)
    steps.append(r"\begin{array}{|" + "c|"*len(df) + r"}\hline "
                 + df_header + r"\\\hline "
                 + df_values + r"\\\hline\end{array}")

    # ─── (c) Per-query derivations ────────────────────────────────────
    for q_no, (term, idx) in enumerate(queries, 1):
        t = term.lower()
        j = idx - 1
        counts, size = tf_counts[j]

        steps.append(rf"\mathbf{{(c{q_no})}}\ \text{{Term “{_esc(term)}” in }}D_{idx}")

        # TF
        tf_exact = counts.get(t, 0) / size
        steps.append(r"\underline{\text{TF calculation}}")
        steps.append(
            rf"\text{{occurrences}}={counts.get(t,0)},\;|D_{idx}|={size}"
            rf" \;\Rightarrow\; \text{{TF}}={_fmt(tf_exact,5)}"
        )

        # IDF
        idf = math.log10(N / df.get(t, 1))
        steps.append(r"\underline{\text{IDF calculation}}")
        steps.append(
            rf"\text{{df}}={df.get(t,0)},\;N={N}"
            rf"\;\Rightarrow\; \text{{IDF}}=\log_{{10}}\frac{{{N}}}{{{df.get(t,0)}}}"
            rf"={idf:.5f}"
        )

        # TF-IDF
        tfidf = tf_exact * idf
        steps.append(r"\underline{\text{TF-IDF}}")
        steps.append(rf"{_fmt(tf_exact,5)} \times {idf:.5f} = {_fmt(tfidf,5)}")
        steps.append(r"\ ")

    # ─── (d) Interpretation ───────────────────────────────────────────
    steps.append(r"\mathbf{(d)\ Interpretation}")
    steps.append(
        r"\text{High document frequency }(\text{df})\Downarrow \text{ low IDF}"
        r"\Downarrow \text{ lower TF-IDF for common terms.}"
    )

    return steps
