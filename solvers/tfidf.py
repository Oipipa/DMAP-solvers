import math

DEC = 4  # decimal places to show

def _esc(txt):
    return txt.replace('_', r'\_')

def _d(num, places=DEC):
    """Return a pure decimal string rounded to <places>."""
    return f"{num:.{places}f}"

def solve_tfidf(docs, queries):
    """
    docs    : list[str]
    queries : list[(term, docIdx)]   docIdx is 1-based
    returns : list of LaTeX snippets (all numeric output is decimal)
    """
    N     = len(docs)
    steps = []

    # (a) list corpus
    steps.append(rf"\mathbf{{(a)\ Corpus\ }}(N={N})")
    tokens = [d.lower().split() for d in docs]
    for i, (d, tok) in enumerate(zip(docs, tokens), 1):
        steps.append(rf"D_{i}: ``\text{{{_esc(d)}}}`` \; [{len(tok)}\ \text{{tokens}}]")

    tf_counts, df = [], {}
    for tok in tokens:
        cnt = {}
        for w in tok:
            cnt[w] = cnt.get(w, 0) + 1
        tf_counts.append((cnt, len(tok)))
        for w in cnt:
            df[w] = df.get(w, 0) + 1

    # (b) DF table (decimals not needed here)
    steps.append(r"\mathbf{(b)\ Document\ Frequency}")
    header = " & ".join(_esc(w) for w in df)
    vals   = " & ".join(str(df[w]) for w in df)
    steps.append(r"\begin{array}{|" + "c|"*len(df) + r"}\hline "
                 + header + r"\\\hline "
                 + vals   + r"\\\hline\end{array}")

    # (c) each query
    for q_no, (term, idx) in enumerate(queries, 1):
        term_l = term.lower()
        j      = idx - 1
        cnts, size = tf_counts[j]

        tf  = cnts.get(term_l, 0) / size
        idf = math.log10(N / df.get(term_l, 1))
        tfidf = tf * idf

        steps.append(rf"\mathbf{{(c{q_no})}}\ \text{{“{_esc(term)}” in }}D_{idx}")
        steps.append(r"\underline{\text{TF}} = "
                     rf"\frac{{{cnts.get(term_l,0)}}}{{{size}}} = {_d(tf)}")
        steps.append(r"\underline{\text{IDF}} = "
                     rf"\log_{{10}}\frac{{{N}}}{{{df.get(term_l,1)}}}"
                     rf" = {_d(idf)}")
        steps.append(r"\text{TF-IDF} = "
                     rf"{_d(tf)} \times {_d(idf)} = {_d(tfidf)}")
        steps.append(r"\ ")

    # (d) note
    steps.append(r"\mathbf{(d)\ Interpretation}")
    steps.append(
        r"\text{Common terms have high df} \;\Rightarrow\; "
        r"\text{low IDF} \;\Rightarrow\; \text{low TF-IDF.}"
    )

    return steps
