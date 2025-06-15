from sympy import Matrix, latex

# ───────────────────────────────────────── helpers
def _tokenize(text):
    """lower-case, keep letters & apostrophes, ignore punctuation"""
    import re
    return re.findall(r"[a-zA-Z']+", text.lower())

def _esc(word):
    """escape underscores etc. for LaTeX table cells"""
    return word.replace('_', r'\_')

# ───────────────────────────────────────── solver
def solve_bow(docs, new_doc=None):
    """
    docs    : list[str]      – original corpus (≥1)
    new_doc : str | None     – optional extra doc “C”
    returns : list[str]      – LaTeX snippets
    """
    steps = []

    # (a) tokenise + show
    steps.append(r"\mathbf{(a)\ Tokenisation}")
    tokens_per_doc = []
    for i, d in enumerate(docs, 1):
        tok = _tokenize(d)
        tokens_per_doc.append(tok)
        steps.append(rf"D_{i}: ``\text{{{_esc(d)}}}`` $\;\to\;$ "
                     + ", ".join(rf"\texttt{{{_esc(w)}}}" for w in tok))

    # (b) vocabulary
    vocab = sorted({w for tok in tokens_per_doc for w in tok})
    steps.append(r"\mathbf{(b)\ Vocabulary\ (alphabetical)}")
    voc_table = " & ".join(rf"{k+1}:{_esc(w)}"
                           for k, w in enumerate(vocab))
    steps.append(r"\begin{array}{|" + "c|"*len(vocab) + r"}\hline "
                 + voc_table + r"\\\hline\end{array}")

    # helper -> vector
    def vec(tok):          # tok is list of words
        return [tok.count(w) for w in vocab]

    # (c) BoW matrix for original docs
    steps.append(r"\mathbf{(c)\ BoW\ Matrix}")
    rows = [vec(tok) for tok in tokens_per_doc]
    M = Matrix(rows)
    col_names = " & ".join(rf"\texttt{{{_esc(w)}}}" for w in vocab)
    arr  = r"\begin{array}{|" + "c|"*len(vocab) + r"}\hline "
    arr += col_names + r"\\\hline "
    for r in rows:
        arr += " & ".join(str(x) for x in r) + r"\\\hline "
    arr += r"\end{array}"
    steps.append(arr)

    # (d) optional new doc
    if new_doc:
        tok_new = _tokenize(new_doc)
        v_new   = vec(tok_new)
        steps.append(r"\mathbf{(d)\ New\ Document\ “C”}")
        steps.append(rf"``{_esc(new_doc)}`` "
                     r"$\;\to\;$ "
                     r"$["+ ",\;".join(map(str, v_new)) + "]$")
        # interpretation note
        unseen = [w for w in tok_new if w not in vocab]
        if unseen:
            steps.append(r"\text{(Words not in vocabulary are ignored: }"
                         + ", ".join(rf"\texttt{{{_esc(w)}}}" for w in unseen)
                         + r"\text{)}")

    # (e) summary note
    steps.append(r"\mathbf{(e)\ Note}")
    steps.append(r"\text{Each vector position corresponds to the "
                 r"alphabetical vocabulary index displayed above.}")

    return steps
