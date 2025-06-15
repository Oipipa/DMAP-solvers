import sympy as sp
from sympy import Rational, nsimplify, latex

DECIMALS = 5 


def _fmt(num):
    """
    Render num as either
      • 10            (if it is an integer)   OR
      • 3/4 (0.75000) (if it is fractional)
    """
    num   = sp.nsimplify(num)
    _, d  = num.as_numer_denom()
    if d == 1:                      # pure integer – no decimal needed
        return latex(num)
    frac = latex(num)
    dec  = f"{float(num):.{DECIMALS}f}"
    return rf"{frac}\,({dec})"


def _array(original, normalized):
    """Build a tiny LaTeX array showing original vs. normalized."""
    header = r"\text{Original} & \text{Normalized} \\ \hline"
    body   = r" \\ ".join(f"{_fmt(o)} & {_fmt(n)}"
                          for o, n in zip(original, normalized))
    return (r"\begin{array}{|c|c|}\hline "
            + header + r" "
            + body   + r" \\ \hline\end{array}")


def solve_normalization(data_list, method):
    """
    data_list : list of floats
    method    : 'minmax' | 'zscore' | 'mean' | 'decimal'
    returns   : list[str]  • each is a LaTeX snippet for your template
    """
    data  = [Rational(x) for x in data_list]
    n     = len(data)
    steps = []

    # (a) original data
    steps.append(r"\mathbf{(a)\ Original\ Data}")
    orig_line = r"X = \bigl[" + r" ,\; ".join(_fmt(x) for x in data) + r"\bigr]"
    steps.append(orig_line)

    if method == "minmax":
        mn, mx  = min(data), max(data)
        norm    = [(x - mn) / (mx - mn) for x in data]

        steps.append(r"\mathbf{(b)\ Min–Max\ Normalization}")
        steps.append(rf"\min(X)={_fmt(mn)},\;\max(X)={_fmt(mx)}")
        steps.append(
            rf"X' = \dfrac{{X-{latex(mn)}}}{{{latex(mx)}-{latex(mn)}}}"
        )

    elif method == "zscore":
        μ  = sum(data) / n
        σ2 = sum((x - μ)**2 for x in data) / n
        σ  = sp.sqrt(σ2)
        norm = [(x - μ) / σ for x in data]

        steps.append(r"\mathbf{(b)\ Z\!-\!Score\ Standardization}")
        steps.append(rf"\mu={_fmt(μ)},\;\sigma={_fmt(σ)}")
        steps.append(rf"X' = \dfrac{{X-{latex(μ)}}}{{{latex(σ)}}}")

    elif method == "mean":
        μ, mn, mx = sum(data) / n, min(data), max(data)
        norm = [(x - μ) / (mx - mn) for x in data]

        steps.append(r"\mathbf{(b)\ Mean\ Normalization}")
        steps.append(
            rf"\mu={_fmt(μ)},\;\min(X)={_fmt(mn)},\;\max(X)={_fmt(mx)}"
        )
        steps.append(
            rf"X' = \dfrac{{X-{latex(μ)}}}{{{latex(mx)}-{latex(mn)}}}"
        )

    elif method == "decimal":
        # smallest power of 10 that makes every |x/10^j| < 1
        j = max(sp.ceiling(sp.log(abs(x), 10)) if x != 0 else 0 for x in data)
        norm = [x / (10 ** j) for x in data]

        steps.append(r"\mathbf{(b)\ Decimal\ Scaling}")
        steps.append(rf"j=\lceil\log_{{10}}\max|X_i|\rceil = {int(j)}")
        steps.append(rf"X' = \dfrac{{X}}{{10^{int(j)}}}")

    else:                       # programmer error
        raise ValueError("unknown normalization method → " + method)

    steps.append(r"\mathbf{(c)\ Original\ vs.\ Normalized}")
    steps.append(_array(data, norm))

    return steps
