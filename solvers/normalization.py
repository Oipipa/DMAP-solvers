# solvers/normalization.py

import sympy as sp
from sympy import Rational, Matrix, latex

def solve_normalization(data_list, method):
    # exact Rationals for clean symbolic work
    data = [Rational(x) for x in data_list]
    n = len(data)
    steps = []

    # (a) show original data vector
    steps.append(r"\mathbf{(a)\ Original\ Data}")
    steps.append(rf"X = {latex(Matrix([data]))}")

    if method == "minmax":
        # Min–Max
        mn = min(data); mx = max(data)
        norm = [(x - mn) / (mx - mn) for x in data]
        steps.append(r"\mathbf{(b)\ Min–Max\ Normalization}")
        steps.append(
            rf"\min(X)={latex(mn)},\;\max(X)={latex(mx)}"
        )
        steps.append(
            rf"X' = \frac{{X - {latex(mn)}}}{{{latex(mx)} - {latex(mn)}}}"
        )

    elif method == "zscore":
        # Z-Score
        μ = sum(data) / n
        σ = sp.sqrt(sum((x-μ)**2 for x in data) / n)
        norm = [(x - μ) / σ for x in data]
        steps.append(r"\mathbf{(b)\ Z\!-\!Score\ Standardization}")
        steps.append(
            rf"\mu = \frac{{\sum X_i}}{{n}} = {latex(μ)}"
        )
        steps.append(
            rf"\sigma = \sqrt{{\frac{{\sum (X_i-\mu)^2}}{n}}} = {latex(σ)}"
        )
        steps.append(
            rf"X' = \frac{{X - {latex(μ)}}}{{{latex(σ)}}}"
        )

    elif method == "mean":
        # Mean Normalization
        μ = sum(data) / n
        mn = min(data); mx = max(data)
        norm = [(x - μ) / (mx - mn) for x in data]
        steps.append(r"\mathbf{(b)\ Mean\ Normalization}")
        steps.append(
            rf"\mu = {latex(μ)},\;\min(X)={latex(mn)},\;\max(X)={latex(mx)}"
        )
        steps.append(
            rf"X' = \frac{{X - {latex(μ)}}}{{{latex(mx)} - {latex(mn)}}}"
        )

    elif method == "decimal":
        # Decimal Scaling
        # find j such that max|X'| < 1 when divide by 10^j
        j = max(len(str(abs(x.as_numer_denom()[0]))) -
                len(str(x.as_numer_denom()[1])) for x in data)
        norm = [x / (10**j) for x in data]
        steps.append(r"\mathbf{(b)\ Decimal\ Scaling}")
        steps.append(
            rf"j = \lceil \log_{{10}}\max(|X_i|)\rceil = {j}"
        )
        steps.append(
            rf"X' = \frac{{X}}{{10^{{{j}}}}}"
        )

    else:
        raise ValueError("Unknown method")

    # (c) display the paired original vs. normalized in a 2×n matrix
    steps.append(r"\mathbf{(c)\ Resulting\ Table\ (rows: original / normalized)}")
    M = Matrix([
        data,
        [sp.nsimplify(v) for v in norm]
    ])
    steps.append(rf"\begin{{bmatrix}} {latex(M)} \end{{bmatrix}}")

    return steps
