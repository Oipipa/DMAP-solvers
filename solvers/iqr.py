import math
from fractions import Fraction

def nice_frac(x):
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        return r"\frac{%d}{%d}" % (x.numerator, x.denominator)
    return f"{x:.4f}"

def median(lst):
    k = len(lst)
    return (lst[k//2] if k % 2 else (lst[k//2-1] + lst[k//2]) / 2)

def quartile(values):
    m = len(values)
    if m % 2 == 0:
        lower, upper = values[:m//2], values[m//2:]
    else:
        lower, upper = values[:m//2], values[m//2+1:]
    return median(lower), median(values), median(upper)

def solve_iqr(data):
    steps = []
    n = len(data)
    total = sum(data)
    mean_frac = Fraction(total, n)
    mean = total / n
    var = sum((x - mean)**2 for x in data) / n
    std = math.sqrt(var)

    # (a)
    steps.append(r"\mathbf{(a)\ Mean\ and\ standard\ deviation}")
    steps.append(rf"n = {n}")
    steps.append(
        rf"\mu = \frac{{\sum x_i}}{{{n}}} = \frac{{{total}}}{{{n}}} = {nice_frac(mean_frac)} = {mean:.4f}"
    )
    steps.append(r"\text{Squared deviations:}")
    for i, x in enumerate(data, 1):
        diff_frac = Fraction(x) - mean_frac
        diff_sq = diff_frac**2
        steps.append(
            rf"(x_{i}-\mu)^2 = ({x} - {nice_frac(mean_frac)})^2 = {nice_frac(diff_sq)}"
        )
    steps.append(rf"\sigma^2 = \frac{{\sum (x_i-\mu)^2}}{{{n}}} = {var:.4f}")
    steps.append(rf"\sigma = \sqrt{{{var:.4f}}} = {std:.4f}")

    # (b)
    steps.append(r"\mathbf{(b)\ Z\text{-}score\ outliers\ (|Z|>2)}")
    out_z = []
    for i, x in enumerate(data, 1):
        z = (x - mean) / std
        steps.append(rf"Z_{i} = \frac{{{x}-{mean:.4f}}}{{{std:.4f}}} = {z:.4f}")
        if abs(z) > 2:
            out_z.append(x)
    if out_z:
        oz = ",".join(str(o) for o in out_z)
        steps.append(rf"\text{{Outliers}}: \{{{oz}\}}")
    else:
        steps.append(r"\text{No outliers.}")

    # (c)
    data_sorted = sorted(data)
    Q1, Q2, Q3 = quartile(data_sorted)
    IQR = Q3 - Q1
    steps.append(r"\mathbf{(c)\ Q1,\ Q2,\ Q3,\ and\ IQR}")
    ds = ",".join(str(x) for x in data_sorted)
    steps.append(rf"\text{{Sorted data}}: \{{{ds}\}}")
    steps.append(rf"\text{{Median}} = {Q2}")
    steps.append(rf"Q_1 = {Q1}")
    steps.append(rf"Q_3 = {Q3}")
    steps.append(rf"IQR = Q_3 - Q_1 = {IQR}")

    # (d)
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    steps.append(r"\mathbf{(d)\ IQR\ method\ outliers}")
    steps.append(
        rf"Lower\ bound = Q_1 - 1.5\times IQR = {Q1} - 1.5\times{IQR} = {lower}"
    )
    steps.append(
        rf"Upper\ bound = Q_3 + 1.5\times IQR = {Q3} + 1.5\times{IQR} = {upper}"
    )
    out_iqr = [x for x in data_sorted if x < lower or x > upper]
    if out_iqr:
        oi = ",".join(str(o) for o in out_iqr)
        steps.append(rf"\text{{Outliers}}: \{{{oi}\}}")
    else:
        steps.append(r"\text{No outliers.}")

    return steps
