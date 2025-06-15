# solvers/cross_entropy.py

import sympy as sp
from sympy import Rational, latex

def solve_cross_entropy(P_list, Q_list):
    """
    Step-by-step Cross-Entropy H(P,Q) and H(Q,P) with
    auto-fallback to decimal if any Rational is “too big.”
    """
    # 1) Convert inputs to exact Rationals
    P = [Rational(x) for x in P_list]
    Q = [Rational(x) for x in Q_list]
    steps = []

    # 2) Decide if we need to go numeric
    THRESHOLD = 100
    numeric_mode = False
    for r in P + Q:
        n, d = r.as_numer_denom()
        if abs(n) > THRESHOLD or abs(d) > THRESHOLD:
            numeric_mode = True
            break

    # (a) Show distributions
    steps.append(r"\mathbf{(a)\ Given\ distributions}")
    if numeric_mode:
        P_str = ", ".join(f"{float(p):.6f}" for p in P)
        Q_str = ", ".join(f"{float(q):.6f}" for q in Q)
        steps.append(rf"P = [{P_str}]")
        steps.append(rf"Q = [{Q_str}]")
    else:
        steps.append(rf"P = [{', '.join(latex(p) for p in P)}]")
        steps.append(rf"Q = [{', '.join(latex(q) for q in Q)}]")

    # (b) Compute H(P,Q)
    steps.append(r"\mathbf{(b)\ Compute\ }H(P,Q)=-\sum_i P(i)\log_2 Q(i)")
    term_reprs = []
    H_PQ = 0
    for i, (p, q) in enumerate(zip(P, Q), 1):
        t = -p * sp.log(q, 2)
        H_PQ += t

        if numeric_mode:
            tr = f"{float(t.evalf(6)):.6f}"
            steps.append(rf"\text{{term }}i={i}:\; {tr}")
        else:
            tr = latex(t)
            val = latex(t.evalf(6))
            steps.append(
                rf"\text{{term }}i={i}:\; -{latex(p)}\log_2\bigl({latex(q)}\bigr)"
                rf" = {tr} \approx {val}"
            )

        term_reprs.append(tr)

    if numeric_mode:
        sum_repr = " + ".join(term_reprs)
        total = f"{float(H_PQ.evalf(6)):.6f}"
        steps.append(rf"H(P,Q) = {sum_repr} = {total}")
    else:
        sum_repr = " + ".join(term_reprs)
        approx = latex(H_PQ.evalf(6))
        steps.append(rf"H(P,Q) = {sum_repr} \approx {approx}")

    # (c) Compute H(Q,P)
    steps.append(r"\mathbf{(c)\ Compute\ }H(Q,P)=-\sum_i Q(i)\log_2 P(i)")
    term2_reprs = []
    H_QP = 0
    for i, (p, q) in enumerate(zip(P, Q), 1):
        t2 = -q * sp.log(p, 2)
        H_QP += t2

        if numeric_mode:
            tr2 = f"{float(t2.evalf(6)):.6f}"
            steps.append(rf"\text{{term }}i={i}:\; {tr2}")
        else:
            tr2 = latex(t2)
            val2 = latex(t2.evalf(6))
            steps.append(
                rf"\text{{term }}i={i}:\; -{latex(q)}\log_2\bigl({latex(p)}\bigr)"
                rf" = {tr2} \approx {val2}"
            )

        term2_reprs.append(tr2)

    if numeric_mode:
        sum2_repr = " + ".join(term2_reprs)
        total2 = f"{float(H_QP.evalf(6)):.6f}"
        steps.append(rf"H(Q,P) = {sum2_repr} = {total2}")
    else:
        sum2_repr = " + ".join(term2_reprs)
        approx2 = latex(H_QP.evalf(6))
        steps.append(rf"H(Q,P) = {sum2_repr} \approx {approx2}")

    # (d) Interpretation
    steps.append(r"\mathbf{(d)\ Interpretation}")
    steps.append(
        r"\text{A lower }H(P,Q)\text{ indicates that the predicted "
        r"distribution }Q\text{ is closer to the true distribution }P."
    )

    return steps
