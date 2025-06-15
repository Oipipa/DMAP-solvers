import math
from sympy import Matrix, Rational, latex

def solve_emd(P_list, Q_list):
    # Convert to exact Rationals
    P = [Rational(x) for x in P_list]
    Q = [Rational(x) for x in Q_list]
    m, n = len(P), len(Q)
    steps = []

    # (a) Show original histograms
    steps.append(r"\mathbf{(a)\ Input\ Histograms}")
    steps.append(rf"P = {latex(Matrix([P]))}")
    steps.append(rf"Q = {latex(Matrix([Q]))}")

    # (b) Build and display the cost matrix d_{ij} = |i - j|
    D = Matrix([[abs(i-j) for j in range(n)] for i in range(m)])
    steps.append(r"\mathbf{(b)\ Cost\ Matrix\ }D = (d_{ij}),\quad d_{ij}=|i-j|")
    steps.append(rf"D = {latex(D)}")

    # (c) Initialize supplies, demands, and an all‐zero flow matrix
    supplies = P.copy()
    demands  = Q.copy()
    F = Matrix([[Rational(0) for _ in range(n)] for _ in range(m)])
    steps.append(r"\mathbf{(c)\ Initialize\ Supplies\ and\ Demands}")
    steps.append(rf"\text{{Supplies: }}{latex(Matrix([supplies]))}")
    steps.append(rf"\text{{Demands: }}{latex(Matrix([demands]))}")
    steps.append(r"\text{Flow matrix initialized to zero: }")
    steps.append(rf"F = {latex(F)}")

    # (d) Greedy NW‐corner allocation, recording each move
    steps.append(r"\mathbf{(d)\ NW\text{-}corner\ Allocation\ Steps}")
    i = j = 0
    step_no = 1
    while i < m and j < n:
        if supplies[i] == 0:
            steps.append(rf"\text{{Bin }}i={i+1}\ \text{{empty, advance to }}i={i+2}")
            i += 1
            continue
        if demands[j] == 0:
            steps.append(rf"\text{{Bin }}j={j+1}\ \text{{filled, advance to }}j={j+2}")
            j += 1
            continue

        # amount to move
        f = min(supplies[i], demands[j])
        F[i, j] += f
        supplies[i] -= f
        demands[j]  -= f

        steps.append(
            rf"\text{{(d{step_no}) Allocate }}f_{{{i+1},{j+1}}}="
            rf"\min(s_i,Q_j)=\min({latex(P[i])},{latex(Q[j])})={latex(f)}"
        )
        steps.append(rf"\text{{Updated flow matrix: }}F = {latex(F)}")
        steps.append(rf"\text{{Remaining supplies: }}{latex(Matrix([supplies]))}")
        steps.append(rf"\text{{Remaining demands: }}{latex(Matrix([demands]))}")
        step_no += 1

    # (e) Show final flow matrix
    steps.append(r"\mathbf{(e)\ Final\ Flow\ Matrix}")
    steps.append(rf"F = {latex(F)}")

    # (f) Compute total work (numerator) and total flow (denominator)
    work_terms = []
    total_flow = Rational(0)
    for i in range(m):
        for j in range(n):
            if F[i, j] != 0:
                work_terms.append(rf"{latex(F[i,j])}\times {latex(D[i,j])}")
                total_flow += F[i,j]

    numerator   = " + ".join(work_terms)
    steps.append(r"\mathbf{(f)\ Compute\ Totals}")
    steps.append(r"\text{Total work (numerator)}:")
    steps.append(rf"\sum_{i,j} f_{{ij}} d_{{ij}} = {numerator}")
    steps.append(r"\text{Total flow (denominator)}:")
    steps.append(rf"\sum_{i,j} f_{{ij}} = {latex(total_flow)}")

    # (g) Final EMD formula and numeric result
    steps.append(r"\mathbf{(g)\ Final\ EMD}")
    steps.append(r"\displaystyle \mathrm{EMD} = "
                 r"\frac{\sum_{i,j} f_{ij}\,d_{ij}}{\sum_{i,j} f_{ij}}")
    steps.append(rf"= \frac{{{numerator}}}{{{latex(total_flow)}}}")
    emd_value = sum(F[i,j] * D[i,j] for i in range(m) for j in range(n)) / total_flow
    steps.append(rf"= {latex(emd_value)}")

    return steps
