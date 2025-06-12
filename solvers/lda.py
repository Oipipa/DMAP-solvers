import sympy as sp
from sympy import Matrix, Rational, latex

def solve_lda(C1, C2):
    pts1 = [Matrix([Rational(x), Rational(y)]) for x, y in C1]
    pts2 = [Matrix([Rational(x), Rational(y)]) for x, y in C2]

    m1 = sum(pts1, Matrix([0, 0])) / len(pts1)
    m2 = sum(pts2, Matrix([0, 0])) / len(pts2)

    steps = []
    steps.append(r"\mathbf{(a)\ Class\ means}")
    steps.append(rf"m_1={latex(m1)}")
    steps.append(rf"m_2={latex(m2)}")

    S1 = Matrix([[0, 0], [0, 0]])
    for p in pts1:
        S1 += (p - m1) * (p - m1).T
    S2 = Matrix([[0, 0], [0, 0]])
    for p in pts2:
        S2 += (p - m2) * (p - m2).T
    SW = S1 + S2

    steps.append(r"\mathbf{(b)\ Within\text{-}class\ scatter\ }S_W")
    steps.append(rf"S_1={latex(S1)}")
    steps.append(rf"S_2={latex(S2)}")
    steps.append(rf"S_W={latex(SW)}")

    SB = (m1 - m2) * (m1 - m2).T
    steps.append(r"\mathbf{(c)\ Between\text{-}class\ scatter\ }S_B")
    steps.append(rf"S_B={latex(SB)}")

    w = SW.inv() * (m1 - m2)
    w = w / w.norm()
    steps.append(r"\mathbf{(d)\ LDA\ projection\ vector}")
    steps.append(rf"w={latex(w)}")

    steps.append(r"\mathbf{(e)\ Projections\ }y_i = w^T x_i")
    for i, p in enumerate(pts1 + pts2, 1):
        y = (w.T * p)[0]
        steps.append(rf"y_{i}={latex(y)}")

    w0 = -Rational(1, 2) * (w.T * (m1 + m2))[0]
    steps.append(r"\mathbf{(f)\ Optimal\ bias\ }w_0")
    steps.append(rf"w_0={latex(w0)}")
    steps.append(r"\text{Decision: class 1 if }w^T x + w_0 > 0\text{ else class 2}")

    return steps
