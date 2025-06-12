import sympy as sp
from sympy import Matrix, Rational, sqrt, symbols, latex

def solve_pca(data):
    pts   = [Matrix([Rational(x), Rational(y)]) for x, y in data]
    n     = len(pts)
    S     = sum(pts, Matrix([0, 0]))
    mu    = S / n
    centred = [p - mu for p in pts]

    # (a) mean & centring
    a = [
        r"\mathbf{(a)\ Mean\ of\ the\ dataset}",
        rf"n={n}",
        rf"\mu=\frac{{1}}{{{n}}}\sum_i P_i=\frac{{{latex(S)}}}{{{n}}}={latex(mu)}",
    ]
    for i, p in enumerate(pts, 1):
        c = centred[i-1]
        a.append(rf"C_{i}={latex(p)}-\mu={latex(c)}")

    # (b) covariance
    xs = [v[0] for v in centred]
    ys = [v[1] for v in centred]
    cxx = sum(u*v for u,v in zip(xs, xs)) / (n-1)
    cxy = sum(u*v for u,v in zip(xs, ys)) / (n-1)
    cyy = sum(u*v for u,v in zip(ys, ys)) / (n-1)
    Sigma = Matrix([[cxx, cxy], [cxy, cyy]])
    b = [
        r"\mathbf{(b)\ Sample\ covariance\ matrix}",
        rf"\Sigma={latex(Sigma)}",
    ]

    # (c) eigenvalues & eigenvectors
    a1, a2, d = Sigma[0,0], Sigma[0,1], Sigma[1,1]
    lam = symbols('lam')
    poly = sp.expand((a1-lam)*(d-lam) - a2**2)
    l1 = sp.simplify((a1 + d + sqrt((a1+d)**2 - 4*(a1*d - a2**2))) / 2)
    l2 = sp.simplify((a1 + d - sqrt((a1+d)**2 - 4*(a1*d - a2**2))) / 2)
    eigs = [(l1, None), (l2, None)]
    c = [
        r"\mathbf{(c)\ Eigenvalues\ and\ eigenvectors}",
        rf"|\Sigma - \lambda I| = {latex(poly)} = 0",
        rf"\lambda_1={latex(l1)},\ \lambda_2={latex(l2)}",
    ]
    eigen = []
    for lam_i in (l1, l2):
        M = Sigma - lam_i*sp.eye(2)
        v = M.nullspace()[0]
        v = v / v.norm()
        if v[0] < 0: v = -v
        eigen.append((lam_i, v))
        c.append(rf"\text{{Eigenvector for }}\lambda={latex(lam_i)}:\ {latex(v)}")

    # (d) principal components
    eigen.sort(key=lambda t: t[0], reverse=True)
    d_comp = [r"\mathbf{(d)\ Principal\ components}"]
    d_comp.append(rf"PC_1={latex(eigen[0][1])}")
    d_comp.append(rf"PC_2={latex(eigen[1][1])}")

    # (e) projection
    e = [r"\mathbf{(e)\ Projection\ onto\ [PC1,PC2]}"]
    P = Matrix.hstack(eigen[0][1], eigen[1][1])
    e.append(rf"P={latex(P)}")
    for i, cvec in enumerate(centred, 1):
        coords = P.T * cvec
        e.append(rf"C_{i}\to{latex(coords)}")

    # (f) variance explained
    total_var = eigen[0][0] + eigen[1][0]
    ratio1 = sp.simplify(eigen[0][0] / total_var)
    ratio2 = sp.simplify(eigen[1][0] / total_var)
    f = [r"\mathbf{(f)\ Variance\ explained}"]
    f.append(rf"\text{{Total variance}}=\lambda_1+\lambda_2={latex(total_var)}")
    f.append(rf"\text{{Prop. var. by PC1}}=\frac{{\lambda_1}}{{\lambda_1+\lambda_2}}={latex(ratio1)}")
    f.append(rf"\text{{Prop. var. by PC2}}=\frac{{\lambda_2}}{{\lambda_1+\lambda_2}}={latex(ratio2)}")

    return a + b + c + d_comp + e + f
