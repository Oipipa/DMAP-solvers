import sympy as sp
from sympy import Matrix, Rational, latex

def solve_svd(A_list):
    A=Matrix([[Rational(x) for x in row] for row in A_list])
    m,n=A.rows,A.cols
    M=A.T*A
    steps=[]
    steps.append(r"\mathbf{(a)\ Compute\ }M=A^T A")
    steps.append(rf"A={latex(A)}")
    steps.append(rf"A^T={latex(A.T)}")
    steps.append(rf"M={latex(M)}")
    evects=M.eigenvects()
    eigen_data=[]
    for val,_,vecs in evects:
        for v in vecs:
            v_n=v/v.norm()
            if v_n[0]<0: v_n=-v_n
            eigen_data.append((val,v_n))
    eigen_sorted=sorted(eigen_data, key=lambda t:t[0], reverse=True)
    r=min(m,n)
    top=eigen_sorted[:r]
    steps.append(r"\mathbf{(b)\ Eigenvalues\ of\ }M")
    for i,(val,_) in enumerate(top,1):
        steps.append(rf"\lambda_{i}={latex(val)}")
    steps.append(r"\mathbf{(c)\ Orthonormal\ eigenvectors\ of\ }M")
    for i,(_,v) in enumerate(top,1):
        steps.append(rf"v_{i}={latex(v)}")
    V=Matrix.hstack(*[v for _,v in top])
    steps.append(r"\mathbf{(d)\ Orthogonal\ matrix\ }V")
    steps.append(rf"V={latex(V)}")
    sigmas=[sp.sqrt(val) for val,_ in top]
    steps.append(r"\mathbf{(e)\ Singular\ values\ }\sigma_i=\sqrt{\lambda_i}")
    for i,s in enumerate(sigmas,1):
        steps.append(rf"\sigma_{i}={latex(s)}")
    Sigma=Matrix([[0]*r for _ in range(r)])
    for i,s in enumerate(sigmas):
        Sigma[i,i]=s
    steps.append(r"\mathbf{(f)\ Diagonal\ matrix\ }\Sigma")
    steps.append(rf"\Sigma={latex(Sigma)}")
    steps.append(r"\mathbf{(g)\ Columns\ of\ }U\text{ via }u_i=\frac1{\sigma_i}A v_i")
    u_cols=[]
    for i,((_,v),s) in enumerate(zip(top,sigmas),1):
        Av=A*v
        ui=Av/s
        ui=ui/ui.norm()
        if ui[0]<0: ui=-ui
        u_cols.append(ui)
        steps.append(rf"v_{i}={latex(v.T)}")
        steps.append(rf"A v_{i}={latex(Av.T)}")
        steps.append(rf"u_{i}={latex(ui.T)}")
    U=Matrix.hstack(*u_cols)
    steps.append(rf"U={latex(U)}")
    recon=U*Sigma*V.T
    steps.append(r"\mathbf{(h)\ SVD\ check\ }A=U\Sigma V^T")
    steps.append(rf"U\Sigma V^T={latex(recon)}")
    steps.append(r"\text{Matches original }A" if recon==A else r"\text{Mismatch!}")
    return steps
