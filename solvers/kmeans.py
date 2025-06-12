from fractions import Fraction

def get_dist(method):
    if method == 'manhattan':
        return lambda a, b: sum(abs(x-y) for x,y in zip(a,b))
    if method == 'chebyshev':
        return lambda a, b: max(abs(x-y) for x,y in zip(a,b))
    return lambda a, b: sum((x-y)**2 for x,y in zip(a,b))

def nice_latex(n):
    from fractions import Fraction
    if isinstance(n, Fraction):
        if n.denominator == 1:
            return str(n.numerator)
        return r"\frac{%d}{%d}" % (n.numerator, n.denominator)
    return f"{n:.2f}"

def cluster_mean(coords):
    n = len(coords)
    d = len(coords[0])
    return tuple(sum(pt[i] for pt in coords)/n for i in range(d))

def solve_kmeans(points, centroids, iters, method='euclidean'):
    dist = get_dist(method)
    steps = []
    pts = list(points.keys())
    for it in range(1, iters+1):
        # (a) distances
        dmat = {p:{c:dist(points[p], centroids[c]) for c in centroids} for p in pts}
        cents = list(centroids.keys())
        colspec = "c|" + "r"*len(cents)
        header = " & ".join([""] + cents)
        rows = []
        for p in pts:
            row = [p] + [nice_latex(dmat[p][c]) for c in cents]
            rows.append(" & ".join(row))
        array = (
            r"\begin{array}{" + colspec + "}\n"
            + header + r" \\" + "\n"
            + r"\hline" + "\n"
            + "\n".join(r + r" \\" for r in rows) + "\n"
            + r"\end{array}"
        )
        steps.append(rf"\mathbf{{(a)\ Distances\ (iter\ {it},\ {method})}}: D^{{({it})}} = {array}")

        # (b) assignments
        assign = {}
        for p in pts:
            choice = min(dmat[p], key=dmat[p].__getitem__)
            assign.setdefault(choice, []).append(p)
        steps.append(r"\mathbf{(b)\ Assignments}")
        for c in cents:
            mem = assign.get(c, [])
            steps.append(f"{c}: " + (", ".join(mem) if mem else r"\text{—}"))

        # (c) new centroids
        steps.append(r"\mathbf{(c)\ New\ centroids}")
        newc = {}
        for c in cents:
            mem = assign.get(c, [])
            if mem:
                mu = cluster_mean([points[p] for p in mem])
            else:
                mu = centroids[c]
            newc[c] = mu
            vals = ",".join(nice_latex(v) for v in mu)
            steps.append(rf"{c}: \bigl({vals}\bigr)")
        centroids = newc

    return steps
