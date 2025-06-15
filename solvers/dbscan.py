# solvers/dbscan.py
import numpy as np
from sklearn.metrics import pairwise_distances

def solve_dbscan(points_raw, eps_raw, min_samples_raw):
    pts = np.array([list(map(float, line.split(','))) for line in points_raw.strip().splitlines() if line.strip()])
    eps = float(eps_raw)
    min_samples = int(min_samples_raw)
    n = len(pts)
    steps = []

    coords = ", ".join(f"({', '.join(f'{x:.2f}' for x in row)})" for row in pts)
    steps.append(r"(a) \{\mathbf p_i\}_{i=1}^%d = \{%s\}" % (n, coords))
    steps.append(r"(b) \epsilon = %.2f,\ \min\_samples = %d" % (eps, min_samples))

    D = pairwise_distances(pts, pts)
    rows = [" & ".join(f"{D[i,j]:.2f}" for j in range(n)) for i in range(n)]
    mat = r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"
    steps.append(r"(c) D = %s" % mat)

    region = []
    for i in range(n):
        nbrs = [j for j in range(n) if i != j and D[i,j] <= eps]
        region.append(nbrs)
        nbrs_tex = ", ".join(f"p_{j+1}" for j in nbrs) or r"\varnothing"
        steps.append(r"(d) \mathrm{regionQuery}(p_{%d}) = \{%s\},\ |N| = %d" % (i+1, nbrs_tex, len(nbrs)))

    core = [i for i,nbrs in enumerate(region) if len(nbrs)+1 >= min_samples]
    if core:
        core_tex = ", ".join(f"p_{i+1}" for i in core)
        steps.append(r"(e) \mathrm{corePoints} = \{%s\}" % core_tex)
    else:
        steps.append(r"(e) \mathrm{corePoints} = \{\varnothing\}")

    labels = [-1] * n
    visited = set()
    cluster = 0

    for i in range(n):
        if i in visited: continue
        visited.add(i)
        if i in core:
            labels[i] = cluster
            steps.append(r"(f) \text{Create cluster %d with }p_{%d}" % (cluster+1, i+1))
            seeds = set(region[i])
            while seeds:
                q = seeds.pop()
                if q not in visited:
                    visited.add(q)
                    if q in core:
                        for r in region[q]:
                            if r not in visited:
                                seeds.add(r)
                        steps.append(r"(g) \text{Expand seeds from core }p_{%d}" % (q+1))
                    labels[q] = cluster
                    steps.append(r"(h) p_{%d} \text{ added to cluster %d}" % (q+1, cluster+1))
            cluster += 1
        else:
            steps.append(r"(i) p_{%d} \text{ marked noise}" % (i+1))

    steps.append(r"(j) \text{Labels} = %s" % labels)

    for c in range(cluster):
        members = ", ".join(str(idx+1) for idx,l in enumerate(labels) if l == c)
        steps.append(r"(k) \text{Cluster %d} = \{%s\}" % (c+1, members or r"\varnothing"))

    noise = ", ".join(str(idx+1) for idx,l in enumerate(labels) if l == -1)
    steps.append(r"(l) \text{Noise} = \{%s\}" % (noise or r"\varnothing"))

    return steps
