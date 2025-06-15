import numpy as np
from sklearn.metrics import pairwise_distances

def solve_dbscan(points_raw, eps_raw, min_samples_raw, metric):
    # parse inputs
    pts = np.array([
        list(map(float, line.strip().split(',')))
        for line in points_raw.strip().splitlines()
        if line.strip()
    ])
    eps = float(eps_raw)
    min_samples = int(min_samples_raw)
    n = len(pts)
    steps = []

    # (a) input points
    coords = "; ".join(
        "(%s)" % ", ".join("%.2f" % x for x in row)
        for row in pts
    )
    steps.append(
        "(a) \\lbrace p_i \\rbrace_{i=1}^{%d} = \\lbrace %s \\rbrace"
        % (n, coords)
    )

    # (b) parameters
    steps.append(
        "(b) metric = %s, \\epsilon = %.2f, \\min\\_samples = %d"
        % (metric, eps, min_samples)
    )

    # (c) full distance matrix
    D = pairwise_distances(pts, pts, metric=metric)
    rows = [" & ".join("%.2f" % D[i,j] for j in range(n)) for i in range(n)]
    mat = "\\begin{bmatrix}%s\\end{bmatrix}" % ("\\\\".join(rows))
    steps.append("(c) D = %s" % mat)

    # compute ε‐neighborhoods
    neigh = []
    for i in range(n):
        nbrs = [j for j in range(n) if j != i and D[i,j] <= eps]
        neigh.append(nbrs)

    # (d) regionQuery in a single aligned block with cases
    cases = []
    for i, nbrs in enumerate(neigh):
        if nbrs:
            pts_tex = "; ".join(
                "(%s)" % ", ".join("%.2f" % x for x in pts[j])
                for j in nbrs
            )
            entry = "\\lbrace %s \\rbrace" % pts_tex
        else:
            entry = "\\varnothing"
        cases.append("%s & i=%d" % (entry, i+1))

    case_block = "\\\\".join(cases)
    steps.append(
        "\\begin{aligned}"
        "(d)\\quad N_\\epsilon(p_i) &= \\begin{cases}%s\\end{cases}"
        "\\end{aligned}" % case_block
    )

    # (e) corePoints
    cores = [i for i, nbrs in enumerate(neigh) if len(nbrs) + 1 >= min_samples]
    core_tex = ", ".join("p_{%d}" % (i+1) for i in cores) if cores else "\\varnothing"
    steps.append(
        "\\begin{aligned}"
        "(e)\\quad \\mathrm{corePoints} &= \\lbrace %s \\rbrace"
        "\\end{aligned}" % core_tex
    )

    # (f) cluster creation & expansion
    labels = [-1] * n
    visited = set()
    cid = 0
    exp_lines = []
    for i in cores:
        if i in visited:
            continue
        visited.add(i)
        labels[i] = cid
        exp_lines.append("Create cluster %d with p_{%d}" % (cid+1, i+1))
        seeds = list(neigh[i])
        while seeds:
            q = seeds.pop(0)
            if q not in visited:
                visited.add(q)
                labels[q] = cid
                exp_lines.append("p_{%d} added to cluster %d" % (q+1, cid+1))
                if q in cores:
                    new = [r for r in neigh[q] if r not in visited and r not in seeds]
                    seeds.extend(new)
                    if new:
                        added = "; ".join("p_{%d}" % (r+1) for r in new)
                        exp_lines.append("expand from p_{%d}: add %s" % (q+1, added))
        cid += 1

    if not exp_lines:
        exp_lines = ["No clusters formed"]
    exp_block = "\\\\".join(exp_lines)
    steps.append(
        "\\begin{aligned}"
        "(f)\\quad %s"
        "\\end{aligned}" % exp_block
    )

    # (g) summary of clusters + noise
    summary_lines = []
    for c in range(cid):
        members = [str(idx+1) for idx, lbl in enumerate(labels) if lbl == c]
        mem_tex = ", ".join(members) if members else "\\varnothing"
        summary_lines.append("Cluster %d: \\lbrace %s \\rbrace" % (c+1, mem_tex))
    noise = [str(idx+1) for idx, lbl in enumerate(labels) if lbl == -1]
    noise_tex = ", ".join(noise) if noise else "\\varnothing"
    summary_lines.append("Noise: \\lbrace %s \\rbrace" % noise_tex)

    summary_block = "\\\\".join(summary_lines)
    steps.append(
        "\\begin{aligned}"
        "(g)\\quad %s"
        "\\end{aligned}" % summary_block
    )

    return steps
