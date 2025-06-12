import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def solve_agglomerative(mat):
    n = len(mat)
    # Labels A, B, C, ...
    labels = [chr(65 + i) for i in range(n)]
    # Initial singleton clusters
    clusters = [[l] for l in labels]
    # Build a lookup of distances
    d = {}
    for i in range(n):
        for j in range(i+1, n):
            d[(i, j)] = d[(j, i)] = mat[i][j]

    steps = []
    # (a) initial distance matrix
    colspec = "c|" + "r"*n
    header = " & ".join([""] + labels)
    rows = []
    for i in range(n):
        row = [labels[i]] + [
            "-" if i == j else str(mat[i][j])
            for j in range(n)
        ]
        rows.append(" & ".join(row))
    array = (
        r"\begin{array}{" + colspec + "}\n"
        + header + r" \\ \hline" + "\n"
        + "\n".join(r + r" \\" for r in rows) + "\n"
        + r"\end{array}"
    )
    steps.append(r"\mathbf{(a)\ Initial\ distance\ matrix}")
    steps.append(rf"D = {array}")

    merges = []
    # Track x-positions of each current cluster (tuple key) for the dendrogram
    gap = 10
    cx = { (lab,): i*gap for i, lab in enumerate(labels) }

    step = 0
    # Agglomerative clustering (single‐link)
    while len(clusters) > 1:
        best_dist = None
        pair = (0, 1)
        # find pair of clusters with minimum single‐link distance
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = min(
                    d[(labels.index(p), labels.index(q))]
                    for p in clusters[i] for q in clusters[j]
                )
                if best_dist is None or dist < best_dist:
                    best_dist, pair = dist, (i, j)

        i, j = pair
        left, right = clusters[i], clusters[j]
        step += 1
        merges.append((left, right, best_dist))
        steps.append(
            rf"\mathbf{{Step\ {step}:}}\ merge\ {left}\ \&\ {right}\ at\ distance\ {best_dist}"
        )

        # Remove and append the merged cluster
        new_cluster = left + right
        for idx in sorted(pair, reverse=True):
            clusters.pop(idx)
        clusters.append(new_cluster)

        # (b) updated distance matrix
        m = len(clusters)
        lbl2 = [f"C{k+1}" for k in range(m)]
        col2 = "c|" + "r"*m
        hdr2 = " & ".join([""] + lbl2)
        rows2 = []
        for a in range(m):
            row = [lbl2[a]]
            for b in range(m):
                if a == b:
                    row.append("-")
                else:
                    dist = min(
                        d[(labels.index(p), labels.index(q))]
                        for p in clusters[a] for q in clusters[b]
                    )
                    row.append(str(dist))
            rows2.append(" & ".join(row))
        array2 = (
            r"\begin{array}{" + col2 + "}\n"
            + hdr2 + r" \\ \hline" + "\n"
            + "\n".join(r + r" \\" for r in rows2) + "\n"
            + r"\end{array}"
        )
        steps.append(r"\textbf{Updated\ distance\ matrix}")
        steps.append(rf"D = {array2}")

        # Record dendrogram x‐position for the new cluster
        left_key = tuple(sorted(left))
        right_key = tuple(sorted(right))
        x1, x2 = cx[left_key], cx[right_key]
        new_key = tuple(sorted(new_cluster))
        cx[new_key] = (x1 + x2) / 2

    # Build the dendrogram plot
    max_h = max(h for *_, h in merges) + 1
    fig, ax = plt.subplots(figsize=(8, 5))
    for left, right, h in merges:
        left_key = tuple(sorted(left))
        right_key = tuple(sorted(right))
        x1, x2 = cx[left_key], cx[right_key]
        # horizontal line
        ax.plot([x1, x2], [h, h], color='black')
        # vertical lines
        ax.plot([x1, x1], [0, h], color='black')
        ax.plot([x2, x2], [0, h], color='black')

    # Leaf labels: only the original singletons
    for key, x in cx.items():
        if len(key) == 1:
            ax.text(x, -0.5, key[0], ha='center', va='top')

    ax.set_ylabel("Distance")
    ax.set_xticks([])
    ax.set_ylim(-1, max_h + 1)
    ax.set_title("Single-linkage Dendrogram")

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    img_data = base64.b64encode(buf.getvalue()).decode('ascii')

    return steps, img_data
