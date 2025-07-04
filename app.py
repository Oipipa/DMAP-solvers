from flask import Flask, render_template, request, redirect, url_for
from solvers.pca import solve_pca
from solvers.lda import solve_lda
from solvers.svd import solve_svd
from solvers.kmeans import solve_kmeans
from solvers.agglomerative import solve_agglomerative
from solvers.dbscan import solve_dbscan
from solvers.apiori import solve_apiori
from solvers.iqr import solve_iqr
from solvers.purity import solve_purity
from solvers.eclat import solve_eclat
from solvers.fpgrowth import solve_fpgrowth
from solvers.cross_entropy import solve_cross_entropy
from solvers.normalization import solve_normalization
from solvers.emd import solve_emd
from solvers.tfidf import solve_tfidf
from solvers.bow import solve_bow
from solvers.metrics import solve_metrics

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    mode = request.form['mode']
    img = None

    if mode == 'pca':
        data = [tuple(map(float, s.split(','))) for s in request.form['data'].split(';') if s.strip()]
        steps = solve_pca(data)

    elif mode == 'lda':
        C1 = [tuple(map(float, s.split(','))) for s in request.form['c1'].split(';') if s.strip()]
        C2 = [tuple(map(float, s.split(','))) for s in request.form['c2'].split(';') if s.strip()]
        steps = solve_lda(C1, C2)

    elif mode == 'svd':
        A = [list(map(float, row.split(','))) for row in request.form['A'].split(';') if row.strip()]
        steps = solve_svd(A)

    elif mode == 'kmeans':
        pts = [tuple(map(float, s.split(','))) for s in request.form['points'].split(';') if s.strip()]
        points = {f"P{i+1}": pt for i, pt in enumerate(pts)}
        cs = [tuple(map(float, s.split(','))) for s in request.form['centroids'].split(';') if s.strip()]
        centroids = {f"C{i+1}": c for i, c in enumerate(cs)}
        iters = int(request.form['iters'])
        method = request.form.get('distance', 'euclidean')
        steps = solve_kmeans(points, centroids, iters, method)

    elif mode == 'agglomerative':
        mat = [list(map(float, row.split(','))) for row in request.form['matrix'].split(';') if row.strip()]
        steps, img = solve_agglomerative(mat)

    elif mode == 'dbscan':
        pts_raw = request.form['dbscan_points']
        eps = request.form['dbscan_eps']
        min_samples = request.form['dbscan_min_samples']
        metric = request.form['dbscan_metric']
        steps = solve_dbscan(pts_raw, eps, min_samples, metric)

    elif mode == 'apiori':
        txns = [t.split(',') for t in request.form['transactions'].split(';') if t.strip()]
        minsup = int(request.form['minsup'])
        minconf = float(request.form['minconf'])
        steps = solve_apiori(txns, minsup, minconf)

    elif mode == 'iqr':
        data = [int(x) for x in request.form['iqr_data'].split(',') if x.strip()]
        steps = solve_iqr(data)

    elif mode == 'purity':
        true = [s for s in request.form['true'].split(',') if s.strip()]
        pred = [s for s in request.form['pred'].split(',') if s.strip()]
        steps = solve_purity(true, pred)

    elif mode == 'eclat':
        parts = [r.strip() for r in request.form['eclat_matrix'].split(';') if r.strip()]
        header = parts[0].split(',')[1:]
        txns = {}
        for row in parts[1:]:
            cols = row.split(',')
            tid = cols[0]
            bits = cols[1:]
            txns[tid] = {header[i] for i, b in enumerate(bits) if b == '1'}
        minsup = int(request.form['eclat_minsup'])
        maxk = request.form.get('eclat_maxk')
        maxk = int(maxk) if maxk and maxk.isdigit() else None
        steps = solve_eclat(txns, minsup, maxk)

    elif mode == 'fpgrowth':
        raw = request.form['fpgrowth_transactions']
        txns = {}
        for part in raw.split(';'):
            if not part.strip(): continue
            tid, items = part.split(':')
            txns[tid.strip()] = set(i.strip() for i in items.split(',') if i.strip())
        minsup = int(request.form['fpgrowth_minsup'])
        steps = solve_fpgrowth(txns, minsup)

    elif mode == 'cross_entropy':
        P = [float(x) for x in request.form['ce_p'].split(',') if x.strip()]
        Q = [float(x) for x in request.form['ce_q'].split(',') if x.strip()]
        steps = solve_cross_entropy(P, Q)

    elif mode == 'normalization':
        data   = [float(x) for x in request.form['norm_data'].split(',')
                  if x.strip()]
        method = request.form['norm_method']
        steps  = solve_normalization(data, method)

    elif mode == 'emd':
        P = [float(x) for x in request.form['emd_p'].split(',') if x.strip()]
        Q = [float(x) for x in request.form['emd_q'].split(',') if x.strip()]
        steps = solve_emd(P, Q)

    elif mode == 'tfidf':
        # documents come from the hidden field we populate in JS
        docs_raw = request.form['tfidf_docs']
        docs     = [d.strip() for d in docs_raw.split(';') if d.strip()]

        # queries come as "term,docIdx;term2,docIdx2"
        raw_queries = request.form['tfidf_queries']
        queries = []
        for q in raw_queries.split(';'):
            if not q.strip():
                continue
            term, idx = q.rsplit(',', 1)
            queries.append((term.strip(), int(idx)))

        steps = solve_tfidf(docs, queries)
    
    elif mode == 'bow':
        docs_raw = request.form['bow_docs']
        docs     = [d.strip() for d in docs_raw.split(';') if d.strip()]
        new_doc  = request.form['bow_newdoc'].strip() or None
        steps    = solve_bow(docs, new_doc)

    elif mode == 'metrics':
        y_true = [t.strip() for t in request.form['metrics_true'].split(',')
                  if t.strip()]
        y_pred = [p.strip() for p in request.form['metrics_pred'].split(',')
                  if p.strip()]
        steps  = solve_metrics(y_true, y_pred)
    else:
        return redirect(url_for('index'))

    return render_template('result.html', mode=mode.upper(), steps=steps, img=img)

if __name__ == '__main__':
    app.run()