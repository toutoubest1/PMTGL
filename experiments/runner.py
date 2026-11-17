"""Fit and evaluate the Monte Carlo regularization paths.

Select F1 and Frobenius penalties on an independent tuning sample. Compute
eBIC and path AUC on the evaluation sample."""
import os, sys, time, zlib, itertools, argparse
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pmtgl.methods import METHODS, LAMS
from pmtgl.sim import make_precision, generate, graph_metrics, roc_auc, ebic


def fit_path(name, X, Om, lams=LAMS):
    t0 = time.perf_counter()
    path = METHODS[name](X, lams)
    t = time.perf_counter() - t0
    mets = [graph_metrics(T, Om) for (T, _, _) in path]
    eb = [ebic(S, Tf, X.shape[0], 0.5) for (_, Tf, S) in path]
    bic = [ebic(S, Tf, X.shape[0], 0.0) for (_, Tf, S) in path]
    return mets, (eb, bic), t


def evaluate(name, Xtu, Xte, Om, lams=LAMS):
    mtu, _, _ = fit_path(name, Xtu, Om, lams)
    mte, (eb, bic), t = fit_path(name, Xte, Om, lams)
    f1tu = np.array([m["f1"] for m in mtu])
    frtu = np.array([m["frob"] for m in mtu])
    iF = int(np.nanargmax(f1tu)) if np.isfinite(f1tu).any() else 0
    iR = int(np.nanargmin(frtu)) if np.isfinite(frtu).any() else 0
    iE = int(np.argmin(eb))
    iB = int(np.argmin(bic))
    g = lambda i, k: mte[i][k]
    row = dict(method=name, time=t,
               auc=roc_auc(np.array([m["tpr"] for m in mte]), np.array([m["fpr"] for m in mte])),
               f1=g(iF, "f1"), prec=g(iF, "prec"), rec=g(iF, "rec"), fdr=g(iF, "fdr"), ne=g(iF, "ne"),
               frob=g(iR, "frob"), spec=g(iR, "spec"),
               eb_f1=g(iE, "f1"), eb_prec=g(iE, "prec"), eb_rec=g(iE, "rec"), eb_fdr=g(iE, "fdr"),
               eb_frob=g(iE, "frob"),
               bic_f1=g(iB, "f1"), bic_prec=g(iB, "prec"), bic_rec=g(iB, "rec"), bic_fdr=g(iB, "fdr"),
               bic_frob=g(iB, "frob"),
               or_f1=np.nanmax([m["f1"] for m in mte]), or_frob=np.nanmin([m["frob"] for m in mte]),
               lam_f1=lams[iF], lam_fro=lams[iR], lam_eb=lams[iE])
    return row


def one_rep(job):
    (n, p, graph, scen, eps, mag), rep, methods, tag = job
    key = f"{tag}-{n}-{p}-{graph}-{scen}-{eps}-{mag}"
    rng = np.random.default_rng([zlib.crc32(key.encode()), rep])
    Om, Sig = make_precision(p, graph, rng=rng)
    Xtu = generate(n, Sig, scen, rng, eps=eps, mag=mag)
    Xte = generate(n, Sig, scen, rng, eps=eps, mag=mag)
    rows = []
    for m in methods:
        try:
            r = evaluate(m, Xtu, Xte, Om)
        except Exception as e:
            r = dict(method=m, error=repr(e))
        r.update(n=n, p=p, graph=graph, scenario=scen, eps=eps, mag=mag, rep=rep)
        rows.append(r)
    return rows


def run(settings, reps, methods, out, tag="main", procs=2):
    """reps: int (replicates 0..reps-1) or an iterable of replicate indices."""
    rr = range(reps) if isinstance(reps, int) else list(reps)
    jobs = [(s, r, methods, tag) for s in settings for r in rr]
    na = lambda v: "NA" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):g}"
    key = lambda n, p, g, s, e, m, r: f"{int(n)}|{int(p)}|{g}|{s}|{na(e)}|{na(m)}|{int(r)}"
    done = set()
    if os.path.exists(out):
        old = pd.read_csv(out)
        done = {key(*t) for t in zip(old.n, old.p, old.graph, old.scenario, old.eps, old.mag, old.rep)}
    jobs = [j for j in jobs if key(*j[0], j[1]) not in done]
    print(f"{len(jobs)} jobs to run -> {out}", flush=True)
    with Pool(procs) as pool:
        for rows in pool.imap_unordered(one_rep, jobs):
            df = pd.DataFrame(rows)
            df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
