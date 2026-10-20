"""Baseline (random forest, basic structural features) vs improved (class-balanced,
CV-tuned LightGBM with neighbourhood-structure features) on SNAP email-Eu-core department labels."""
import gzip, io, json, sys, time
import numpy as np, pandas as pd, networkx as nx, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

SEED = 42
SEEDS = [42, 43, 44]  # model-randomness seeds (split is fixed at SEED)
EDGES_URL = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
LABELS_URL = "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"
BASE_COLS = ["degree", "in_degree", "out_degree", "clustering", "pagerank"]
EXT_COLS = ["nb_deg_mean", "nb_deg_max", "nb_deg_std", "nb_pr_mean", "nb_clu_mean", "reciprocity", "kcore"]


def fetch(url, tries=5):
    """GET a gzip file (timeout=60); retry with exponential backoff on 429/5xx/network errors."""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return gzip.decompress(r.content)
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code} for {url}")
            reason = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            reason = str(e)
        print(f"Download failed ({reason}); retrying in {2 ** i}s")
        time.sleep(2 ** i)
    raise RuntimeError(f"Giving up on {url}")


def evaluate(model, X, y):
    """Shared evaluation: macro-F1 (main), accuracy, weighted-F1 and log-loss on the held-out nodes."""
    pred, proba = model.predict(X), model.predict_proba(X)
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "weighted_f1": f1_score(y, pred, average="weighted"),
            "accuracy": accuracy_score(y, pred),
            "logloss": log_loss(y, proba, labels=model.classes_)}


def main():
    t0 = time.time()
    print("Downloading SNAP email-Eu-core graph and department labels...")
    edges = pd.read_csv(io.BytesIO(fetch(EDGES_URL)), sep=r"\s+", comment="#", header=None, names=["src", "dst"])
    labels = pd.read_csv(io.BytesIO(fetch(LABELS_URL)), sep=r"\s+", comment="#", header=None, names=["node", "label"])
    G = nx.DiGraph()
    G.add_nodes_from(labels.node.tolist())
    G.add_edges_from(map(tuple, edges.values.tolist()))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges, "
          f"{labels.label.nunique()} departments")

    # Structural features are label-free and computed on the whole graph (transductive, no test labels used)
    UG = G.to_undirected()
    deg, clu, core = dict(UG.degree()), nx.clustering(UG), nx.core_number(UG)
    pr = nx.pagerank(G)
    rows = []
    for v in labels.node.tolist():
        nb = list(UG.neighbors(v))
        nd = np.array([deg[u] for u in nb] or [0.0])
        npr = np.array([pr[u] for u in nb] or [0.0])
        ncl = np.array([clu[u] for u in nb] or [0.0])
        out, inn = set(G.successors(v)), set(G.predecessors(v))
        rows.append([deg[v], G.in_degree(v), G.out_degree(v), clu[v], pr[v],
                     nd.mean(), nd.max(), nd.std(), npr.mean(), ncl.mean(),
                     len(out & inn) / max(len(out | inn), 1), core[v]])  # reciprocity = fraction of reciprocated ties
    X_all = pd.DataFrame(rows, columns=BASE_COLS + EXT_COLS)
    y = labels.label.to_numpy()
    print("Features computed.")

    # Single stratified 70/30 node-level split with fixed seed.
    # Departments with only one node cannot be split across train and test, so those
    # nodes are always placed in the training set; the rest is split stratified 70/30.
    counts = pd.Series(y).value_counts()
    singleton_labels = counts[counts < 2].index
    is_single = np.isin(y, singleton_labels)
    all_idx = np.arange(len(y))
    rest_idx = all_idx[~is_single]
    single_idx = all_idx[is_single]
    if len(single_idx):
        print(f"{len(single_idx)} node(s) in singleton department(s) kept in training set")
    idx_rest_tr, idx_te = train_test_split(rest_idx, test_size=0.3, stratify=y[rest_idx], random_state=SEED)
    idx_tr = np.concatenate([idx_rest_tr, single_idx])
    ytr, yte = y[idx_tr], y[idx_te]
    Xb_tr, Xb_te = X_all[BASE_COLS].iloc[idx_tr], X_all[BASE_COLS].iloc[idx_te]
    Xe_tr, Xe_te = X_all.iloc[idx_tr], X_all.iloc[idx_te]
    print(f"Split: {len(idx_tr)} train / {len(idx_te)} test nodes")

    # Tune the improved model ONLY by 5-fold CV on training nodes (macro-F1)
    print("Tuning class-balanced LightGBM by 5-fold CV on training nodes...")
    grid = {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 300], "min_child_samples": [5, 20]}
    gs = GridSearchCV(LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1),
                      grid, scoring="f1_macro", cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                      n_jobs=-1, refit=False)
    gs.fit(Xe_tr, ytr)
    best = gs.best_params_
    print(f"Best CV params: {best} (CV macro-F1 {gs.best_score_:.4f})")

    # Both models fitted on the same training nodes, scored once on the same test nodes, per seed
    res = {"baseline": [], "improved": []}
    for s in SEEDS:
        print(f"Seed {s}: fitting random forest (baseline) and LightGBM (improved)...")
        rf = RandomForestClassifier(n_estimators=500, random_state=s, n_jobs=-1).fit(Xb_tr, ytr)
        lgb = LGBMClassifier(class_weight="balanced", random_state=s, verbose=-1, **best).fit(Xe_tr, ytr)
        res["baseline"].append(evaluate(rf, Xb_te, yte))
        res["improved"].append(evaluate(lgb, Xe_te, yte))

    def agg(name, key):
        v = np.array([r[key] for r in res[name]], dtype=float)
        return float(v.mean()), float(v.std())

    m = {"cv_best_macro_f1": float(gs.best_score_)}
    for name in ["baseline", "improved"]:
        mf, sf = agg(name, "macro_f1")
        m[f"{name}_score"], m[f"{name}_score_std"] = mf, sf
        m[f"{name}_weighted_f1"], _ = agg(name, "weighted_f1")
        m[f"{name}_accuracy"], _ = agg(name, "accuracy")
        m[f"{name}_logloss"], _ = agg(name, "logloss")
    m["improvement"] = m["improved_score"] - m["baseline_score"]  # higher macro-F1 is better
    for k, v in m.items():
        if not np.isfinite(v):
            sys.exit(f"Non-finite metric {k}; cannot produce meaningful results")
    print(f"Macro-F1  baseline {m['baseline_score']:.4f} +/- {m['baseline_score_std']:.4f} | "
          f"improved {m['improved_score']:.4f} +/- {m['improved_score_std']:.4f}")

    # Optional chart
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Random forest\n(baseline)", "CV-tuned LightGBM\n(improved)"],
           [m["baseline_score"], m["improved_score"]],
           yerr=[m["baseline_score_std"], m["improved_score_std"]], capsize=6, color=["grey", "steelblue"])
    ax.set_ylabel("Test macro-F1 (mean +/- sd over seeds)")
    ax.set_title("email-Eu-core department labels")
    fig.tight_layout()
    fig.savefig("figures/macro_f1_comparison.png", dpi=120)
    plt.close(fig)

    out = {
        "question": "Can a class-balanced, CV-tuned LightGBM model with neighbourhood structure features beat a standard random forest on structural features at predicting labelled community membership of nodes in a public communication network?",
        "method": "Stratified 70/30 node split (seed 42) on SNAP email-Eu-core with department labels as community stand-in; nodes in single-node departments (which cannot be split) are kept in training. Baseline: random forest on degree, in/out-degree, clustering, PageRank. Improved: class-balanced LightGBM tuned by 5-fold CV on training nodes, with added neighbour-degree statistics, k-core and reciprocity features. Test set used once per model; models run with seeds 42, 43, 44.",
        "dataset": {
            "name": "SNAP email-Eu-core (department labels)",
            "source_url": EDGES_URL + " ; " + LABELS_URL,
            "n_rows": int(len(y)),
            "n_train": int(len(idx_tr)),
            "n_test": int(len(idx_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 on held-out test nodes",
        "higher_is_better": True,
        "baseline": "Standard random forest (500 trees, default settings, no class weighting) on basic per-node structural features: degree, in-degree, out-degree, clustering coefficient, PageRank.",
        "improved": "Class-balanced LightGBM with hyper-parameters chosen by 5-fold CV macro-F1 on training nodes, using the same basic features plus neighbour-degree mean/max/std, neighbour PageRank and clustering means, reciprocity fraction and k-core number.",
        "metrics": m,
        "notes": "Stand-in for the paper's darknet forum data, which is not public; email department labels are not cybercrime communities. Only 3 model seeds vary; the split is fixed, so the std reflects model randomness only. Nodes belonging to departments with a single member cannot be stratified, so they are always placed in training (the test split is stratified over all other nodes). Structural features are computed on the full graph without labels (transductive, label-free). Tuning used a small grid; test scores were not used for selection. Results on ~1,000 nodes with 42 classes are noisy, especially for rare departments.",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
