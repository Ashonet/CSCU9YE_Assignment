# Runs the experiments and makes the stats + figures, usage: python run.py [experiments|analysis|all]
import json
import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
import mkp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
INST1, INST2 = os.path.join(HERE, "multi_resource_n28_m2.txt"), os.path.join(HERE, "multi_resource_n90_m5.txt")
SEED, M, WORKERS, B1, B2 = 2026, 30, 8, 10_000, 20_000
A, B, C, D, B4 = "GA-Repair(LP)", "GA-AdaptivePenalty", "GA-Repair(Random)", "GA-StaticPenalty", "GA-AdaptivePenalty-4xBudget"

# every experiment as (results file, algorithm, instance, {label: (handler, extra kwargs, budget)})
EXPERIMENTS = [
    ("part1_hc", "hc", INST1, {f"HC-{n}": (k, {}, B1) for n, k in [("Repair(LP)", "repair"), ("Repair(Random)", "repair-random"),
                                                                    ("StaticPenalty", "static-penalty"), ("DeathPenalty", "death")]}),
    ("part1_patience", "hc", INST1, {f"patience={p}": ("repair-random", {"patience": p}, B1) for p in [50, 150, 300, 1000, 10_000]}),
    ("part2_ga", "ga", INST2, {A: ("repair", {}, B2), B: ("adaptive-penalty", {}, B2), C: ("repair-random", {}, B2),
                               D: ("static-penalty", {}, B2), B4: ("adaptive-penalty", {}, 4 * B2)}),
    ("part2_hc", "hc", INST2, {"HC-Repair(LP)": ("repair", {}, B2), "HC-StaticPenalty": ("static-penalty", {}, B2)}),
]


# one run; run r of every variant uses the same seed so they all start from the same population
def job(args):
    algo, path, label, kind, kw, budget, run = args
    inst = mkp.Instance(path)
    handler = mkp.make_handler(inst, kind, np.random.default_rng([SEED, run, 1]))
    res = (mkp.hill_climber if algo == "hc" else mkp.ga)(inst, handler, budget, np.random.default_rng([SEED, run]), **kw)
    return dict(res, label=label, run=run)


def run_experiments():
    os.makedirs(RES, exist_ok=True)
    with Pool(WORKERS) as pool:
        for name, algo, path, cfg in EXPERIMENTS:
            t = time.perf_counter()
            out = pool.map(job, [(algo, path, lab, k, kw, bud, r) for lab, (k, kw, bud) in cfg.items() for r in range(M)], chunksize=1)
            json.dump(out, open(os.path.join(RES, name + ".json"), "w"))
            print(f"[{name}] {len(out)} runs in {time.perf_counter() - t:.1f}s")
            for lab, rs in group(out).items():
                print(f"   {lab:30s} mean={np.mean(col(rs, 'best_profit')):10.1f} hits={sum(r['hit_optimum'] for r in rs)}/{len(rs)}")


def group(records):
    g = {}
    for r in records:
        g.setdefault(r["label"], []).append(r)
    return g


def col(runs, key):
    return np.array([r[key] for r in runs], dtype=float)


# evals it took to hit the optimum, inf if the run never got there
def evals_to_opt(runs):
    return np.array([r["best_eval"] if r["hit_optimum"] else np.inf for r in runs])


def pairwise(g, labels, key):
    out = [dict(mkp.compare(col(g[a], key), col(g[b], key)), a=a, b=b) for a, b in combinations(labels, 2)]
    for r, ph in zip(out, mkp.holm([r["p"] for r in out])):
        r["p_holm"] = ph
    return out


def describe(g):
    rows = {}
    for lab, rs in g.items():
        e = evals_to_opt(rs)
        rows[lab] = dict(profit=mkp.summary(col(rs, "best_profit")), gap=mkp.summary(col(rs, "gap")), hits=int(np.isfinite(e).sum()),
                         evals_to_opt_hit_median=float(np.median(e[np.isfinite(e)])) if np.isfinite(e).any() else None,
                         evals_to_best=mkp.summary(col(rs, "best_eval")), time=mkp.summary(col(rs, "time")),
                         us_per_eval=float(np.mean(col(rs, "time") / col(rs, "evals")) * 1e6),
                         infeasible_rate=float(np.mean(col(rs, "infeasible_rate"))))
        rows[lab].update({k: float(np.mean(col(rs, k))) for k in ("final_feasible_fraction", "final_diversity", "duplicates", "restarts", "items_dropped") if k in rs[0]})
    return rows


# plot styling, grey is for the extra comparison lines
COL, GREY, INK, INK2 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def optimum_line(ax, opt, label=True):
    ax.axhline(opt, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    if label:
        ax.annotate(f"proven optimum {opt:,.0f}", xy=(0, opt), xycoords=("axes fraction", "data"), xytext=(4, 4),
                    textcoords="offset points", color=INK2, fontsize=8)


# median best-so-far with an IQR band, x is evals by default or seconds if x="tcurve"
def band(ax, runs, c, label, x="evals", grid=None):
    Y = np.array([r["curve"] for r in runs])
    if x == "evals":
        grid = np.arange(1, Y.shape[1] + 1) * runs[0]["checkpoint"]
    else:
        Y = np.array([np.interp(grid, r["tcurve"], r["curve"], left=np.nan) for r in runs])
    q1, med, q3 = np.nanpercentile(Y, [25, 50, 75], axis=0)
    ax.fill_between(grid, q1, q3, color=c, alpha=0.18, linewidth=0)
    ax.plot(grid, med, color=c, label=label)
    ax.set_xscale("log")


def convergence(g, labels, names, opt, ylim, title, fname):
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for lab, name, c in zip(labels, names, COL):
        band(ax, g[lab], c, name)
    optimum_line(ax, opt)
    ax.set(ylim=ylim, xlabel="fitness evaluations (log scale)", ylabel="best feasible utility\n(median, IQR band)", title=title)
    ax.legend(loc="lower right")
    save(fig, fname)


def boxplot(g, labels, names, colours, opt, title, fname):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    data = [col(g[l], "best_profit") for l in labels]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color=INK, linewidth=1.5))
    jitter = np.random.default_rng(0)
    for i, (d, c, patch) in enumerate(zip(data, colours, bp["boxes"]), 1):
        patch.set(facecolor=c, alpha=0.45, edgecolor=c)
        ax.scatter(i + jitter.uniform(-0.12, 0.12, d.size), d, s=10, color=c, edgecolor="white", linewidth=0.4, zorder=3)
    optimum_line(ax, opt, False)
    ax.set_xticks(range(1, len(names) + 1), names)
    ax.set(ylabel="final best feasible utility", title=title)
    save(fig, fname)


def part1(out):
    g, gp, opt = group(load("part1_hc")), group(load("part1_patience")), mkp.OPTIMA["multi_resource_n28_m2"]
    labels = list(g)
    lo = {l: np.concatenate([r["local_optima"] for r in g[l]]) for l in labels}
    out["part1"] = dict(summary=describe(g), patience=describe(gp), pairwise_profit=pairwise(g, labels, "best_profit"),
                        kruskal=dict(zip(("H", "p"), map(float, stats.kruskal(*[col(g[l], "best_profit") for l in labels])))),
                        repair_lp_vs_random_evals_to_best=mkp.compare(col(g[labels[0]], "best_eval"), col(g[labels[1]], "best_eval")),
                        local_optima_per_restart={l: dict(mean=float(v.mean()), frac_opt=float(np.mean(v >= opt))) for l, v in lo.items()})
    convergence(g, labels, labels, opt, (95000, 133500), "Instance 1 (n=28, m=2): hill-climber convergence, 30 runs per strategy", "p1_convergence.png")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
    pv = [int(l.split("=")[1]) for l in gp]
    axs[0].plot(pv, [np.mean(col(gp[l], "hit_optimum")) for l in gp], color=COL[0], marker="o", markersize=5)
    axs[1].plot(pv, [np.mean(col(gp[l], "restarts")) for l in gp], color=COL[0], marker="o", markersize=5)
    axs[0].set(xscale="log", ylim=(0, 1.05), xlabel="patience (log scale)", ylabel="success rate (optimum reached)")
    axs[1].set(xscale="log", yscale="log", xlabel="patience (log scale)", ylabel="restarts per run (mean)")
    fig.suptitle("Restart-patience sensitivity: HC-Repair(Random), Instance 1, 30 runs per setting", fontsize=9)
    fig.tight_layout()
    save(fig, "p1_patience.png")


def part2(out):
    g, gh, opt = group(load("part2_ga")), group(load("part2_hc")), mkp.OPTIMA["multi_resource_n90_m5"]
    main = [A, B, C, D]
    curves = {l: np.array([r["curve"] for r in g[l]]) for l in (A, B, C)}
    anytime = []
    for budget in (250, 500, 1000, 2000, 5000, 10000, 20000):
        v = {l: c[:, budget // 50 - 1] for l, c in curves.items()}
        anytime.append(dict(budget=budget, **{f"{k}_median": float(np.median(v[l])) for k, l in zip("ABC", (A, B, C))},
                            **{f"{k}_hits": int((v[l] >= opt).sum()) for k, l in zip("ABC", (A, B, C))},
                            AvsB=mkp.compare(v[A], v[B]), AvsC=mkp.compare(v[A], v[C]), CvsB=mkp.compare(v[C], v[B])))
    for key in ("AvsB", "AvsC", "CvsB"):
        for a, ph in zip(anytime, mkp.holm([a[key]["p"] for a in anytime])):
            a[key]["p_holm"] = ph
    tA = float(np.median(col(g[A], "time")))
    at_tA = np.array([r["curve"][max(int(np.searchsorted(r["tcurve"], tA, side="right")) - 1, 0)] for r in g[B4]])
    hits = lambda l: int(sum(r["hit_optimum"] for r in g[l]))  # noqa: E731
    out["part2"] = dict(summary=describe({l: g[l] for l in main + [B4]}), hc_summary=describe(gh), primary=dict(
        profit=mkp.compare(col(g[A], "best_profit"), col(g[B], "best_profit"), "greater"),
        hits_fisher=float(stats.fisher_exact([[hits(A), M - hits(A)], [hits(B), M - hits(B)]])[1]),
        evals_to_best=mkp.compare(col(g[A], "best_eval"), col(g[B], "best_eval"), "less"),
        time=mkp.compare(col(g[A], "time"), col(g[B], "time")), anytime=anytime,
        equal_time=dict(time_budget_s=tA, B_profit_at_time=mkp.summary(at_tA), B_hits_at_time=int((at_tA >= opt).sum()),
                        test=mkp.compare(col(g[A], "best_profit"), at_tA, "greater"),
                        B4_full=mkp.compare(col(g[A], "best_profit"), col(g[B4], "best_profit"), "greater"),
                        B4_full_fisher=float(stats.fisher_exact([[hits(A), M - hits(A)], [hits(B4), M - hits(B4)]])[1]))),
        ablation=dict(repair_order_evals_to_opt=mkp.compare(evals_to_opt(g[A]), evals_to_opt(g[C]), "less"),
                      penalty_adaptive_vs_static=mkp.compare(col(g[B], "best_profit"), col(g[D], "best_profit")),
                      pairwise_profit=pairwise(g, main, "best_profit")),
        ga_vs_hc=dict(repair_evals_to_best=mkp.compare(col(g[A], "best_eval"), col(gh["HC-Repair(LP)"], "best_eval")),
                      penalty_profit=mkp.compare(col(g[D], "best_profit"), col(gh["HC-StaticPenalty"], "best_profit"))))

    boxplot(g, main + [B4], ["A: GA-\nRepair(LP)", "B: GA-Adaptive\nPenalty", "C: GA-\nRepair(Random)", "D: GA-Static\nPenalty",
                             "B with 4x budget"], COL + [GREY], opt, "Instance 2 (n=90, m=5): final utility, 30 runs per variant", "p2_boxplot.png")
    convergence(g, main, ["A: GA-Repair(LP)", "B: GA-AdaptivePenalty", "C: GA-Repair(Random)", "D: GA-StaticPenalty"], opt,
                (10200, 11240), "Instance 2: convergence under an equal evaluation budget (20,000)", "p2_convergence.png")

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    grid = np.geomspace(0.01, np.median(col(g[B4], "time")), 300)
    band(ax, g[A], COL[0], "A: GA-Repair(LP), 20k evaluations", "tcurve", grid)
    band(ax, g[B4], COL[1], "B: GA-AdaptivePenalty, 80k evaluations", "tcurve", grid)
    optimum_line(ax, opt, False)
    ax.axvline(tA, color=INK2, linewidth=0.8)
    ax.annotate(f"median run time of A ({tA:.1f} s)", xy=(tA, 10900), xytext=(-4, 0), textcoords="offset points", ha="right", color=INK2, fontsize=8)
    ax.set(ylim=(10600, 11230), xlabel="wall-clock time in seconds (log scale)", ylabel="best feasible utility\n(median, IQR band)",
           title="Instance 2: equal wall-clock comparison")
    ax.legend(loc="lower right")
    save(fig, "p2_time.png")

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for name, runs, c in [("A: GA-Repair(LP)", g[A], COL[0]), ("HC-Repair(LP)", gh["HC-Repair(LP)"], GREY),
                          ("C: GA-Repair(Random)", g[C], COL[2]), ("B: GA-AdaptivePenalty, 80k", g[B4], COL[1])]:
        e = np.sort(evals_to_opt(runs))
        e = e[np.isfinite(e)]
        ax.step(np.r_[1, e, 4 * B2], np.r_[0, np.arange(1, e.size + 1), e.size] / len(runs), where="post", color=c, label=f"{name} ({e.size}/{len(runs)} runs)")
    ax.set(xscale="log", ylim=(-0.02, 1.02), xlabel="fitness evaluations (log scale)", ylabel="fraction of runs that\nhave reached the optimum",
           title="Instance 2: run-length distribution (evaluations to reach the optimum)")
    ax.legend(loc="upper left")
    save(fig, "p2_rld.png")

    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9))
    L = np.array([r["lam_trace"] for r in g[B]])
    axs[0].plot(L[:8].T, color=COL[1], alpha=0.3, linewidth=1)
    axs[0].plot(np.exp(np.median(np.log(L), axis=0)), color=COL[1], label="median of 30 runs")
    axs[0].set(yscale="log", xlabel="generation (100 evaluations)", ylabel="penalty multiplier λ (log scale)")
    axs[0].set_title("Adaptive λ in variant B (8 runs + median)", fontsize=9)
    axs[0].legend(loc="upper right")
    us = [out["part2"]["summary"][l]["us_per_eval"] for l in main]
    for i, (bar, v) in enumerate(zip(axs[1].bar(range(4), us, color=COL, width=0.6), us)):
        axs[1].annotate(f"{v:.0f} µs", xy=(i, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    axs[1].set_xticks(range(4), list("ABCD"))
    axs[1].set_ylabel("wall-clock per evaluation (µs)")
    axs[1].set_title("Cost of one evaluation, by variant", fontsize=9)
    fig.tight_layout()
    save(fig, "p2_penalty_cost.png")


def load(name):
    return json.load(open(os.path.join(RES, name + ".json")))


def run_analysis():
    os.makedirs(FIG, exist_ok=True)
    out = {}
    part1(out)
    part2(out)
    json.dump(out, open(os.path.join(RES, "stats.json"), "w"), indent=1)
    print("wrote results/stats.json and figures/")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("experiments", "all"):
        run_experiments()
    if what in ("analysis", "all"):
        run_analysis()
