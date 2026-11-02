# Hill climber, GA, constraint handlers and stats for the workload allocation problem (0-1 multidimensional knapsack)
import os
import time
import numpy as np
from scipy import stats
from scipy.optimize import linprog
from workload_data_loader import load_workload_instance

# optimal values from scipy's milp solver, only used to work out the gap afterwards
OPTIMA = {"multi_resource_n28_m2": 130883.0, "multi_resource_n90_m5": 11191.0}


class Instance:
    def __init__(self, path):
        self.n, self.m, self.p, self.R, self.b = load_workload_instance(path)
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.optimum = OPTIMA.get(self.name)
        self.cols = self.R.T.copy()  # row j holds the demands of item j

    def feasible(self, x):
        return bool(np.all(self.R @ x <= self.b))

    # LP relaxation bound and its dual prices, which say how scarce each resource is
    def lp(self):
        res = linprog(-self.p, A_ub=self.R, b_ub=self.b, bounds=(0, 1), method="highs")
        return -res.fun, np.maximum(-res.ineqlin.marginals, 0.0)


# counts evaluations (every candidate costs one, feasible or not) and logs the best feasible so far
class Tracker:
    def __init__(self, inst, budget, every=50):
        self.inst, self.budget, self.every, self.t0 = inst, budget, every, time.perf_counter()
        self.evals = self.best_eval = self.n_infeasible = 0
        self.best, self.best_x, self.curve, self.tcurve = 0.0, np.zeros(inst.n), [], []

    @property
    def done(self):
        return self.evals >= self.budget

    def log(self, x, profit, feasible):
        self.evals += 1
        if feasible and profit > self.best:
            self.best, self.best_x, self.best_eval = profit, x.copy(), self.evals
        if self.evals % self.every == 0:
            self.curve.append(self.best)
            self.tcurve.append(time.perf_counter() - self.t0)

    def result(self, **extra):
        opt = self.inst.optimum
        assert self.inst.feasible(self.best_x), "reported solution must be feasible"
        return dict(best_profit=self.best, best_eval=self.best_eval, evals=self.evals, checkpoint=self.every,
                    infeasible_rate=self.n_infeasible / self.evals, time=time.perf_counter() - self.t0,
                    curve=self.curve, tcurve=self.tcurve, hit_optimum=self.best >= opt,
                    gap=100.0 * (opt - self.best) / opt, **extra)


# drop the least efficient items until it fits, add the best ones that still fit, and keep the result
class Repair:
    adaptive = False

    def __init__(self, inst, rng=None, random_order=False):
        self.inst, self.rng, self.dropped = inst, rng, 0
        self.name = "repair-random" if random_order else "repair"
        if random_order:
            self.asc = None
        else:  # profit per unit of the resources that actually bind (Chu & Beasley 1998)
            denom = inst.lp()[1] @ inst.R
            with np.errstate(divide="ignore"):
                u = np.where(denom > 0, inst.p / np.where(denom > 0, denom, 1.0), np.inf)
            self.asc = np.argsort(u, kind="stable")
            self.desc = self.asc[::-1]

    def __call__(self, x, tr):
        i = self.inst
        asc, desc = (self.rng.permutation(i.n), self.rng.permutation(i.n)) if self.asc is None else (self.asc, self.desc)
        load = i.R @ x
        if np.any(load > i.b):
            tr.n_infeasible += 1
            sel = asc[x[asc] > 0]  # selected items, worst first
            left = load[:, None] - np.cumsum(i.R[:, sel], axis=1)  # load after each removal
            k = int(np.argmax(np.all(left <= i.b[:, None], axis=0)))  # first point where it fits
            x[sel[:k + 1]] = 0
            load, self.dropped = left[:, k], self.dropped + k + 1
        slack = i.b - load
        # anything that doesn't fit now never will, so only loop over items that do
        for j in desc[(x[desc] == 0) & np.all(i.cols[desc] <= slack, axis=1)]:
            if np.all(i.cols[j] <= slack):
                x[j] = 1
                slack = slack - i.cols[j]
        f = float(i.p @ x)
        tr.log(x, f, True)
        return x, f


# profit minus lam*rho*overflow, rho = best profit per unit of any resource so lam=1 is never worth breaking (Olsen 1994)
class Penalty:
    def __init__(self, inst, adaptive=False, k=3, up=2.0, down=1.5, lam_min=1e-3, lam_max=1e3):
        self.inst, self.adaptive, self.k, self.up, self.down, self.lims = inst, adaptive, k, up, down, (lam_min, lam_max)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rho = float(np.where(inst.R > 0, inst.p / np.where(inst.R > 0, inst.R, 1.0), 0.0).max())
        self.lam, self.history, self.lam_trace = 1.0, [], [1.0]
        self.name = "adaptive-penalty" if adaptive else "static-penalty"

    def rescore(self, x):
        over = np.maximum(0.0, self.inst.R @ x - self.inst.b).sum()
        return float(self.inst.p @ x), over, float(self.inst.p @ x) - self.lam * self.rho * over

    def __call__(self, x, tr):
        profit, over, f = self.rescore(x)
        tr.n_infeasible += over > 0
        tr.log(x, profit, over == 0)
        return x, f

    # raise lam if the best has been infeasible for k gens, lower it if it's been feasible for k (Hadj-Alouane & Bean 1997)
    def end_generation(self, best_feasible):
        old = self.lam
        self.history.append(best_feasible)
        if len(self.history) >= self.k and len(set(self.history[-self.k:])) == 1:
            self.lam = min(self.lims[1], self.lam * self.up) if not best_feasible else max(self.lims[0], self.lam / self.down)
            self.history.clear()
        self.lam_trace.append(self.lam)
        return self.lam != old


# anything infeasible gets -inf so it's never accepted
class Death:
    name, adaptive = "death", False

    def __init__(self, inst):
        self.inst = inst

    def __call__(self, x, tr):
        ok, profit = self.inst.feasible(x), float(self.inst.p @ x)
        tr.n_infeasible += not ok
        tr.log(x, profit, ok)
        return x, profit if ok else -np.inf


def make_handler(inst, kind, rng=None):
    return {"repair": lambda: Repair(inst), "repair-random": lambda: Repair(inst, rng, True),
            "static-penalty": lambda: Penalty(inst), "adaptive-penalty": lambda: Penalty(inst, True),
            "death": lambda: Death(inst)}[kind]()


# go through the items in a random order and add each one if it still fits
def random_feasible(inst, rng):
    x, slack = np.zeros(inst.n), inst.b.astype(float)
    for j in rng.permutation(inst.n):
        if np.all(inst.cols[j] <= slack):
            x[j] = 1.0
            slack = slack - inst.cols[j]
    return x


def flip(x, rng, k=1):
    y = x.copy()
    idx = rng.choice(x.size, size=k, replace=False)
    y[idx] = 1.0 - y[idx]
    return y


# hill climber move: half the time swap one item out for another (slides along the capacity limit), else flip one bit
def mixed_move(x, rng, p_swap=0.5):
    if rng.random() >= p_swap:
        return flip(x, rng)
    ones, zeros = np.flatnonzero(x), np.flatnonzero(x == 0)
    if ones.size == 0 or zeros.size == 0:
        return flip(x, rng)
    y = x.copy()
    y[rng.choice(ones)], y[rng.choice(zeros)] = 0.0, 1.0
    return y


# try one random neighbour at a time, restart after `patience` evals with no improvement
def hill_climber(inst, handler, budget, rng, patience=300):
    tr, restarts, local_optima = Tracker(inst, budget), 0, []
    while not tr.done:
        x, f = handler(random_feasible(inst, rng), tr)
        stall = 0
        while not tr.done and stall < patience:
            y, fy = handler(mixed_move(x, rng), tr)
            if fy > f:
                x, f, stall = y, fy, 0
            else:
                x, stall = (y if fy == f else x), stall + 1  # sideways moves get across flat bits
        restarts += 1
        if inst.feasible(x):
            local_optima.append(float(inst.p @ x))
    return tr.result(restarts=restarts, local_optima=local_optima)


# steady-state GA (Chu & Beasley 1998): 2 tournaments, uniform crossover, flip 2 bits, handle constraints, replace the worst
def ga(inst, handler, budget, rng, pop_size=100, mut_bits=2):
    tr, pop, fit, keys, duplicates = Tracker(inst, budget), [], [], set(), 0
    while len(pop) < pop_size and not tr.done:
        x, f = handler(random_feasible(inst, rng), tr)
        if x.tobytes() not in keys:
            keys.add(x.tobytes())
            pop.append(x)
            fit.append(f)
    fit, next_gen = np.array(fit, dtype=float), tr.evals + pop_size
    while not tr.done:
        a, b = (pop[int(c[np.argmax(fit[c])])] for c in (rng.integers(len(fit), size=2), rng.integers(len(fit), size=2)))
        child, fc = handler(flip(np.where(rng.random(inst.n) < 0.5, a, b), rng, mut_bits), tr)
        w = int(np.argmin(fit))
        if child.tobytes() in keys:
            duplicates += 1
        elif fc >= fit[w]:
            keys.discard(pop[w].tobytes())
            keys.add(child.tobytes())
            pop[w], fit[w] = child, fc
        if tr.evals >= next_gen:  # every pop_size evals counts as a generation for the adaptive penalty
            next_gen += pop_size
            if handler.adaptive and handler.end_generation(inst.feasible(pop[int(np.argmax(fit))])):
                fit = np.array([handler.rescore(x)[2] for x in pop])  # lam changed so old scores are out of date
    P = np.array(pop)
    extra = dict(duplicates=duplicates, final_feasible_fraction=float(np.mean([inst.feasible(x) for x in P])),
                 final_diversity=float(np.abs(P - P[int(np.argmax(fit))]).sum(axis=1).mean()))
    if hasattr(handler, "lam_trace"):
        extra["lam_trace"] = handler.lam_trace
    if hasattr(handler, "dropped"):
        extra["items_dropped"] = handler.dropped
    return tr.result(**extra)


# Vargha-Delaney A12, the chance a run of x beats a run of y (0.5 means no difference)
def a12(x, y):
    x, y = np.asarray(x, float)[:, None], np.asarray(y, float)[None, :]
    return float(((x > y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


# Mann-Whitney U plus effect size, handles the case where both groups are all the same value
def compare(x, y, alternative="two-sided"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    same = np.all(x == x[0]) and np.all(y == x[0])
    u, p = (x.size * y.size / 2.0, 1.0) if same else stats.mannwhitneyu(x, y, alternative=alternative)
    a = a12(x, y)
    d = abs(a - 0.5)
    mag = "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"
    return dict(U=float(u), p=float(p), A12=a, magnitude=mag)


# Holm correction for when several tests are run together
def holm(p):
    p, adj, running = np.asarray(p, float), np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def summary(v):
    v = np.asarray(v, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)), best=float(v.max()), worst=float(v.min()),
                median=float(med), q1=float(q1), q3=float(q3))
