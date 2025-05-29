#!/usr/bin/env python3
"""
Portfolio Optimisation for Royce Lim – basket-aware + drawdown cap + date-exclusion
+ multistart + Monte-Carlo
==================================================================================

Changes in this version
-----------------------
1. **Fair-share basket attribution** – if a fund sits in several *non-Benchmark*
   baskets its weight is split equally across those baskets.  The "Benchmark"
   basket is ignored in the split, so a fund present in *Benchmark* + one other
   basket contributes its *full* weight to that other basket.

2. Both the on-screen basket summaries *and* the (optional) basket-concentration
   penalty inside the objective now use this logic.

No other behaviour has been changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt

# -----------------------------  GLOBAL DEFAULTS -----------------------------
PRI_ORDER        = "sharpe_first"  # "return_first" | "sharpe_first"
TARGET_RETURN    = 0.20            # annual return hurdle
TARGET_SHARPE    = 4.0             # annual Sharpe hurdle
RF_ANNUAL        = 0.03            # annual risk-free rate

MAX_DRAWDOWN     = 0.025           # drawdown cap (positive number)
ENFORCE_DD_LIMIT = False            # apply as hard constraint

PENALISE_CORR      = True
PENALISE_CONC      = True
PENALISE_BASK_CONC = False
PENALISE_DD        = False
LAMB_CORR,  K_CORR       = 0.05, 2.5
LAMB_CONC,  K_CONC       = 0.05, 2.5
LAMB_BASK_CONC, K_BASK_CONC = 0.05, 2.5
LAMB_DD, K_DD            = 0.10, 5.0

DROP_COLS: List[str] = ["Geosol", "SPY"]
START_DATE = "2019-01-01"
END_DATE   = "2025-01-01"

# Exclude ranges (start, end) inclusive – strings or ISO-format dates ----------
EXCLUDE_RANGES: List[Tuple[str, str]] = []  # e.g. [("2020-03-01", "2020-04-30")]

# ---- MULTISTART SETTINGS ----
N_MONTE = 50
N_RESTARTS_DEFAULT = 10  # number of random restarts for SLSQP
SEED_DEFAULT = 420       # RNG seed for reproducibility

# -----------------------------  BASKET DEFINITIONS ---------------------------
BASKETS: Dict[str, List[str]] = {
    "Benchmark":   ["SPY"],
    "Commodities": ["Startar", "Svelland", "Geosol"],
    "Arbitrage":   ["LCAO", "Aravali AFO", "India Strategy", "Brahman Kova"],
    "Macro":       ["AAAP MAC", "Brahman Kova"],
    "Equities":    ["SPY", "Lim Advisors"],
}

# ---------------------------------------------------------------------------
#  Pre-compute the number of non-Benchmark baskets each asset belongs to.
# ---------------------------------------------------------------------------
ASSET_BASKET_COUNT: dict[str, int] = defaultdict(int)
for basket, members in BASKETS.items():
    if basket == "Benchmark":
        continue
    for asset in members:
        ASSET_BASKET_COUNT[asset] += 1


# ------------------------------  HELPERS -------------------------------------
def basket_allocations_fair(weights: pd.Series) -> Dict[str, float]:
    """
    Return basket allocations with overlapped assets split equally across the
    *non-Benchmark* baskets they belong to.

    If an asset is in Benchmark + one other basket, count its full weight
    toward that other basket.
    """
    alloc = {b: 0.0 for b in BASKETS if b != "Benchmark"}

    for asset, w in weights.items():
        k = ASSET_BASKET_COUNT.get(asset, 1)  # default 1 ⇒ put 100 % in one bucket
        share = w / k
        for b, members in BASKETS.items():
            if b == "Benchmark":
                continue
            if asset in members:
                alloc[b] += share
    return alloc


def parse_returns(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, index_col="Date")
    df = (
        raw.replace({r"%$": ""}, regex=True)
           .apply(pd.to_numeric, errors="coerce")
           .div(100)
    )
    dt1 = pd.to_datetime(df.index, format="%b-%y", errors="coerce")
    dt2 = pd.to_datetime(df.index, errors="coerce")
    idx = dt1.where(~dt1.isna(), dt2) + MonthEnd(0)
    df = df.loc[~idx.isna()]
    df.index = idx.dropna()
    if df.empty:
        raise ValueError("CSV contained no parsable data rows.")
    return df.sort_index()


def annualise(mean_m: float, std_m: float):
    return (1 + mean_m) ** 12 - 1, std_m * np.sqrt(12)


def max_drawdown(ret: np.ndarray) -> float:
    nav = np.cumprod(1 + ret)
    return ((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)).min()


def regression_alpha_beta(y: np.ndarray, x: np.ndarray):
    model = sm.OLS(y, sm.add_constant(x)).fit()
    return model.params[0], model.params[1]


# -----------------------------  OPTIMISER ------------------------------------
def optimise_portfolio(
    returns: pd.DataFrame,
    *,
    pri_order: str,
    target_return: float,
    target_sharpe: float,
    rf_annual: float,
    max_drawdown_cap: float,
    enforce_dd_limit: bool,
    penalise_dd: bool,
    penalise_corr: bool,
    penalise_conc: bool,
    penalise_basket_conc: bool,
    drop_cols: Optional[List[str]],
    exclude_ranges: List[Tuple[str, str]],
    n_restarts: int = N_RESTARTS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> Dict:

    # --- slice & exclude -----------------------------------------------------
    returns = returns.loc[START_DATE:END_DATE]
    for rng in exclude_ranges:
        beg, end = map(pd.to_datetime, rng)
        returns = returns.drop(returns.loc[beg:end].index)
    if returns.empty:
        raise ValueError("No data left after exclusions.")

    returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if "SPY" not in returns.columns:
        raise ValueError("SPY column missing in dataset.")

    spy = returns["SPY"].to_numpy()
    if drop_cols:
        returns = returns.drop(columns=drop_cols, errors="ignore")
    assets = returns.columns
    if assets.empty:
        raise ValueError("No investable assets remain after DROP_COLS.")

    r = returns[assets].to_numpy()

    # --- objective -----------------------------------------------------------
    def obj(w: np.ndarray) -> float:
        port = r @ w
        mean_m, std_m = port.mean(), port.std()
        ann_ret, ann_std = annualise(mean_m, std_m)
        sharpe = (ann_ret - rf_annual) / ann_std if ann_std else -np.inf
        dd_mag = abs(max_drawdown(port))

        core = (
            -ann_ret if pri_order == "return_first"  and ann_ret < target_return else
            -sharpe  if pri_order == "sharpe_first" and sharpe  < target_sharpe  else
            -sharpe  if pri_order == "return_first" else -ann_ret
        )

        pen = 0.0
        if penalise_corr and spy.size > 1:
            pen += LAMB_CORR * np.exp(K_CORR * abs(np.corrcoef(port, spy)[0, 1]))
        if penalise_conc:
            pen += LAMB_CONC * np.exp(K_CONC * w.max())
        if penalise_basket_conc:
            fair_allocs = basket_allocations_fair(pd.Series(w, index=assets))
            pen += LAMB_BASK_CONC * np.exp(K_BASK_CONC * max(fair_allocs.values()))
        if penalise_dd and dd_mag > max_drawdown_cap:
            pen += LAMB_DD * np.exp(K_DD * (dd_mag - max_drawdown_cap))
        return core + pen

    bounds = [(0, 1)] * len(assets)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if enforce_dd_limit:
        cons.append({"type": "ineq",
                     "fun": lambda w: max_drawdown_cap - abs(max_drawdown(r @ w))})

    rng = np.random.default_rng(seed)
    best_res = None
    for _ in range(max(1, n_restarts)):
        w0 = rng.dirichlet(np.ones(len(assets)))
        res = opt.minimize(obj, w0, method="SLSQP",
                           bounds=bounds, constraints=cons)
        if not res.success:
            continue
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    if best_res is None or not best_res.success:
        raise RuntimeError("Optimisation failed in all restarts.")

    w_opt = best_res.x
    port = r @ w_opt
    mean_m, std_m = port.mean(), port.std()
    ann_ret, ann_std = annualise(mean_m, std_m)
    sharpe = (ann_ret - rf_annual) / ann_std if ann_std else np.nan
    rf_m = (1 + rf_annual) ** (1/12) - 1
    alpha_m, beta = regression_alpha_beta(port - rf_m, spy - rf_m)

    return {
        "weights": pd.Series(w_opt, index=assets),
        "series":  pd.Series(port,  index=returns.index),
        "annual_return": ann_ret,
        "annual_std":   ann_std,
        "sharpe":       sharpe,
        "alpha":        (1 + alpha_m) ** 12 - 1,
        "beta":         beta,
        "corr": float(np.corrcoef(port, spy)[0, 1]),
        "max_dd": max_drawdown(port),
    }


# ------------------------------  CLI & MAIN ----------------------------------
def default_csv_path() -> Path:
    try:
        return (Path(__file__).resolve().parent.parent /
                "DATA" / "Portfolio_Optimisation_Royce(Raw).csv")
    except NameError:
        return Path.cwd() / "DATA" / "Portfolio_Optimisation_Royce(Raw).csv"


def print_stats(res: Dict, label: str = "Annualised performance"):
    print(f"\n{label}:")
    print(f"  Return            : {res['annual_return']:.2%}")
    print(f"  Stdev             : {res['annual_std']:.2%}")
    print(f"  Sharpe            : {res['sharpe']:.2f}")
    print(f"  Alpha (vs SPY)    : {res['alpha']:.2%}")
    print(f"  Beta  (vs SPY)    : {res['beta']:.3f}")
    print(f"  Corr  (vs SPY)    : {res['corr']:.3f}")
    print(f"  Max drawdown      : {res['max_dd']:.2%}")


def aggregate_results(results: List[Dict]) -> Dict:
    """Average weights and diagnostics across Monte-Carlo runs."""
    keys = ["annual_return", "annual_std", "sharpe",
            "alpha", "beta", "corr", "max_dd"]
    agg = {}
    agg["weights"] = pd.concat(
        [r["weights"] for r in results], axis=1).mean(axis=1)
    for k in keys:
        agg[k] = np.mean([r[k] for r in results])
    return agg


def main():
    global N_MONTE

    ap = argparse.ArgumentParser(
        "Sharpe/Return optimiser with DD control, multistart & Monte-Carlo")
    ap.add_argument("--file", default=str(default_csv_path()),
                    help="CSV file of monthly returns")
    ap.add_argument("--start_date", default=START_DATE)
    ap.add_argument("--end_date",   default=END_DATE)
    ap.add_argument("--exclude", action="append", default=[],
                    metavar="YYYY-MM-DD:YYYY-MM-DD",
                    help="date range to exclude (inclusive); can repeat")
    ap.add_argument("--pri_order", choices=["return_first", "sharpe_first"],
                    default=PRI_ORDER)
    ap.add_argument("--target_return", type=float, default=TARGET_RETURN)
    ap.add_argument("--target_sharpe", type=float, default=TARGET_SHARPE)
    ap.add_argument("--max_dd", type=float, default=MAX_DRAWDOWN,
                    help="Drawdown cap (positive number)")
    ap.add_argument("--no_dd_limit", action="store_true",
                    help="Disable hard drawdown constraint")
    ap.add_argument("--no_dd_penalty", action="store_true",
                    help="Disable soft drawdown penalty")
    ap.add_argument("--n_restarts", type=int, default=N_RESTARTS_DEFAULT,
                    help="Number of SLSQP restarts")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT,
                    help="Base RNG seed for reproducibility")
    ap.add_argument("--monte", type=int, default=N_MONTE,
                    help="Number of Monte-Carlo optimiser runs (0 = single run)")
    args = ap.parse_args()

    exclude_ranges = EXCLUDE_RANGES.copy()
    for rng in args.exclude:
        try:
            beg, end = rng.split(":")
            exclude_ranges.append((beg, end))
        except ValueError:
            raise SystemExit(f"Bad --exclude format: {rng}. "
                             "Use YYYY-MM-DD:YYYY-MM-DD")

    df = parse_returns(Path(args.file))

    # ---------------------------------------------------------------------
    # Single run ----------------------------------------------------------
    # ---------------------------------------------------------------------
    if args.monte <= 0:
        res = optimise_portfolio(
            df,
            pri_order=args.pri_order,
            target_return=args.target_return,
            target_sharpe=args.target_sharpe,
            rf_annual=RF_ANNUAL,
            max_drawdown_cap=args.max_dd,
            enforce_dd_limit=not args.no_dd_limit,
            penalise_dd=not args.no_dd_penalty,
            penalise_corr=PENALISE_CORR,
            penalise_conc=PENALISE_CONC,
            penalise_basket_conc=PENALISE_BASK_CONC,
            drop_cols=DROP_COLS,
            exclude_ranges=exclude_ranges,
            n_restarts=args.n_restarts,
            seed=args.seed,
        )

        # ---- weights ----------------------------------------------------
        print("\nWeights:")
        for asset, w in res["weights"].items():
            print(f"  {asset:15s}: {w:.2%}")

        # ---- baskets ----------------------------------------------------
        print("\nBasket allocation (fair-share):")
        for b, alloc in basket_allocations_fair(res["weights"]).items():
            print(f"  {b:12s}: {alloc:.2%}")

        print_stats(res)

        # ---- equity curve ----------------------------------------------
        equity = (1 + res["series"]).cumprod()
        plt.figure(figsize=(10, 5))
        plt.plot(equity.index, equity, label="Optimised Portfolio")
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Net Asset Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # Monte-Carlo ---------------------------------------------------------
    # ---------------------------------------------------------------------
    else:
        mc_results = []
        for i in range(args.monte):
            res_i = optimise_portfolio(
                df,
                pri_order=args.pri_order,
                target_return=args.target_return,
                target_sharpe=args.target_sharpe,
                rf_annual=RF_ANNUAL,
                max_drawdown_cap=args.max_dd,
                enforce_dd_limit=not args.no_dd_limit,
                penalise_dd=not args.no_dd_penalty,
                penalise_corr=PENALISE_CORR,
                penalise_conc=PENALISE_CONC,
                penalise_basket_conc=PENALISE_BASK_CONC,
                drop_cols=DROP_COLS,
                exclude_ranges=exclude_ranges,
                n_restarts=args.n_restarts,
                seed=args.seed + i + 1,  # different seed each run
            )
            mc_results.append(res_i)

        avg_res = aggregate_results(mc_results)

        # ---- averaged weights ------------------------------------------
        print(f"\n===== Monte-Carlo averages over {args.monte} runs =====")
        print("\nAverage Weights:")
        for asset, w in avg_res["weights"].items():
            print(f"  {asset:15s}: {w:.2%}")

        print("\nAverage Basket allocation (fair-share):")
        for b, alloc in basket_allocations_fair(avg_res["weights"]).items():
            print(f"  {b:12s}: {alloc:.2%}")

        print_stats(avg_res, label="Average annualised performance")
        # (no equity curve for averages, but easy to add if desired)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
