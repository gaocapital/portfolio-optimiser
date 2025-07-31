#!/usr/bin/env python3
"""
Portfolio Optimisation for Royce Lim – basket-aware + drawdown cap + date-exclusion
+ multistart + Monte-Carlo
==================================================================================
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
PRI_ORDER        = "return_first"  # "return_first" | "sharpe_first"
TARGET_RETURN    = 0.20            # annual return hurdle
TARGET_SHARPE    = 2.5             # annual Sharpe hurdle
RF_ANNUAL        = 0.03            # annual risk-free rate

MAX_DRAWDOWN     = 0.025           # drawdown cap (positive number)
ENFORCE_DD_LIMIT = False            # apply as hard constraint

PENALISE_CORR      = True
PENALISE_CONC      = True
PENALISE_BASK_CONC = True
PENALISE_DD        = False
LAMB_CORR,  K_CORR       = 0.05, 2.5
LAMB_CONC,  K_CONC       = 0.1, 5.0
LAMB_BASK_CONC, K_BASK_CONC = 0.05, 2.5
LAMB_DD, K_DD            = 0.10, 5.0

INCLUDE_SPY = False
DROP_COLS: List[str] = ["Geosol", "AAAP MAC", "RV", "Aravali AFO", "Brahman Kova"]
FIXED_ALLOCS = {
    "Startar" : 0.1,
    "Svelland" : 0.05,
    "LCAO" : 0.15,
    "India Strategy" : 0.15,
    "Lim Advisors" : 0.15,
    "Caxton": 0.05,
    "ADAPT": 0.05,
    "Wizard Global": 0.05,
    "Wizard Neutral": 0.1,
    "Schonfield": 0.15,
}
FIXED_BASKET_ALLOCS = {
    # "Commodities": 0.20,        
    # "Equities":    0.15,
    # "Macro":       0.15
}

START_DATE = "2020-01-01"
END_DATE   = "2025-06-01"

# Exclude ranges (start, end) inclusive – strings or ISO-format dates ----------
EXCLUDE_RANGES: List[Tuple[str, str]] = []  # e.g. [("2020-03-01", "2020-04-30")]

# ---- MULTISTART SETTINGS ----
N_MONTE = 0
N_RESTARTS_DEFAULT = 10  # number of random restarts for SLSQP
SEED_DEFAULT = 420       # RNG seed for reproducibility

# -----------------------------  BASKET DEFINITIONS ---------------------------
BASKETS: Dict[str, List[str]] = {
    "Benchmark":   ["SPY"],
    "Commodities": ["Startar", "Svelland", "Geosol"],
    "Arbitrage":   ["LCAO", "Aravali AFO", "India Strategy", "Brahman Kova", "Wizard Neutral", "Schonfield"],
    "Macro":       ["AAAP MAC", "RV", "Caxton", "ADAPT"],
    "Equities":    ["SPY", "Lim Advisors", "Wizard Global"],
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

    # --- 1) strip %-signs, coerce to numeric, convert to decimal
    df = (
        raw.replace({r"%$": ""}, regex=True)
           .apply(pd.to_numeric, errors="coerce")
           .div(100)                     # 45.6 % → 0.456
    )

    # --- 2) robust date parsing -----------------------------------------
    iso   = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    month = pd.to_datetime(df.index, format="%b-%y",   errors="coerce")

    # prefer ISO; fall back to Mon-YY where ISO failed
    idx = iso.where(~iso.isna(), month)

    # drop rows whose dates we still couldn’t parse
    keep = ~idx.isna()
    df   = df.loc[keep]
    idx  = idx[keep] + MonthEnd(0)       # roll to month-end if needed
    df.index = idx

    if df.empty:
        raise ValueError("CSV contained no parsable data rows.")

    return df.sort_index()
def annualise(mean_m: float, std_m: float):
    return (1 + mean_m) ** 12 - 1, std_m * np.sqrt(12)


def max_drawdown(ret: np.ndarray) -> float:
    nav = np.cumprod(1 + ret)
    return ((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)).min()

def basket_return_series(
    weights: pd.Series,            # optimised weights (index = assets)
    returns: pd.DataFrame          # monthly return matrix
) -> dict[str, pd.Series]:
    """
    Build a monthly return series for every non-Benchmark basket, using the
    same “fair-share” split that basket_allocations_fair() applies to weights.
    """
    series = {}
    for basket, members in BASKETS.items():
        if basket == "Benchmark":
            continue

        # share of each asset’s weight that belongs to this basket
        w_share = {
            m: weights.get(m, 0.0) / ASSET_BASKET_COUNT.get(m, 1)
            for m in members if m in weights
        }
        
        basket_w = sum(w_share.values())

        if not w_share:        # basket empty after DROP_COLS etc.
            series[basket] = pd.Series(0.0, index=returns.index)
            continue

        basket_ret = (
            returns[list(w_share.keys())]
            .mul(pd.Series(w_share), axis=1)
            .sum(axis=1)
            / basket_w
        )
        series[basket] = basket_ret
    return series

def calendar_year_returns_annualised(series: pd.Series) -> pd.Series:
    """
    Return a Series of annual returns for every calendar year in *series*.
    - If the year has < 12 months of data (e.g. the current year), the
      partial-period return is geometric-annualised to a 12-month equivalent.
    """
    out = {}
    for year, grp in series.groupby(series.index.year):
        n_months = len(grp)
        r = (1 + grp).prod() - 1           # total return for the months we have
        if n_months < 12:                  # incomplete year → annualise
            r = (1 + r) ** (12 / n_months) - 1
        out[year] = r
    return pd.Series(out)

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
    fixed_alloc: Optional[Dict[str, float]] = None,
    include_spy: bool = False,                 
    n_restarts: int = N_RESTARTS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> Dict:
    """
    Run the multistart optimiser.

    Parameters
    ----------
    include_spy : bool, default False
        * False → SPY is kept only as benchmark (cannot be allocated).
        * True  → SPY may receive a weight (free or fixed).  Still used as
          benchmark for alpha/beta/corr diagnostics.
    All remaining parameters as before …
    """

    # ---------------- 1) Date window & explicit exclusions ------------------
    returns = returns.loc[START_DATE:END_DATE]
    for beg, end in exclude_ranges:
        returns = returns.drop(
            returns.loc[pd.to_datetime(beg): pd.to_datetime(end)].index
        )
    if returns.empty:
        raise ValueError("No data left after date filtering.")

    # Basic cleaning
    returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")

    # ---------------- 2) Safe DROP_COLS ------------------------------------
    drop_cols = drop_cols or []
    if "SPY" in drop_cols:
        drop_cols = [c for c in drop_cols if c != "SPY"]

    returns = returns.drop(columns=[c for c in drop_cols if c in returns.columns],
                           errors="ignore")

    if "SPY" not in returns.columns:
        raise ValueError("SPY column missing in dataset.")

    # ---------------- 3) Fixed allocations ---------------------------------
    fixed_alloc = {k.strip(): float(v) for k, v in (fixed_alloc or {}).items()}
    fixed_alloc = {k: v for k, v in fixed_alloc.items() if k in returns.columns}

    bad_fixed = [k for k in fixed_alloc if k not in returns.columns]
    if bad_fixed:
        raise KeyError(f"Fixed-allocation fund(s) not in data: {bad_fixed}")
    if any(w < 0 or w > 1 for w in fixed_alloc.values()):
        raise ValueError("Fixed weights must lie in [0, 1].")
    w_fixed_tot = sum(fixed_alloc.values())
    if w_fixed_tot > 1:
        raise ValueError("Sum of fixed weights exceeds 100 %.")
        
    fixed_basket_alloc = FIXED_BASKET_ALLOCS.copy()   # or pass as argument
    bad_baskets = [b for b in fixed_basket_alloc if b not in BASKETS]
    if bad_baskets:
        raise KeyError(f"Unknown basket(s) in FIXED_BASKET_ALLOCS: {bad_baskets}")
    if any(w < 0 or w > 1 for w in fixed_basket_alloc.values()):
        raise ValueError("Basket weights must lie in [0, 1].")
    basket_tot = sum(fixed_basket_alloc.values())
    if basket_tot > 1:
        raise ValueError("Sum of basket weights exceeds 100 %.")


    # ---------------- 4) Partition assets ----------------------------------
    if include_spy:
        all_assets = [c for c in returns.columns if c != "SPY"] + ["SPY"]
    else:
        all_assets = [c for c in returns.columns if c != "SPY"]

    fixed_assets = [a for a in fixed_alloc if a in all_assets]
    free_assets  = [a for a in all_assets if a not in fixed_assets]

    if not free_assets and w_fixed_tot < 1:
        raise ValueError("No free assets left to optimise after DROP_COLS.")

    r_free  = returns[free_assets].to_numpy()  if free_assets else None
    r_fixed = returns[fixed_assets].to_numpy() if fixed_assets else None
    spy     = returns["SPY"].to_numpy()

    port_fixed_leg = (
        r_fixed @ np.array([fixed_alloc[a] for a in fixed_assets])
        if fixed_assets else 0.0
    )

    # ---------------- 5) Objective -----------------------------------------
    def objective(w_free: np.ndarray) -> float:
        w_full = pd.Series(
            {**{a: w for a, w in zip(free_assets, w_free)}, **fixed_alloc}
        )[all_assets]

        port = port_fixed_leg + (r_free @ w_free if free_assets else 0.0)

        mean_m, std_m = port.mean(), port.std()
        ann_ret, ann_std = annualise(mean_m, std_m)
        sharpe = (mean_m * 12 - rf_annual) / ann_std if ann_std else -np.inf
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
            pen += LAMB_CONC * np.exp(K_CONC * w_full.max())
        if penalise_basket_conc:
            pen += LAMB_BASK_CONC * np.exp(
                K_BASK_CONC * max(basket_allocations_fair(w_full).values())
            )
        if penalise_dd and dd_mag > max_drawdown_cap:
            pen += LAMB_DD * np.exp(K_DD * (dd_mag - max_drawdown_cap))
        return core + pen

    # ---------------- 6) Constraints / bounds ------------------------------
    n_free = len(free_assets)
    bounds = [(0, 1)] * n_free
    cons = [
        {
            "type": "eq",
            "fun": lambda w: w.sum() - (1 - w_fixed_tot),
        }
    ]
    
    for basket, target in fixed_basket_alloc.items():
        members = [a for a in BASKETS[basket] if a in all_assets]   # ignore SPY if excluded
        idxs = [free_assets.index(a) for a in members if a in free_assets]
    
        def basket_fun_factory(idxs, target, members):
            def _fn(w_free):
                # sum of free + fixed parts for this basket minus target
                s_free  = w_free[idxs].sum() if idxs else 0.0
                s_fixed = sum(fixed_alloc.get(a, 0.0) for a in members)
                return s_free + s_fixed - target
            return _fn
    
        cons.append({"type": "eq", "fun": basket_fun_factory(idxs, target, members)})
    
    if enforce_dd_limit:

        def dd_constraint(w):
            return max_drawdown_cap - abs(
                max_drawdown(port_fixed_leg + (r_free @ w if free_assets else 0.0))
            )

        cons.append({"type": "ineq", "fun": dd_constraint})

    # ---------------- 7) Optimise ------------------------------------------
    if n_free == 0:                      # fully fixed portfolio
        w_free_opt = np.empty(0)
    else:
        rng = np.random.default_rng(seed)
        best_res = None
        for _ in range(max(1, n_restarts)):
            w0 = rng.dirichlet(np.ones(n_free)) * (1 - w_fixed_tot)
            res = opt.minimize(objective, w0, method="SLSQP",
                               bounds=bounds, constraints=cons)
            if res.success and (best_res is None or res.fun < best_res.fun):
                best_res = res
        if best_res is None:
            raise RuntimeError("Optimisation failed in all restarts.")
        w_free_opt = best_res.x

    # ---------------- 8) Assemble output -----------------------------------
    w_full_opt = pd.Series(
        {**{a: w for a, w in zip(free_assets, w_free_opt)}, **fixed_alloc}
    )[all_assets]

    port = port_fixed_leg + (r_free @ w_free_opt if free_assets else 0.0)
    mean_m, std_m = port.mean(), port.std()
    ann_ret, ann_std = annualise(mean_m, std_m)
    sharpe = (mean_m * 12 - rf_annual) / ann_std if ann_std else np.nan
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    alpha_m, beta = regression_alpha_beta(port - rf_m, spy - rf_m)

    return {
        "weights": w_full_opt,
        "series":  pd.Series(port, index=returns.index),
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
        return (
            Path(__file__).resolve().parent.parent
            / "DATA"
            / "Portfolio_Optimisation_Royce_Revised.csv"
        )
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


def calendar_ytd(series: pd.Series) -> pd.Series:
    """Cumulative return for each calendar year."""
    return series.groupby(series.index.year).apply(lambda x: (1 + x).prod() - 1)

def aggregate_results(results: List[Dict]) -> Dict:
    """
    Average weights and key diagnostics across many optimiser runs.

    Parameters
    ----------
    results : list of dict
        Each element is the dictionary returned by `optimise_portfolio`.

    Returns
    -------
    dict
        Same keys as one result, but with weights averaged column-wise and
        scalar metrics averaged arithmetically.
    """
    if not results:
        raise ValueError("No results supplied to aggregate_results().")

    scalar_keys = [
        "annual_return", "annual_std", "sharpe",
        "alpha", "beta", "corr", "max_dd"
    ]

    # Average weights
    weights_mat = pd.concat([r["weights"] for r in results], axis=1)
    avg_weights = weights_mat.mean(axis=1)

    # Average scalars
    avg_scalars = {k: np.mean([r[k] for r in results]) for k in scalar_keys}

    avg_res = {"weights": avg_weights}
    avg_res.update(avg_scalars)
    return avg_res


def main() -> None:
    global N_MONTE

    ap = argparse.ArgumentParser(
        "Sharpe/Return optimiser with DD control, multistart & Monte-Carlo"
    )
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

    # -------------------------------------------------------------------
    # Parse exclusions
    # -------------------------------------------------------------------
    exclude_ranges = EXCLUDE_RANGES.copy()
    for rng in args.exclude:
        try:
            beg, end = rng.split(":")
            exclude_ranges.append((beg, end))
        except ValueError:
            raise SystemExit(
                f"Bad --exclude format: {rng}. Use YYYY-MM-DD:YYYY-MM-DD"
            )

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    df = parse_returns(Path(args.file))

    # -------------------------------------------------------------------
    # SINGLE-RUN MODE
    # -------------------------------------------------------------------
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
            fixed_alloc=FIXED_ALLOCS,
            include_spy=INCLUDE_SPY,
            n_restarts=args.n_restarts,
            seed=args.seed,
        )

        # ---- weights --------------------------------------------------
        print("\nWeights:")
        for asset, w in res["weights"].items():
            print(f"  {asset:15s}: {w:.2%}")

        # ---- baskets --------------------------------------------------
        print("\nBasket allocation (fair-share):")
        for b, alloc in basket_allocations_fair(res["weights"]).items():
            print(f"  {b:12s}: {alloc:.2%}")

        print_stats(res)

        # ---- equity curves & YTD table -------------------------------
        spy_series = df["SPY"].loc[res["series"].index]
        eq_port = (1 + res["series"]).cumprod()
        eq_spy  = (1 + spy_series).cumprod()

        plt.figure(figsize=(10, 5))
        plt.plot(eq_port.index, eq_port, label="Optimised Portfolio")
        plt.plot(eq_spy.index,  eq_spy,  linestyle="--", label="SPY Benchmark")
        plt.legend()
        plt.title("Equity Curve: Portfolio vs SPY")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

        ytd_tbl = pd.DataFrame({
            "Portfolio": calendar_ytd(res["series"]),
            "SPY":       calendar_ytd(spy_series),
        })
        ytd_tbl["Outperformance"] = ytd_tbl["Portfolio"] - ytd_tbl["SPY"]

        print("\nCalendar-Year YTD Returns:")
        print(ytd_tbl.applymap(lambda x: f"{x:.2%}"))
        
        # ---- Basket returns: YTD grid + Avg (annualised) ------------------------
        basket_series = basket_return_series(res["weights"],
                                             df.loc[res["series"].index])
        
        # Y-to-D numbers (what you want to *see* per year)
        basket_ytd = pd.DataFrame({
            b: calendar_ytd(s) for b, s in basket_series.items()
        }).T.fillna(0.0)
        
        # Annualised numbers (used *only* for the average calculation)
        basket_ann = pd.DataFrame({
            b: calendar_year_returns_annualised(s) for b, s in basket_series.items()
        }).T
        
        basket_ytd["Avg"] = basket_ann.mean(axis=1)        # arithmetic mean
        
        print("\nCalendar-Year YTD Returns by Basket:")
        print(basket_ytd.applymap(lambda x: f"{x:.2%}"))

    # -------------------------------------------------------------------
    # MONTE-CARLO MODE
    # -------------------------------------------------------------------
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
                fixed_alloc=FIXED_ALLOCS,
                include_spy=INCLUDE_SPY,
                n_restarts=args.n_restarts,
                seed=args.seed + i + 1,
            )
            mc_results.append(res_i)

        avg_res = aggregate_results(mc_results)

        # ---- averaged weights ----------------------------------------
        print(f"\n===== Monte-Carlo averages over {args.monte} runs =====")
        print("\nAverage Weights:")
        for asset, w in avg_res["weights"].items():
            print(f"  {asset:15s}: {w:.2%}")

        print("\nAverage Basket allocation (fair-share):")
        for b, alloc in basket_allocations_fair(avg_res["weights"]).items():
            print(f"  {b:12s}: {alloc:.2%}")

        print_stats(avg_res, label="Average annualised performance")

       # ---- equity curve & YTD stats for FIRST run ------------------
        # (optional: you could average curves, but here we show run #0)
        first_run = mc_results[0]
        spy_series = df["SPY"].loc[first_run["series"].index]
        eq_port = (1 + first_run["series"]).cumprod()
        eq_spy  = (1 + spy_series).cumprod()
        
        plt.figure(figsize=(10, 5))
        plt.plot(eq_port.index, eq_port, label="Portfolio (run 1)")
        plt.plot(eq_spy.index,  eq_spy,  linestyle="--", label="SPY Benchmark")
        plt.legend()
        plt.title("Equity Curve (First MC Run) vs SPY")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()
        
        ytd_tbl = pd.DataFrame({
            "Portfolio": calendar_ytd(first_run["series"]),
            "SPY":       calendar_ytd(spy_series),
        })
        ytd_tbl["Outperformance"] = ytd_tbl["Portfolio"] - ytd_tbl["SPY"]
        
        print("\nCalendar-Year YTD Returns (First MC Run):")
        print(ytd_tbl.applymap(lambda x: f"{x:.2%}"))
        
        # ---- Basket returns for first MC run -----------------------------------
        basket_series = basket_return_series(first_run["weights"],
                                             df.loc[first_run["series"].index])
        
        basket_ytd = pd.DataFrame({
            b: calendar_ytd(s) for b, s in basket_series.items()
        }).T.fillna(0.0)
        
        basket_ann = pd.DataFrame({
            b: calendar_year_returns_annualised(s) for b, s in basket_series.items()
        }).T
        
        basket_ytd["Avg"] = basket_ann.mean(axis=1)
        
        print("\nCalendar-Year YTD Returns by Basket (First MC Run):")
        print(basket_ytd.applymap(lambda x: f"{x:.2%}"))
                
        # 1) Grab the aggregated portfolio monthly returns
        port_mtm = first_run["series"]
        
        # 2) Print the last 5 months
        print("\nMonthly Returns of Aggregated Portfolio (First MC Run):")
        print(port_mtm.tail().apply(lambda x: f"{x:.2%}"))
        
        # 3) Save the full series to CSV alongside your input data
        data_folder = Path(args.file).parent
        output_path = data_folder / "aggregated_portfolio_mtm_returns.csv"
        port_mtm.to_frame(name="Portfolio_MTM_Return").to_csv(output_path)
        print(f"\nSaved aggregated portfolio MTM returns to: {output_path}")



# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
