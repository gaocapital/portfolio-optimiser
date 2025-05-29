Here's a comprehensive `README.md` for your **Portfolio Optimisation** project:

---

# 📈 Portfolio Optimisation for Royce Lim

**Basket-aware + Drawdown-Constrained + Monte Carlo Portfolio Optimisation**

This project performs advanced portfolio optimisation tailored for Royce Lim. It includes features such as multi-basket-aware asset allocation, drawdown constraints, multistart and Monte Carlo-based optimisation, and flexible date exclusions.

---

## 🚀 Features

### ✅ Core Capabilities

* **Fair-share basket attribution**
  Allocates weights fairly across overlapping baskets, excluding the Benchmark from dilution.

* **Drawdown constraints**
  Optionally apply a hard limit and/or soft penalty to maximum portfolio drawdown.

* **Flexible exclusion ranges**
  Remove specific date ranges (e.g., crisis months) from the optimisation process.

* **Multistart + Monte Carlo optimiser**
  Improves solution robustness via multiple randomised restarts and multiple independent runs.

* **Comprehensive output**
  Includes weights, basket allocation summaries, alpha/beta vs. benchmark, Sharpe ratio, max drawdown, and equity curve plot.

---

## 🧮 Optimisation Logic

The optimiser:

* Maximises **Sharpe ratio** or **annual return** (configurable).
* Applies penalties for:

  * Excessive correlation with the benchmark (`SPY`)
  * Concentrated asset weights
  * Excessive basket concentration
  * Excess drawdown (if soft penalty enabled)

All objectives and penalties are weighted with tunable lambda and k parameters.

---

## 🧾 Baskets

Defined in the code as:

```python
BASKETS = {
    "Benchmark":   ["SPY"],
    "Commodities": ["Startar", "Svelland", "Geosol"],
    "Arbitrage":   ["LCAO", "Aravali AFO", "India Strategy", "Brahman Kova"],
    "Macro":       ["AAAP MAC", "Brahman Kova"],
    "Equities":    ["SPY", "Lim Advisors"],
}
```

> **Note:** Asset weights are split fairly across multiple non-Benchmark baskets. "SPY" is used only as the benchmark and excluded from basket weight splitting.

---

## 🗂️ Input Data

Place your monthly returns CSV in the following location:

```
DATA/Portfolio_Optimisation_Royce(Raw).csv
```

The file should contain percentage returns per asset, with a `Date` column. Example format:

| Date   | SPY  | AAAP MAC | LCAO | ... |
| ------ | ---- | -------- | ---- | --- |
| Jan-19 | 2.3% | 1.1%     | 1.8% | ... |
| Feb-19 | 1.2% | 0.9%     | 2.0% | ... |
| ...    | ...  | ...      | ...  | ... |

> ⚠️ You can **update this file with new data** at any time. The optimiser will automatically ingest the updated content.

---

## 🔧 Usage

### CLI Execution

Run the optimiser with default settings:

```bash
python3 optimise.py
```

### Customisation Options

Use command-line arguments to customise behaviour:

```bash
python3 optimise.py \
    --pri_order sharpe_first \
    --target_return 0.2 \
    --target_sharpe 4.0 \
    --exclude 2020-03-01:2020-04-30 \
    --monte 50 \
    --n_restarts 10
```

### All Available Options

| Argument          | Description                                    | Default                                      |
| ----------------- | ---------------------------------------------- | -------------------------------------------- |
| `--file`          | CSV path for monthly returns                   | `DATA/Portfolio_Optimisation_Royce(Raw).csv` |
| `--start_date`    | Start of date range                            | `2019-01-01`                                 |
| `--end_date`      | End of date range                              | `2025-01-01`                                 |
| `--exclude`       | Date ranges to exclude (can repeat)            | none                                         |
| `--pri_order`     | Primary goal: `sharpe_first` or `return_first` | `sharpe_first`                               |
| `--target_return` | Return hurdle                                  | `0.20`                                       |
| `--target_sharpe` | Sharpe hurdle                                  | `4.0`                                        |
| `--max_dd`        | Max drawdown cap (positive)                    | `0.025`                                      |
| `--no_dd_limit`   | Disable hard drawdown limit                    | disabled                                     |
| `--no_dd_penalty` | Disable soft drawdown penalty                  | disabled                                     |
| `--n_restarts`    | SLSQP restart attempts                         | `10`                                         |
| `--seed`          | RNG seed                                       | `420`                                        |
| `--monte`         | Monte Carlo runs                               | `50`                                         |

---

## 📊 Output

For each run, the script prints:

* Optimised asset weights
* Basket allocations (with fair-share logic)
* Key statistics:

  * Annual return, volatility
  * Sharpe ratio
  * Alpha, beta, correlation vs. SPY
  * Max drawdown
* Equity curve plot

For Monte Carlo mode (`--monte > 0`), average results are reported.

---

## 🧪 Dependencies

Ensure the following Python packages are installed:

```bash
pip install numpy pandas scipy statsmodels matplotlib
```

---

## 📌 Notes

* Assets listed in `DROP_COLS` (e.g., `"Geosol", "SPY"`) are excluded from investable universe.
* Optimisation handles missing or unparseable data robustly.
* Extendable to include new constraints or custom performance metrics.

---

## 📤 Future Improvements

* Add automatic benchmark comparison
* Store and compare Monte Carlo distributions
* Export equity curves to CSV/PDF

---

## 🧑‍💻 Author

*Developed for GAO Capital by Royce Lim

---

Let me know if you want a version of this in `.docx`, HTML, or if you'd like it embedded with badge support or visuals.
