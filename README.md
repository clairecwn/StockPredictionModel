# Stock reaction and drift model

Predicts how a stock's abnormal return, volatility and trading volume evolve in the days and weeks
after company-specific "noise": leadership changes, legal and regulatory news, accounting issues,
activist or short-seller campaigns, operational problems, policy and political news, and other
corporate events. The end goal is a live tracker that scores new events for a watchlist of stocks.

No data is stored in this repository. Licensed data (WRDS) is downloaded by each user into `data/`,
which is git-ignored. API keys live in `.env` (git-ignored; see `.env.example`).

## Pipeline
| Step | Script | Output |
|---|---|---|
| 1. Download WRDS data by hand | see `docs/WRDS_GUIDE.md` | `data/raw/wrds/*` |
| 2. Build event table | `python src/s02_events.py` | `data/processed/events.parquet` |
| 3. Train and validate | `python src/s03_train.py` | `results/report.md`, `models/` |
| 4. Live tracker | `python src/s04_live.py` (`python src/s04_live.py --ticker NVDA --date 2026-11-09 --cls leadership`) | `results/live/` |

Research code only. Not financial advice.


## Live board

<!-- LIVE-BOARD:START -->
_Updated 2026-11-16 11:33 UTC. Research only, not financial advice._

| Stock | Price | Event (day 0) | Day | Signal | Expected further dip | 1-in-10 worst point (20 days) | Risk vs normal | Recommended share if buying |
|---|---|---|---|---|---|---|---|---|
| NVDA | 162.85 | price/volume shock, no matched news (2026-11-12) | 1 | BUY ⚠️ | 5.3% | -14.5% | 0.97x | 15% |
| TSLA | 390.22 | price/volume shock, no matched news (2026-11-04) | 7 | WAIT ⚠️ | 4.5% | -13.4% | 0.91x | 25% |
| META | 641.69 | quiet: no event in the last 15 trading days | | | | | | |
| BABA | 94.38 | price/volume shock, no matched news (2026-11-05) | 6 | BUY ⚠️ | 4.5% | -16.5% | 1.09x | 8% |
| BA | 171.50 | earnings guidance (2026-10-27) | 13 | WAIT | 2.3% | -10.4% | 0.95x | 25% |
| INTC | 59.66 | earnings guidance (2026-10-29) | 11 | BUY ⚠️ | 5.7% | -21.9% | 0.94x | 18% |
| AAPL | 356.27 | earnings guidance (2026-11-02) | 9 | WAIT | 2.4% | -5.8% | 1.03x | 11% |
| GRAB | 3.24 | price/volume shock, no matched news (2026-11-05) | 6 | WAIT ⚠️ | 4.5% | -18.7% | 0.88x | 6% |
| SE | 75.36 | earnings guidance (2026-11-10) | 3 | WAIT ⚠️ | 5.1% | -14.5% | 1.07x | 17% |
<!-- LIVE-BOARD:END -->
