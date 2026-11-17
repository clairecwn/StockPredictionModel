# Downloading the WRDS data by hand

Everything goes into `data/raw/wrds/` inside this repo. That folder is git-ignored, so WRDS data
never reaches GitHub. Use the exact file names below; the code looks for them.

| File to save in `data/raw/wrds/` | WRDS form | Covers |
|---|---|---|
| `execucomp_ceo.csv` | ExecuComp annual compensation (CEO rows) | CEO tenure, age, ownership |
| `ccm_link.csv` | CRSP/Compustat Merged link history | gvkey <-> permno |
| `crsp_daily.csv.gz` | CRSP > Annual Update > Stock - Version 2 > Daily Stock File | prices, returns, volume, market index (to 2025-12-31) |
| `compustat_ids.csv.gz` | Compustat > North America > Fundamentals Annual (gvkey, cik, cusip, tic, conm, fyear) | links SEC CIK / CUSIP to gvkey |
| `aa_officer_changes.csv.gz` | Audit Analytics > Director and Officer Changes | leadership events |
| `aa_legal_case.csv.gz` | Audit Analytics > Legal Case | lawsuit details |
| `aa_legal_party.csv.gz` | Audit Analytics > Company Legal Party | which company is sued |
| `aa_activism.csv.gz` | Audit Analytics > Shareholder Activism | activist campaigns |
| `aa_restatements.csv.gz` | Audit Analytics > Financial Restatements | accounting problems |
| `ibes_recs.csv.gz` | LSEG I/B/E/S > Recommendations > Detail (US file) | analyst up/downgrades |

Not in the NTU subscription (checked Nov 2026): Capital IQ Key Developments, CRSP monthly-update, Audit Analytics
Cybersecurity / AAER / Bankruptcy. Politics, public sentiment and live news come from free sources instead.

The two upload lists are already made for you in `data/wrds_upload/`:
- `permnos.txt`: 4,611 CRSP stock IDs (every firm ever in the S&P 1500 / ExecuComp universe)
- `gvkeys.txt`: the same 4,246 firms as Compustat IDs

General rules for every WRDS query form:
1. Log in at https://wrds-www.wharton.upenn.edu
2. On the form, use the **upload a file** option for company codes, not "search entire database".
3. Output format: **comma-delimited text (.csv)**. Compression: **gzip (.csv.gz)** if offered, otherwise none.
4. Submit, then wait on the "Your query is running" page (or come back later via the Query History link).
5. Download, rename the file to the name in the table, and move it into `data/raw/wrds/`.

---

## Step 3: CRSP daily stock file (Version 2)

The old CRSP format ends on 31 Dec 2024, so use **Version 2** (also called CIZ).

1. Get Data -> CRSP -> **Stock - Version 2** -> **Daily Stock File** (on some menus: "CRSP Stock / Security Files (v2) -> Daily Stock File").
2. Step 1, date range: **1999-01-01** to the **latest available date**.
3. Step 2, company codes: choose **PERMNO**, choose **Upload a plain text file**, pick `data/wrds_upload/permnos.txt`.
4. Step 3, variables: tick exactly these (names may show with a "Dly" prefix):
   `PERMNO`, `DlyCalDt`, `DlyRet`, `DlyPrc`, `DlyVol`, `DlyCap`, `ShrOut`, `DlyBid`, `DlyAsk`
5. Step 4: csv, gzip. Submit.
6. Expect roughly 15-20 million rows, about 300-500 MB compressed. Server time is usually 10-40 minutes.
7. Save as `data/raw/wrds/crsp_daily.csv.gz`.

If the query fails for being too large, run it twice with date ranges 1999-01-01 to 2012-12-31 and
2013-01-01 to latest, save as `crsp_daily_part1.csv.gz` and `crsp_daily_part2.csv.gz`, and tell Claude.

## Step 4: Fama-French daily factors (market return)

1. Get Data -> Fama-French Portfolios and Factors -> **Fama-French Portfolios and Factors** -> Daily frequency ("Fama-French Factors - Daily Frequency").
2. Date range: 1999-01-01 to latest.
3. Variables: `date`, `mktrf`, `smb`, `hml`, `rf` (tick `umd` too if listed).
4. csv, no compression. Save as `data/raw/wrds/ff_daily.csv`.

## Step 5: Capital IQ Key Developments (company news events)

1. Get Data -> Compustat - Capital IQ -> Capital IQ -> **Key Developments** (the WRDS-linked version, which has a GVKEY field).
   If you cannot see Key Developments, NTU does not subscribe to it: skip this step and tell Claude
   (the code then falls back to leadership changes only, and the live tracker supplies other news types).
2. Date range: **2000-01-01** to latest.
3. Company codes: **GVKEY**, upload `data/wrds_upload/gvkeys.txt`.
4. Event types: **select all**. (The code drops routine items like conference calls and dividend dates itself.)
   If the form forces you to pick and the full list is too slow, untick these to save time:
   Conferences, Company Conference Presentations, Earnings Calls, Shareholder/Analyst Calls,
   Estimated Earnings Release Date, Earnings Release Date, Ex-Div Date, Dividend Affirmations,
   Annual General Meeting, Board Meeting, Fiscal Year End Changes.
5. Variables: `keydevid`, `gvkey`, `companyname`, `announcedate`, `eventtype`, `keydeveventtypeid`, `headline`.
   Do **not** tick `situation` (the long article text); it makes the file 10x bigger.
6. csv, gzip. Save as `data/raw/wrds/ciq_keydev.csv.gz`.

---

## After downloading

```
cd ~/Downloads/stock-prediction-model-
pip install -r requirements.txt
python src/pipeline/02_build_events.py      # first run converts CRSP to a fast cache: ~5 min; event study ~5-15 min
python src/pipeline/17_train_risk_models.py       # ~10-20 min
```
Results land in `results/report.md`. Models land in `models/`.
