# FinRL dashboard

React + FastAPI UI for exploring portfolio curves, FinRL Contest metrics, regimes, and LLM signals.

The UI is intentionally **plain**: zinc surfaces, no animated backgrounds, and a single-column reading width so results read like a lab notebook—not a game HUD.

## Run locally

From repo root:

```bash
./dashboard/start.sh
```

Or manually:

```bash
cd dashboard/backend && pip install -r requirements.txt && uvicorn main:app --reload --port 8000
cd dashboard/frontend && npm install && npm run dev
```

- UI: http://localhost:5173  
- API: http://localhost:8000/docs  

## Configuration

Copy `frontend/.env.example` to `frontend/.env` if the API is not on `http://localhost:8000`:

```
VITE_API_URL=http://127.0.0.1:8000
```

## Data sources (under repo root)

| File | Dashboard use |
|------|----------------|
| `backtest_results/portfolio_value.csv` | Primary in-sample equity (`cppo_value`, `bh_value`) |
| `backtest_results/multi_seed_portfolio.csv` | Optional **`CPPO (5-seed μ)`** curve (`mean` or `seed_*` columns) |
| `backtest_results/oos_portfolio.csv` | **Out-of-sample** panel (`agent`, `buy_hold`) |
| `backtest_results/metrics.json` | KPI cards |
| `trade_data_multi_signal_2019_2023.csv` | Signal heatmap (trade-period source) |

If CSVs are missing, the backend serves synthetic demo series so the UI still loads.
