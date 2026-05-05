const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function get<T>(path: string, params?: Record<string, string | number | boolean>): Promise<T> {
  const url = new URL(BASE + path);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)));
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

// ─── Types ───────────────────────────────────────────────────────────────────

export interface PortfolioMeta {
  mode: "real" | "synthetic" | "unknown";
  primary_source: string | null;
  features: string[];
}

export interface OosPortfolioBlock {
  dates: string[];
  agents: Record<string, number[]>;
}

export interface PortfolioData {
  dates: string[];
  agents: Record<string, number[]>;
  meta?: PortfolioMeta;
  oos?: OosPortfolioBlock | null;
}

export interface AgentMetrics {
  name: string;
  cumulative_return: number;       // %
  max_drawdown: number;            // %
  rachev_ratio: number | null;
  sharpe_ratio: number | null;
  outperform_freq_overall: number | null;   // %
  outperform_freq_downturns: number | null; // %
}

export interface RegimePoint {
  date: string;
  regime: "bull" | "bear";
}

export interface RegimeSummary {
  total_days: number;
  bull_days: number;
  bear_days: number;
  bull_pct: number;
  bear_pct: number;
}

export interface SignalRow {
  date: string;
  ticker: string;
  llm_sentiment: number;
  llm_risk: number;
  llm_confidence: number;
  llm_volatility_forecast: number;
}

export interface DrawdownData {
  dates: string[];
  agents: Record<string, number[]>;
}

export interface BacktestMetricSet {
  initial_capital:    number;
  final_value:        number;
  cumulative_return:  number;
  annual_return:      number;
  sharpe_ratio:       number;
  sortino_ratio:      number;
  max_drawdown_pct:   number;
  calmar_ratio:       number;
  n_trading_days:     number;
}

export interface BacktestResults {
  agent:     BacktestMetricSet;
  buy_hold:  BacktestMetricSet;
}

// ─── Query functions ──────────────────────────────────────────────────────────

export const fetchPortfolio = (normalise = true) =>
  get<PortfolioData>("/api/portfolio", { normalise });
export const fetchMetrics       = ()                  => get<AgentMetrics[]>("/api/metrics");
export const fetchRegime        = ()                  => get<RegimePoint[]>("/api/regime");
export const fetchRegimeSummary = ()                  => get<RegimeSummary>("/api/regime/summary");
export const fetchSignals       = (limit = 200)       => get<SignalRow[]>("/api/signals", { limit, source: "trade" });
export const fetchSignalSummary = ()                  => get<Record<string, number>[]>("/api/signals/summary");
export const fetchDrawdown      = ()                  => get<DrawdownData>("/api/drawdown");
export const fetchBacktest      = ()                  => get<BacktestResults>("/api/backtest");
