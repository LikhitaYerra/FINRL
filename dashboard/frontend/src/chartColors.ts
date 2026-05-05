/** Shared stroke colors for portfolio / drawdown charts and toggles */
export const AGENT_COLORS: Record<string, string> = {
  PPO: "#60a5fa",
  CPPO: "#a78bfa",
  "PPO-DeepSeek": "#34d399",
  "CPPO-DeepSeek": "#fb923c",
  "CPPO-MultiSignal": "#f472b6",
  "CPPO (5-seed μ)": "#22d3ee",
  "CPPO (OOS)": "#a855f7",
  "Regime-Switch": "#facc15",
  "NASDAQ-100 (QQQ)": "#9ca3af",
  "Buy & Hold (EW)": "#94a3b8",
};

/** Dashed lines for benchmarks */
export const STROKE_DASH: Record<string, string> = {
  "NASDAQ-100 (QQQ)": "6 3",
  "Buy & Hold (EW)": "5 4",
  "CPPO (5-seed μ)": "4 3",
};
