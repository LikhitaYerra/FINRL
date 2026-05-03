import { useQuery } from "@tanstack/react-query";
import { fetchBacktest, fetchRegimeSummary } from "../api";
import clsx from "clsx";

interface Stat { label: string; value: string; sub?: string; color?: string }

function Card({ label, value, sub, color }: Stat) {
  return (
    <div className="card flex flex-col gap-1">
      <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
      <span className={clsx("text-2xl font-bold tabular-nums", color ?? "text-white")}>{value}</span>
      {sub && <span className="text-xs text-gray-500">{sub}</span>}
    </div>
  );
}

export default function StatCards() {
  const { data: bt }     = useQuery({ queryKey: ["backtest"], queryFn: fetchBacktest, refetchInterval: 120_000 });
  const { data: regime } = useQuery({ queryKey: ["regimeSummary"], queryFn: fetchRegimeSummary, refetchInterval: 60_000 });

  const agent = bt?.agent;
  const bh    = bt?.buy_hold;

  const ddDiff = agent && bh ? (bh.max_drawdown_pct - agent.max_drawdown_pct) : null;

  const stats: Stat[] = [
    {
      label: "Cumulative Return",
      value: agent ? `${agent.cumulative_return.toFixed(1)}%` : "—",
      sub:   bh ? `Buy & Hold: ${bh.cumulative_return.toFixed(1)}%` : undefined,
      color: (agent?.cumulative_return ?? 0) > 0 ? "text-emerald-400" : "text-red-400",
    },
    {
      label: "Annual Return",
      value: agent ? `${agent.annual_return.toFixed(1)}%` : "—",
      sub:   bh ? `vs B&H ${bh.annual_return.toFixed(1)}%` : undefined,
      color: (agent?.annual_return ?? 0) > 0 ? "text-emerald-400" : "text-red-400",
    },
    {
      label: "Sharpe Ratio",
      value: agent ? agent.sharpe_ratio.toFixed(3) : "—",
      sub:   bh ? `B&H: ${bh.sharpe_ratio.toFixed(3)}` : undefined,
      color: (agent?.sharpe_ratio ?? 0) > 1 ? "text-emerald-400" : "text-amber-400",
    },
    {
      label: "Sortino Ratio",
      value: agent ? agent.sortino_ratio.toFixed(3) : "—",
      sub:   bh ? `B&H: ${bh.sortino_ratio.toFixed(3)}` : undefined,
      color: (agent?.sortino_ratio ?? 0) > 1 ? "text-emerald-400" : "text-amber-400",
    },
    {
      label: "Max Drawdown",
      value: agent ? `${agent.max_drawdown_pct.toFixed(1)}%` : "—",
      sub:   ddDiff != null ? `${ddDiff.toFixed(1)}pp less than B&H` : undefined,
      color: Math.abs(agent?.max_drawdown_pct ?? 99) < 35 ? "text-emerald-400" : "text-amber-400",
    },
    {
      label: "Bear Regime Days",
      value: regime ? `${regime.bear_pct}%` : "—",
      sub:   regime ? `${regime.bear_days} of ${regime.total_days} days` : undefined,
      color: "text-red-400",
    },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      {stats.map((s) => <Card key={s.label} {...s} />)}
    </div>
  );
}
