import { useQuery } from "@tanstack/react-query";
import { fetchMetrics, type AgentMetrics } from "../api";
import clsx from "clsx";

const COLS: { key: keyof AgentMetrics; label: string; unit?: string; good?: "high" | "low" }[] = [
  { key: "cumulative_return", label: "Cum. return", unit: "%", good: "high" },
  { key: "max_drawdown", label: "Max DD", unit: "%", good: "low" },
  { key: "rachev_ratio", label: "Rachev", good: "high" },
  { key: "sharpe_ratio", label: "Sharpe", good: "high" },
  { key: "outperform_freq_overall", label: "Outperf.", unit: "%", good: "high" },
  { key: "outperform_freq_downturns", label: "Outperf. stress", unit: "%", good: "high" },
];

function colorFor(
  val: number | null,
  good: "high" | "low" | undefined,
  key: keyof AgentMetrics
): string {
  if (val === null) return "text-zinc-500";
  if (key === "max_drawdown") return val > 40 ? "text-rose-400" : val > 25 ? "text-amber-400" : "text-zinc-200";
  if (good === "high") return val > 0 ? "text-emerald-400" : "text-rose-400";
  if (good === "low") return val < 25 ? "text-emerald-400" : val < 38 ? "text-amber-400" : "text-rose-400";
  return "text-zinc-200";
}

function fmt(val: number | null, unit?: string): string {
  if (val === null || val === undefined) return "—";
  return val.toFixed(unit === "%" ? 1 : 3) + (unit ?? "");
}

export default function MetricsTable() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["metrics"],
    queryFn: fetchMetrics,
    refetchInterval: 60_000,
  });

  if (isLoading) return <Skeleton />;
  if (error || !data)
    return <div className="card border-red-900/40 bg-red-950/10 text-sm text-red-300">Could not load metrics.</div>;

  return (
    <div className="card overflow-x-auto">
      <p className="mb-4 text-xs text-zinc-500">FinRL Contest 2025 · Task 1 definitions.</p>

      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-700">
            <th className="py-2 pr-4 text-left font-medium text-zinc-400">Strategy</th>
            {COLS.map((c) => (
              <th key={c.key} className="px-2 py-2 text-right font-medium text-zinc-400 whitespace-nowrap">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={row.name} className="border-b border-zinc-800/80 hover:bg-zinc-800/30">
              <td className="py-2.5 pr-4 font-medium text-zinc-100 whitespace-nowrap">
                {i === 0 && <span className="mr-1.5 text-amber-500">●</span>}
                {row.name}
              </td>
              {COLS.map((c) => {
                const val = row[c.key] as number | null;
                return (
                  <td
                    key={c.key}
                    className={clsx("py-2.5 px-2 text-right font-mono tabular-nums", colorFor(val, c.good, c.key))}
                  >
                    {fmt(val, c.unit)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      <p className="mt-4 text-xs text-zinc-500 leading-relaxed">
        ● Highest cumulative return in this table. Rachev compares extreme upside vs downside tail losses.
      </p>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="card animate-pulse space-y-2">
      <div className="mb-4 h-4 w-48 rounded bg-zinc-800" />
      {[...Array(6)].map((_, i) => (
        <div key={i} className="h-9 rounded bg-zinc-800/80" />
      ))}
    </div>
  );
}
