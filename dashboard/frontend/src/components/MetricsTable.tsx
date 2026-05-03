import { useQuery } from "@tanstack/react-query";
import { fetchMetrics, type AgentMetrics } from "../api";
import { BarChart2 } from "lucide-react";
import clsx from "clsx";

const COLS: { key: keyof AgentMetrics; label: string; unit?: string; good?: "high" | "low" }[] = [
  { key: "cumulative_return",        label: "Cum. Return",     unit: "%",  good: "high" },
  { key: "max_drawdown",             label: "Max Drawdown",    unit: "%",  good: "low"  },
  { key: "rachev_ratio",             label: "Rachev Ratio",               good: "high" },
  { key: "sharpe_ratio",             label: "Sharpe Ratio",               good: "high" },
  { key: "outperform_freq_overall",  label: "Outperf. Overall",unit: "%",  good: "high" },
  { key: "outperform_freq_downturns",label: "Outperf. Downturn",unit: "%", good: "high" },
];

function colorFor(
  val: number | null,
  good: "high" | "low" | undefined,
  key: keyof AgentMetrics
): string {
  if (val === null) return "text-gray-500";
  if (key === "max_drawdown") return val > 40 ? "text-red-400" : val > 20 ? "text-amber-400" : "text-emerald-400";
  if (good === "high") return val > 0 ? "text-emerald-400" : "text-red-400";
  if (good === "low")  return val < 15 ? "text-emerald-400" : val < 30 ? "text-amber-400" : "text-red-400";
  return "text-gray-300";
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
  if (error || !data) return <div className="card text-red-400 text-sm">Failed to load metrics.</div>;

  return (
    <div className="card overflow-x-auto">
      <div className="flex items-center gap-2 mb-4">
        <BarChart2 size={18} className="text-brand-500" />
        <h2 className="font-semibold text-lg">Contest Metrics Comparison</h2>
        <span className="ml-auto text-xs text-gray-500">FinRL Contest 2025 Task 1</span>
      </div>

      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800">
            <th className="text-left py-2 pr-4 text-gray-400 font-medium">Agent</th>
            {COLS.map((c) => (
              <th key={c.key} className="text-right py-2 px-3 text-gray-400 font-medium whitespace-nowrap">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr
              key={row.name}
              className={clsx(
                "border-b border-gray-800/50 transition-colors hover:bg-gray-800/40",
                i === 0 && "bg-brand-900/20"
              )}
            >
              <td className="py-2.5 pr-4 font-medium whitespace-nowrap">
                {i === 0 && <span className="mr-1.5 text-yellow-400">★</span>}
                {row.name}
              </td>
              {COLS.map((c) => {
                const val = row[c.key] as number | null;
                return (
                  <td
                    key={c.key}
                    className={clsx(
                      "py-2.5 px-3 text-right font-mono tabular-nums",
                      colorFor(val, c.good, c.key)
                    )}
                  >
                    {fmt(val, c.unit)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      <p className="mt-3 text-xs text-gray-600">
        ★ Best cumulative return. Rachev ratio: expected top-5% return / expected bottom-5% loss. Higher is better.
      </p>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="card animate-pulse space-y-2">
      <div className="h-5 w-56 bg-gray-800 rounded mb-4" />
      {[...Array(7)].map((_, i) => (
        <div key={i} className="h-8 bg-gray-800 rounded" />
      ))}
    </div>
  );
}
