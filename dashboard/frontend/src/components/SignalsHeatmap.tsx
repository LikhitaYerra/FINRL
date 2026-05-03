import { useQuery } from "@tanstack/react-query";
import { fetchSignalSummary } from "../api";
import { Zap } from "lucide-react";
import clsx from "clsx";

const SIGNALS = [
  { key: "llm_sentiment",            label: "Sentiment",         desc: "1=neg → 5=pos" },
  { key: "llm_risk",                 label: "Risk",              desc: "1=low → 5=high" },
  { key: "llm_confidence",           label: "Confidence",        desc: "1=uncertain → 5=certain" },
  { key: "llm_volatility_forecast",  label: "Volatility Fcst",   desc: "1=calm → 5=volatile" },
];

function cellColor(key: string, val: number): string {
  const v = Math.round(val);
  if (key === "llm_sentiment") {
    if (v >= 4) return "bg-emerald-700/70 text-emerald-200";
    if (v === 3) return "bg-gray-700/70 text-gray-300";
    return "bg-red-800/70 text-red-200";
  }
    if (key === "llm_risk" || key === "llm_volatility_forecast") {
    if (v >= 4) return "bg-red-800/70 text-red-200";
    if (v === 3) return "bg-amber-800/60 text-amber-200";
    return "bg-emerald-700/70 text-emerald-200";
  }
  // confidence: high = good
  if (v >= 4) return "bg-blue-800/70 text-blue-200";
  if (v === 3) return "bg-gray-700/70 text-gray-300";
  return "bg-gray-800/70 text-gray-400";
}

export default function SignalsHeatmap() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["signalSummary"],
    queryFn: fetchSignalSummary,
    refetchInterval: 30_000,
  });

  if (isLoading) return <Skeleton />;
  if (error || !data) return <div className="card text-red-400 text-sm">Failed to load signals.</div>;

  const rows = data.filter((r) => r.ticker);

  return (
    <div className="card overflow-x-auto">
      <div className="flex items-center gap-2 mb-4">
        <Zap size={18} className="text-brand-500" />
        <h2 className="font-semibold text-lg">LLM Signal Heatmap</h2>
        <span className="ml-auto text-xs text-gray-500">Average per ticker (trade period)</span>
      </div>

      <table className="w-full text-sm border-separate border-spacing-y-0.5">
        <thead>
          <tr>
            <th className="text-left pb-2 pr-3 text-gray-400 font-medium">Ticker</th>
            {SIGNALS.map((s) => (
              <th key={s.key} className="text-center pb-2 px-2 text-gray-400 font-medium whitespace-nowrap">
                <div>{s.label}</div>
                <div className="text-[10px] text-gray-600 font-normal">{s.desc}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row.ticker)} className="hover:brightness-110 transition-all">
              <td className="py-1 pr-3 font-mono font-semibold text-gray-200 text-xs">
                {String(row.ticker)}
              </td>
              {SIGNALS.map((s) => {
                const val = row[s.key] as number | undefined;
                return (
                  <td key={s.key} className="py-1 px-2 text-center">
                    {val !== undefined ? (
                      <span
                        className={clsx(
                          "inline-block w-10 py-0.5 rounded text-xs font-mono font-bold",
                          cellColor(s.key, val)
                        )}
                      >
                        {val.toFixed(1)}
                      </span>
                    ) : (
                      <span className="text-gray-600">—</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mt-3 flex gap-4 text-[11px] text-gray-500">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-emerald-700/70" /> Positive / Low risk</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-amber-800/60" /> Neutral</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-red-800/70" /> Negative / High risk</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-blue-800/70" /> High confidence</span>
      </div>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="card animate-pulse">
      <div className="h-5 w-48 bg-gray-800 rounded mb-4" />
      {[...Array(8)].map((_, i) => (
        <div key={i} className="h-8 bg-gray-800 rounded mb-1" />
      ))}
    </div>
  );
}
