import { useQuery } from "@tanstack/react-query";
import { fetchSignalSummary } from "../api";
import clsx from "clsx";

const SIGNALS = [
  { key: "llm_sentiment", label: "Sentiment", desc: "1–5" },
  { key: "llm_risk", label: "Risk", desc: "1–5" },
  { key: "llm_confidence", label: "Confidence", desc: "1–5" },
  { key: "llm_volatility_forecast", label: "Vol. fcst.", desc: "1–5" },
];

function cellTone(key: string, val: number): string {
  const v = Math.round(val);
  if (key === "llm_sentiment") {
    if (v >= 4) return "bg-emerald-950 text-emerald-300 border-emerald-800";
    if (v === 3) return "bg-zinc-800 text-zinc-300 border-zinc-700";
    return "bg-rose-950 text-rose-300 border-rose-900";
  }
  if (key === "llm_risk" || key === "llm_volatility_forecast") {
    if (v >= 4) return "bg-rose-950 text-rose-300 border-rose-900";
    if (v === 3) return "bg-amber-950 text-amber-200 border-amber-900";
    return "bg-emerald-950 text-emerald-300 border-emerald-800";
  }
  if (v >= 4) return "bg-sky-950 text-sky-300 border-sky-900";
  if (v === 3) return "bg-zinc-800 text-zinc-300 border-zinc-700";
  return "bg-zinc-900 text-zinc-500 border-zinc-800";
}

export default function SignalsHeatmap() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["signalSummary"],
    queryFn: fetchSignalSummary,
    refetchInterval: 30_000,
  });

  if (isLoading) return <Skeleton />;
  if (error || !data)
    return <div className="card border-red-900/40 bg-red-950/10 text-sm text-red-300">Could not load signals.</div>;

  const rows = data.filter((r) => r.ticker);

  return (
    <div className="card overflow-x-auto">
      <p className="mb-4 text-xs text-zinc-500">Per-ticker averages over the trade CSV tail.</p>

      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-700">
            <th className="py-2 pr-3 text-left font-medium text-zinc-400">Ticker</th>
            {SIGNALS.map((s) => (
              <th key={s.key} className="py-2 px-2 text-center font-medium text-zinc-400 whitespace-nowrap">
                <div>{s.label}</div>
                <div className="text-[10px] font-normal text-zinc-600">{s.desc}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row.ticker)} className="border-b border-zinc-800/80 hover:bg-zinc-800/25">
              <td className="py-1.5 pr-3 font-mono text-xs font-semibold text-zinc-200">{String(row.ticker)}</td>
              {SIGNALS.map((s) => {
                const val = row[s.key] as number | undefined;
                return (
                  <td key={s.key} className="py-1.5 px-2 text-center">
                    {val !== undefined ? (
                      <span
                        className={clsx(
                          "inline-flex min-w-[2.25rem] justify-center rounded border px-1 py-0.5 text-xs font-mono font-medium",
                          cellTone(s.key, val)
                        )}
                      >
                        {val.toFixed(1)}
                      </span>
                    ) : (
                      <span className="text-zinc-600">—</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="card animate-pulse">
      <div className="mb-4 h-4 w-56 rounded bg-zinc-800" />
      {[...Array(8)].map((_, i) => (
        <div key={i} className="mb-1 h-8 rounded bg-zinc-800/80" />
      ))}
    </div>
  );
}
