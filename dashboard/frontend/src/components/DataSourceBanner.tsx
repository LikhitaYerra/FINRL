import { useQuery } from "@tanstack/react-query";
import { Database, FileQuestion } from "lucide-react";
import { fetchPortfolio, type PortfolioMeta } from "../api";

function tags(meta: PortfolioMeta): string[] {
  const f = meta.features ?? [];
  const out: string[] = [];
  if (f.includes("multi_seed_mean")) out.push("Multi-seed mean");
  if (f.includes("oos_csv")) out.push("OOS file present");
  return out;
}

export default function DataSourceBanner() {
  const { data, isLoading } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => fetchPortfolio(true),
    staleTime: 20_000,
  });

  if (isLoading || !data?.meta) return null;

  const { mode, primary_source } = data.meta;
  const extra = tags(data.meta);

  const RealIcon = Database;
  const DemoIcon = FileQuestion;

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-4 py-3 flex flex-wrap items-start gap-3 text-sm">
      <div className="mt-0.5 shrink-0 text-zinc-500">
        {mode === "real" ? <RealIcon size={16} aria-hidden /> : <DemoIcon size={16} aria-hidden />}
      </div>
      <div className="min-w-0 flex-1 space-y-1">
        <p className="text-zinc-200 font-medium">
          {mode === "real" ? "Using saved backtest CSVs" : "Demo curves — drop CSVs in backtest_results/ for live data"}
        </p>
        {primary_source && (
          <p className="text-xs font-mono text-zinc-500 break-all">{primary_source}</p>
        )}
        {extra.length > 0 && (
          <p className="text-xs text-zinc-500">{extra.join(" · ")}</p>
        )}
      </div>
    </div>
  );
}
