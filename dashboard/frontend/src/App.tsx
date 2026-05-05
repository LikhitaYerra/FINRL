import { useCallback, useEffect, useMemo, useState } from "react";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { LineChart, RefreshCw } from "lucide-react";

import StatCards from "./components/StatCards";
import PortfolioChart from "./components/PortfolioChart";
import DrawdownChart from "./components/DrawdownChart";
import MetricsTable from "./components/MetricsTable";
import RegimeTimeline from "./components/RegimeTimeline";
import SignalsHeatmap from "./components/SignalsHeatmap";
import DataSourceBanner from "./components/DataSourceBanner";
import OosPanel from "./components/OosPanel";
import AgentToggle from "./components/AgentToggle";
import { fetchPortfolio } from "./api";

const qc = new QueryClient({
  defaultOptions: { queries: { staleTime: 20_000, retry: 2 } },
});

const SECTIONS = [
  { id: "overview", label: "Overview", href: "#overview" },
  { id: "equity", label: "Equity", href: "#equity" },
  { id: "oos", label: "Out-of-sample", href: "#oos" },
  { id: "regimes", label: "Regimes", href: "#regimes" },
  { id: "metrics", label: "Metrics", href: "#metrics" },
  { id: "signals", label: "Signals", href: "#signals" },
];

function Dashboard() {
  const [hiddenAgents, setHiddenAgents] = useState<Set<string>>(new Set());

  const { data: pf, dataUpdatedAt, refetch, isFetching } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => fetchPortfolio(true),
  });

  const agentIds = useMemo(() => (pf?.agents ? Object.keys(pf.agents) : []), [pf]);

  useEffect(() => {
    setHiddenAgents((h) => {
      const next = new Set<string>();
      for (const x of h) {
        if (agentIds.includes(x)) next.add(x);
      }
      return next;
    });
  }, [agentIds.join("|")]);

  const visibleAgents = useMemo(() => {
    const vis = new Set(agentIds.filter((a) => !hiddenAgents.has(a)));
    if (vis.size === 0 && agentIds.length) return new Set(agentIds);
    return vis;
  }, [agentIds, hiddenAgents]);

  const toggleAgent = useCallback(
    (a: string) => {
      setHiddenAgents((h) => {
        const n = new Set(h);
        if (n.has(a)) {
          n.delete(a);
          return n;
        }
        const nVisible = agentIds.filter((x) => !n.has(x)).length;
        if (nVisible <= 1) return h;
        n.add(a);
        return n;
      });
    },
    [agentIds]
  );

  const lastUpdated = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : "—";

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-20 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-5">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-zinc-500 mb-1">
                <LineChart className="shrink-0" size={18} strokeWidth={1.75} aria-hidden />
                <span className="text-xs font-medium uppercase tracking-wide">FinRL · DeepSeek</span>
              </div>
              <h1 className="text-xl sm:text-2xl font-semibold text-zinc-50 tracking-tight">
                Backtest results
              </h1>
              <p className="mt-1 text-sm text-zinc-500 max-w-lg">
                Reads CSVs from your machine. For research only — not investment advice.
              </p>
            </div>
            <div className="flex items-center gap-3 shrink-0">
              <span className="text-xs font-mono text-zinc-500 tabular-nums">Updated {lastUpdated}</span>
              <button
                type="button"
                onClick={() => refetch()}
                className="inline-flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-800 hover:border-zinc-600 transition-colors"
                title="Reload data"
              >
                <RefreshCw size={15} className={isFetching ? "animate-spin text-zinc-400" : "text-zinc-500"} />
                Refresh
              </button>
            </div>
          </div>

          <nav
            className="mt-6 flex gap-1 overflow-x-auto scrollbar-hide pb-1 -mx-1 px-1 border-t border-zinc-800/80 pt-4"
            aria-label="Sections"
          >
            {SECTIONS.map((s) => (
              <a
                key={s.id}
                href={s.href}
                className="px-3 py-1.5 rounded-lg text-sm text-zinc-500 hover:text-zinc-100 hover:bg-zinc-800/80 whitespace-nowrap transition-colors"
              >
                {s.label}
              </a>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 sm:px-6 py-10 space-y-14 pb-24">
        <section id="overview" className="scroll-mt-36 space-y-6">
          <DataSourceBanner />
          <StatCards />
          {agentIds.length > 0 && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 px-4 py-4">
              <p className="text-xs font-medium text-zinc-500 mb-3">Which lines to show on charts</p>
              <AgentToggle agents={agentIds} visible={visibleAgents} onToggle={toggleAgent} />
            </div>
          )}
        </section>

        <section id="equity" className="scroll-mt-36 space-y-4">
          <div>
            <h2 className="text-lg font-semibold text-zinc-100">In-sample equity & drawdown</h2>
            <p className="text-sm text-zinc-500 mt-0.5">2019–2023 window unless your CSVs differ.</p>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PortfolioChart agentsVisible={visibleAgents} />
            <DrawdownChart agentsVisible={visibleAgents} />
          </div>
        </section>

        <section id="oos" className="scroll-mt-36 space-y-4">
          <div>
            <h2 className="text-lg font-semibold text-zinc-100">Out-of-sample</h2>
            <p className="text-sm text-zinc-500 mt-0.5">Shown only if oos_portfolio.csv exists.</p>
          </div>
          <OosPanel />
        </section>

        <section id="regimes" className="scroll-mt-36 space-y-4">
          <h2 className="text-lg font-semibold text-zinc-100">Regimes</h2>
          <RegimeTimeline />
        </section>

        <section id="metrics" className="scroll-mt-36 space-y-4">
          <h2 className="text-lg font-semibold text-zinc-100">Contest metrics</h2>
          <MetricsTable />
        </section>

        <section id="signals" className="scroll-mt-36 space-y-4">
          <h2 className="text-lg font-semibold text-zinc-100">LLM signals</h2>
          <SignalsHeatmap />
        </section>
      </main>

      <footer className="border-t border-zinc-800 py-6 text-center text-xs text-zinc-600">
        <a
          href="https://arxiv.org/abs/2502.07393"
          target="_blank"
          rel="noopener noreferrer"
          className="text-zinc-400 hover:text-zinc-300 underline underline-offset-2"
        >
          Paper (arXiv:2502.07393)
        </a>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <Dashboard />
    </QueryClientProvider>
  );
}
