import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { Bot, RefreshCw } from "lucide-react";

import StatCards      from "./components/StatCards";
import PortfolioChart from "./components/PortfolioChart";
import DrawdownChart  from "./components/DrawdownChart";
import MetricsTable   from "./components/MetricsTable";
import RegimeTimeline from "./components/RegimeTimeline";
import SignalsHeatmap from "./components/SignalsHeatmap";
import { fetchPortfolio } from "./api";

const qc = new QueryClient({
  defaultOptions: { queries: { staleTime: 20_000, retry: 2 } },
});

const ALL_AGENTS = ["CPPO-MultiSignal", "Buy & Hold (EW)"];

function Dashboard() {
  const visibleAgents = new Set(ALL_AGENTS);

  const { dataUpdatedAt, refetch, isFetching } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => fetchPortfolio(true),
  });

  const lastUpdated = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString()
    : "—";

  return (
    <div className="min-h-screen bg-gray-950">
      {/* ── Header ── */}
      <header className="sticky top-0 z-20 bg-gray-950/90 backdrop-blur border-b border-gray-800">
        <div className="max-w-screen-2xl mx-auto px-4 py-3 flex items-center gap-3">
          <Bot size={22} className="text-brand-500" />
          <span className="font-bold text-base tracking-tight">FinRL LLM Trading Bot</span>
          <span className="hidden sm:inline text-xs text-gray-500 border border-gray-700 rounded px-2 py-0.5">
            CPPO · Multi-Signal · 2019–2023
          </span>
          <span className="ml-auto text-xs text-gray-600 hidden sm:inline">Updated {lastUpdated}</span>
          <button
            onClick={() => refetch()}
            className="p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
            title="Refresh data"
          >
            <RefreshCw size={14} className={isFetching ? "animate-spin text-brand-500" : "text-gray-500"} />
          </button>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="max-w-screen-2xl mx-auto px-4 py-6 space-y-5">

        {/* KPI summary cards */}
        <StatCards />

        {/* Portfolio equity curve + Drawdown */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          <PortfolioChart agentsVisible={visibleAgents} />
          <DrawdownChart />
        </div>

        {/* Regime timeline */}
        <RegimeTimeline />

        {/* Performance metrics table */}
        <MetricsTable />

        {/* LLM signals heatmap */}
        <SignalsHeatmap />

      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-gray-800 mt-8 py-4 text-center text-xs text-gray-600">
        FinRL-DeepSeek · LLM-Infused Risk-Sensitive Reinforcement Learning for Trading ·{" "}
        <a href="https://arxiv.org/abs/2502.07393" target="_blank" rel="noopener noreferrer"
           className="underline hover:text-gray-400">
          arXiv:2502.07393
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
