import { useQuery } from "@tanstack/react-query";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { fetchPortfolio } from "../api";
import { TrendingUp } from "lucide-react";

const AGENT_COLORS: Record<string, string> = {
  "PPO":               "#60a5fa",
  "CPPO":              "#a78bfa",
  "PPO-DeepSeek":      "#34d399",
  "CPPO-DeepSeek":     "#fb923c",
  "CPPO-MultiSignal":  "#f472b6",
  "Regime-Switch":     "#facc15",
  "NASDAQ-100 (QQQ)":  "#9ca3af",
};

const STROKE_DASH: Record<string, string> = {
  "NASDAQ-100 (QQQ)": "6 3",
};

// Show every ~60th date label
function tickFormatter(v: string, idx: number) {
  return idx % 60 === 0 ? v.slice(0, 7) : "";
}

interface Props { agentsVisible?: Set<string> }

export default function PortfolioChart({ agentsVisible }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["portfolio"],
    queryFn:  () => fetchPortfolio(true),
    refetchInterval: 30_000,
  });

  if (isLoading) return <Skeleton />;
  if (error || !data) return <ErrorCard msg={String(error)} />;

  const agents = Object.keys(data.agents).filter(
    (a) => !agentsVisible || agentsVisible.has(a)
  );

  const chartData = data.dates.map((date, i) => {
    const row: Record<string, string | number> = { date };
    agents.forEach((a) => { row[a] = data.agents[a][i] ?? 0; });
    return row;
  });

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <TrendingUp size={18} className="text-brand-500" />
        <h2 className="font-semibold text-lg">Portfolio Performance (2019 – 2023)</h2>
        <span className="ml-auto text-xs text-gray-500">Normalised to 1.0 at start</span>
      </div>

      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickFormatter={tickFormatter}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
          />
          <YAxis
            tickFormatter={(v: number) => v.toFixed(2) + "×"}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
            width={55}
          />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
            labelStyle={{ color: "#9ca3af", fontSize: 11 }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(v: any, name: any) => [(+v || 0).toFixed(4) + "×", String(name)]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 12 }}
            formatter={(v) => <span style={{ color: AGENT_COLORS[v] ?? "#fff" }}>{v}</span>}
          />
          <ReferenceLine y={1} stroke="#374151" strokeDasharray="4 2" />

          {agents.map((a) => (
            <Line
              key={a}
              type="monotone"
              dataKey={a}
              stroke={AGENT_COLORS[a] ?? "#fff"}
              dot={false}
              strokeWidth={a === "Regime-Switch" ? 2.5 : 1.5}
              strokeDasharray={STROKE_DASH[a]}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="card animate-pulse">
      <div className="h-5 w-64 bg-gray-800 rounded mb-4" />
      <div className="h-[340px] bg-gray-800 rounded" />
    </div>
  );
}
function ErrorCard({ msg }: { msg: string }) {
  return <div className="card text-red-400 text-sm">Failed to load portfolio: {msg}</div>;
}
