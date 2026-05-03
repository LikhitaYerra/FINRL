import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import { fetchDrawdown } from "../api";
import { TrendingDown } from "lucide-react";

const AGENT_COLORS: Record<string, string> = {
  "PPO":               "#60a5fa",
  "CPPO":              "#a78bfa",
  "PPO-DeepSeek":      "#34d399",
  "CPPO-DeepSeek":     "#fb923c",
  "CPPO-MultiSignal":  "#f472b6",
  "Regime-Switch":     "#facc15",
  "NASDAQ-100 (QQQ)":  "#9ca3af",
};

function tickFmt(v: string, idx: number) {
  return idx % 60 === 0 ? v.slice(0, 7) : "";
}

export default function DrawdownChart() {
  const { data, isLoading } = useQuery({
    queryKey: ["drawdown"],
    queryFn: fetchDrawdown,
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return (
      <div className="card animate-pulse">
        <div className="h-5 w-64 bg-gray-800 rounded mb-4" />
        <div className="h-[260px] bg-gray-800 rounded" />
      </div>
    );
  }

  if (!data) return null;

  const agents = Object.keys(data.agents);
  const chartData = data.dates.map((date, i) => {
    const row: Record<string, string | number> = { date };
    agents.forEach((a) => { row[a] = -(data.agents[a][i] ?? 0); });
    return row;
  });

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <TrendingDown size={18} className="text-brand-500" />
        <h2 className="font-semibold text-lg">Drawdown (%)</h2>
        <span className="ml-auto text-xs text-gray-500">Negative = loss from peak</span>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickFormatter={tickFmt}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
          />
          <YAxis
            tickFormatter={(v: number) => v.toFixed(0) + "%"}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
            width={48}
          />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
            labelStyle={{ color: "#9ca3af", fontSize: 11 }}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(v: any, name: any) => [(+v || 0).toFixed(2) + "%", String(name)]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            formatter={(v) => <span style={{ color: AGENT_COLORS[v] ?? "#fff" }}>{v}</span>}
          />
          {agents.map((a) => (
            <Area
              key={a}
              type="monotone"
              dataKey={a}
              stroke={AGENT_COLORS[a] ?? "#fff"}
              fill={AGENT_COLORS[a] ?? "#374151"}
              fillOpacity={0.05}
              strokeWidth={1.5}
              dot={false}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
