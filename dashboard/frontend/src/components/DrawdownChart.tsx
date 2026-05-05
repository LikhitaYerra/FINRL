import { useQuery } from "@tanstack/react-query";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { fetchDrawdown } from "../api";
import { AGENT_COLORS } from "../chartColors";

const GRID = "#27272a";
const AXIS = "#71717a";

function tickFmt(v: string, idx: number) {
  return idx % 60 === 0 ? v.slice(0, 7) : "";
}

interface Props {
  agentsVisible?: Set<string>;
}

export default function DrawdownChart({ agentsVisible }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ["drawdown"],
    queryFn: fetchDrawdown,
    refetchInterval: 30_000,
  });

  if (isLoading) {
    return (
      <div className="card animate-pulse">
        <div className="mb-4 h-5 w-36 rounded bg-zinc-800" />
        <div className="h-[260px] rounded-lg bg-zinc-800/80" />
      </div>
    );
  }

  if (!data) return null;

  const agents = Object.keys(data.agents).filter((a) => !agentsVisible || agentsVisible.has(a));
  const chartData = data.dates.map((date, i) => {
    const row: Record<string, string | number> = { date };
    agents.forEach((a) => {
      row[a] = -(data.agents[a][i] ?? 0);
    });
    return row;
  });

  return (
    <div className="card">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-medium text-zinc-100">Drawdown</h3>
        <span className="text-xs text-zinc-500">% below trailing peak</span>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={chartData} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={tickFmt}
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v: number) => v.toFixed(0) + "%"}
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={44}
          />
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#a1a1aa" }}
            formatter={(value, name) => [`${Number(value ?? 0).toFixed(2)}%`, String(name)]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            formatter={(v) => <span style={{ color: AGENT_COLORS[v] ?? "#e4e4e7" }}>{v}</span>}
          />
          {agents.map((a) => (
            <Area
              key={a}
              type="monotone"
              dataKey={a}
              stroke={AGENT_COLORS[a] ?? "#e4e4e7"}
              fill={AGENT_COLORS[a] ?? "#71717a"}
              fillOpacity={0.06}
              strokeWidth={1.5}
              dot={false}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
