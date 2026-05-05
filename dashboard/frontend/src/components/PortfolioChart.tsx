import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { fetchPortfolio } from "../api";
import { AGENT_COLORS, STROKE_DASH } from "../chartColors";

const GRID = "#27272a";
const AXIS = "#71717a";

function tickFormatter(v: string, idx: number) {
  return idx % 60 === 0 ? v.slice(0, 7) : "";
}

interface Props {
  agentsVisible?: Set<string>;
}

export default function PortfolioChart({ agentsVisible }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => fetchPortfolio(true),
    refetchInterval: 30_000,
  });

  if (isLoading) return <Skeleton />;
  if (error || !data) return <ErrorCard msg={String(error)} />;

  const agents = Object.keys(data.agents).filter((a) => !agentsVisible || agentsVisible.has(a));

  const chartData = data.dates.map((date, i) => {
    const row: Record<string, string | number> = { date };
    agents.forEach((a) => {
      row[a] = data.agents[a][i] ?? 0;
    });
    return row;
  });

  const src = data.meta?.mode === "real" ? "Your CSVs" : "Demo / partial CSV";

  return (
    <div className="card">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-medium text-zinc-100">Portfolio value</h3>
        <span className="text-xs text-zinc-500">{src} · normalised to 1</span>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={chartData} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={tickFormatter}
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v: number) => v.toFixed(2) + "×"}
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={48}
          />
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#a1a1aa" }}
            formatter={(value, name) => [`${Number(value ?? 0).toFixed(4)}×`, String(name)]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            formatter={(v) => <span style={{ color: AGENT_COLORS[v] ?? "#e4e4e7" }}>{v}</span>}
          />
          <ReferenceLine y={1} stroke="#52525b" strokeDasharray="4 4" />

          {agents.map((a) => (
            <Line
              key={a}
              type="monotone"
              dataKey={a}
              stroke={AGENT_COLORS[a] ?? "#e4e4e7"}
              dot={false}
              strokeWidth={1.75}
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
      <div className="mb-4 h-5 w-40 rounded bg-zinc-800" />
      <div className="h-[320px] rounded-lg bg-zinc-800/80" />
    </div>
  );
}

function ErrorCard({ msg }: { msg: string }) {
  return <div className="card border-red-900/50 bg-red-950/20 text-sm text-red-300">Could not load portfolio: {msg}</div>;
}
