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
  return idx % 40 === 0 ? v.slice(0, 7) : "";
}

export default function OosPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => fetchPortfolio(true),
    staleTime: 20_000,
  });

  if (isLoading || !data?.oos) return null;

  const { dates, agents } = data.oos;
  const keys = Object.keys(agents);
  const chartData = dates.map((date, i) => {
    const row: Record<string, string | number> = { date };
    keys.forEach((k) => {
      row[k] = agents[k][i] ?? 0;
    });
    return row;
  });

  return (
    <div className="card border-zinc-700/80">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-medium text-zinc-100">Hold-out period</h3>
        <span className="text-xs text-zinc-500 font-mono">oos_portfolio.csv</span>
      </div>

      <ResponsiveContainer width="100%" height={260}>
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
          {keys.map((k) => (
            <Line
              key={k}
              type="monotone"
              dataKey={k}
              stroke={AGENT_COLORS[k] ?? "#e4e4e7"}
              strokeWidth={1.75}
              strokeDasharray={STROKE_DASH[k]}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
