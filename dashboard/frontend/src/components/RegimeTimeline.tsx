import { useQuery } from "@tanstack/react-query";
import { fetchRegime, fetchRegimeSummary } from "../api";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

const GRID = "#27272a";
const AXIS = "#71717a";

function tickFmt(v: string, idx: number) {
  return idx % 60 === 0 ? v.slice(0, 7) : "";
}

export default function RegimeTimeline() {
  const { data: regime } = useQuery({ queryKey: ["regime"], queryFn: fetchRegime, refetchInterval: 60_000 });
  const { data: summary } = useQuery({
    queryKey: ["regimeSummary"],
    queryFn: fetchRegimeSummary,
    refetchInterval: 60_000,
  });

  const chartData =
    regime?.map((r) => ({
      date: r.date,
      value: r.regime === "bear" ? 1 : 0,
    })) ?? [];

  return (
    <div className="card">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-medium text-zinc-100">Bull / bear timeline</h3>
        {summary && (
          <div className="flex gap-2 text-xs">
            <span className="badge-bull">Bull {summary.bull_pct}%</span>
            <span className="badge-bear">Bear {summary.bear_pct}%</span>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={132}>
        <AreaChart data={chartData} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="bearGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#fb7185" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#fb7185" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            tickFormatter={tickFmt}
            tick={{ fill: AXIS, fontSize: 11 }}
            axisLine={{ stroke: GRID }}
            tickLine={false}
          />
          <YAxis
            ticks={[0, 1]}
            tickFormatter={(v) => (v === 1 ? "Bear" : "Bull")}
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
            formatter={(v: number | undefined) => [(v ?? 0) === 1 ? "Bear" : "Bull", "Label"]}
          />
          <Area
            type="stepAfter"
            dataKey="value"
            stroke="#fb7185"
            fill="url(#bearGrad)"
            strokeWidth={1.25}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      <p className="mt-3 text-xs text-zinc-500">Step chart from your regime CSV or synthetic fallback.</p>
    </div>
  );
}
