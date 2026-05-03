import { useQuery } from "@tanstack/react-query";
import { fetchRegime, fetchRegimeSummary } from "../api";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { Activity } from "lucide-react";

function tickFmt(v: string, idx: number) {
  return idx % 60 === 0 ? v.slice(0, 7) : "";
}

export default function RegimeTimeline() {
  const { data: regime }   = useQuery({ queryKey: ["regime"],        queryFn: fetchRegime,        refetchInterval: 60_000 });
  const { data: summary }  = useQuery({ queryKey: ["regimeSummary"], queryFn: fetchRegimeSummary, refetchInterval: 60_000 });

  const chartData = regime?.map((r) => ({
    date:  r.date,
    value: r.regime === "bear" ? 1 : 0,
  })) ?? [];

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-4">
        <Activity size={18} className="text-brand-500" />
        <h2 className="font-semibold text-lg">Regime Timeline</h2>
        {summary && (
          <div className="ml-auto flex gap-3 text-xs">
            <span className="badge-bull">Bull {summary.bull_pct}%</span>
            <span className="badge-bear">Bear {summary.bear_pct}%</span>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={140}>
        <AreaChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="bearGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tickFormatter={tickFmt}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
          />
          <YAxis
            ticks={[0, 1]}
            tickFormatter={(v) => (v === 1 ? "Bear" : "Bull")}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={{ stroke: "#374151" }}
            width={40}
          />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
            labelStyle={{ color: "#9ca3af", fontSize: 11 }}
            formatter={(v: number | undefined) => [(v ?? 0) === 1 ? "🐻 Bear" : "🐂 Bull", "Regime"]}
          />
          <Area
            type="stepAfter"
            dataKey="value"
            stroke="#ef4444"
            fill="url(#bearGrad)"
            strokeWidth={1.5}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      <p className="mt-2 text-xs text-gray-600">
        Bear regime (value = 1) → CPPO-DeepSeek agent active. Bull (0) → PPO agent.
      </p>
    </div>
  );
}
