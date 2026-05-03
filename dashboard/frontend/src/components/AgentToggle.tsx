import clsx from "clsx";

const AGENT_COLORS: Record<string, string> = {
  "PPO":               "#60a5fa",
  "CPPO":              "#a78bfa",
  "PPO-DeepSeek":      "#34d399",
  "CPPO-DeepSeek":     "#fb923c",
  "CPPO-MultiSignal":  "#f472b6",
  "Regime-Switch":     "#facc15",
  "NASDAQ-100 (QQQ)":  "#9ca3af",
};

interface Props {
  agents: string[];
  visible: Set<string>;
  onToggle: (agent: string) => void;
}

export default function AgentToggle({ agents, visible, onToggle }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {agents.map((a) => (
        <button
          key={a}
          onClick={() => onToggle(a)}
          className={clsx(
            "flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border transition-all",
            visible.has(a)
              ? "border-transparent text-gray-900"
              : "border-gray-700 text-gray-500 bg-transparent hover:border-gray-600"
          )}
          style={visible.has(a) ? { background: AGENT_COLORS[a] ?? "#374151" } : {}}
        >
          <span
            className="inline-block w-2 h-2 rounded-full"
            style={{ background: AGENT_COLORS[a] ?? "#6b7280" }}
          />
          {a}
        </button>
      ))}
    </div>
  );
}
