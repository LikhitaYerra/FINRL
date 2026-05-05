import clsx from "clsx";
import { AGENT_COLORS } from "../chartColors";

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
          type="button"
          onClick={() => onToggle(a)}
          className={clsx(
            "flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors",
            visible.has(a)
              ? "border-zinc-500 bg-zinc-800 text-zinc-50"
              : "border-zinc-800 bg-transparent text-zinc-500 hover:border-zinc-600 hover:text-zinc-400"
          )}
        >
          <span
            className="h-2 w-2 shrink-0 rounded-full ring-2 ring-zinc-950"
            style={{ backgroundColor: AGENT_COLORS[a] ?? "#71717a" }}
          />
          <span className="truncate max-w-[14rem]">{a}</span>
        </button>
      ))}
    </div>
  );
}
