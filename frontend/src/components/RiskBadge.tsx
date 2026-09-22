import type { RiskLevel } from "../types";

export default function RiskBadge({ level }: { level: RiskLevel }) {
  return (
    <span className={`risk-badge-${level} px-2.5 py-1 rounded-md text-xs font-semibold tracking-wide`}>
      {level}
    </span>
  );
}
