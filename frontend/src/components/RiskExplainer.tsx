import RiskBadge from "./RiskBadge";
import type { RiskLevel } from "../types";

const LEVEL_INFO: { level: RiskLevel; action: string; meaning: string }[] = [
  { level: "LOW", action: "ALLOW", meaning: "No significant risk indicators detected." },
  { level: "MEDIUM", action: "MONITOR", meaning: "Some risk indicators present; keep watching." },
  { level: "HIGH", action: "SECONDARY_VERIFICATION", meaning: "Ask for additional identity verification." },
  { level: "CRITICAL", action: "ESCALATE", meaning: "Pause the request and involve a human reviewer." },
];

export default function RiskExplainer() {
  return (
    <div className="bg-panel border border-panelBorder rounded-xl p-4">
      <h2 className="text-sm font-semibold text-slate-300 mb-1">How Risk Scores Are Calculated</h2>
      <p className="text-xs text-slate-500 mb-3">
        Every score combines ten signals &mdash; synthetic speech probability, speaker similarity,
        prosody anomaly, caller reputation, registered-contact status, transaction amount,
        privileged action, urgency, callback-bypass requests, and historical fraud flags &mdash;
        into one explainable number. A HIGH-risk call where the caller asked to skip a callback is
        routed to <span className="text-slate-300 font-medium">CALLBACK_AND_MFA</span> instead of
        generic secondary verification. This is a risk indicator to guide further checks, not
        proof of fraud.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {LEVEL_INFO.map((row) => (
          <div key={row.level} className="bg-bg/50 border border-panelBorder rounded-lg p-3 space-y-1.5">
            <RiskBadge level={row.level} />
            <div className="text-xs text-slate-300 font-medium">{row.action}</div>
            <div className="text-xs text-slate-500">{row.meaning}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
