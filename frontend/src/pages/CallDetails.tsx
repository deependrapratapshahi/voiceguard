import { Fragment, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { api } from "../services/api";
import RiskBadge from "../components/RiskBadge";
import type { Call, RiskSnapshot } from "../types";

export default function CallDetails() {
  const { callId } = useParams();
  const [call, setCall] = useState<Call | null>(null);
  const [history, setHistory] = useState<RiskSnapshot[]>([]);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  useEffect(() => {
    if (!callId) return;
    api.getCall(callId).then(setCall).catch(() => setCall(null));
    api.getCallRisk(callId).then(setHistory).catch(() => setHistory([]));
  }, [callId]);

  if (!call) return <p className="text-sm text-slate-500">Loading call details...</p>;

  const chartData = history.map((h, i) => ({ idx: i, risk: h.risk_score }));
  const latest = history[history.length - 1];

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-slate-100">Call {call.call_id}</h1>
        <p className="text-sm text-slate-500">
          Caller: {call.caller_label ?? "Unknown"} &middot; Started {new Date(call.started_at).toLocaleString()}
        </p>
      </header>

      {latest && (
        <div className="bg-panel border border-panelBorder rounded-xl p-4 space-y-2">
          <div className="flex items-center gap-3">
            <RiskBadge level={latest.risk_level} />
            <span className="text-sm text-slate-400">Current risk score: {latest.risk_score}</span>
            {latest.recommended_action && (
              <span className="text-sm text-accent">&rarr; {latest.recommended_action}</span>
            )}
          </div>
          {latest.reasons?.length > 0 && (
            <div className="text-sm text-slate-400">
              <span className="font-medium text-slate-300">Why: </span>
              {latest.reasons.join(", ")}
            </div>
          )}
        </div>
      )}

      <div className="bg-panel border border-panelBorder rounded-xl p-4">
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Risk Over Time</h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <XAxis dataKey="idx" stroke="#475569" fontSize={11} />
              <YAxis domain={[0, 100]} stroke="#475569" fontSize={11} />
              <Tooltip contentStyle={{ background: "#111826", border: "1px solid #1f2937" }} />
              <Line type="monotone" dataKey="risk" stroke="#22d3ee" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-panel border border-panelBorder rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead className="text-slate-500 text-xs uppercase bg-bg/40">
            <tr className="text-left">
              <th className="py-2 px-4">Time</th>
              <th>Synthetic Prob.</th>
              <th>Speaker Sim.</th>
              <th>Prosody Anomaly</th>
              <th>Risk</th>
              <th>Action</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {history.map((h, i) => (
              <Fragment key={i}>
                <tr
                  className="border-t border-panelBorder cursor-pointer hover:bg-white/5"
                  onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                >
                  <td className="py-1.5 px-4 text-slate-400 text-xs">{new Date(h.timestamp).toLocaleTimeString()}</td>
                  <td className="text-slate-300">{h.synthetic_probability?.toFixed(2) ?? "-"}</td>
                  <td className="text-slate-300">{h.speaker_similarity?.toFixed(2) ?? "-"}</td>
                  <td className="text-slate-300">{h.prosody_anomaly?.toFixed(2) ?? "-"}</td>
                  <td><RiskBadge level={h.risk_level} /></td>
                  <td className="text-slate-400 text-xs">{h.recommended_action ?? "-"}</td>
                  <td className="text-slate-500 text-xs pr-4">
                    {h.reasons?.length > 0 ? (expandedIdx === i ? "hide why \u25b4" : "why? \u25be") : ""}
                  </td>
                </tr>
                {expandedIdx === i && h.reasons?.length > 0 && (
                  <tr className="bg-bg/40">
                    <td colSpan={7} className="px-4 py-2 text-xs text-slate-400">
                      {h.reasons.join(" \u00b7 ")}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
        {history.length === 0 && (
          <p className="p-4 text-sm text-slate-500">
            No risk events recorded yet for this call. Stream some audio via Live Call Monitoring
            to populate this history.
          </p>
        )}
      </div>
    </div>
  );
}
