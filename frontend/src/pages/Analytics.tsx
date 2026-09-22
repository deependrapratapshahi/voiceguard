import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { api } from "../services/api";
import type { Alert, Call, ModelInfo } from "../types";

export default function Analytics() {
  const [calls, setCalls] = useState<Call[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    api.listCalls().then(setCalls).catch(() => setCalls([]));
    api.listAlerts().then(setAlerts).catch(() => setAlerts([]));
    api.models().then(setModels).catch(() => setModels([]));
  }, []);

  const severityCounts = ["LOW", "MEDIUM", "HIGH", "CRITICAL"].map((level) => ({
    level,
    count: alerts.filter((a) => a.severity === level).length,
  }));

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-slate-100">Analytics</h1>
        <p className="text-sm text-slate-500">
          Detection statistics and model performance. Metrics here reflect demo/baseline
          model behavior, not validated production accuracy figures.
        </p>
      </header>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-panel border border-panelBorder rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-3">Alerts by Severity</h2>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={severityCounts}>
                <XAxis dataKey="level" stroke="#475569" fontSize={11} />
                <YAxis stroke="#475569" fontSize={11} allowDecimals={false} />
                <Tooltip contentStyle={{ background: "#111826", border: "1px solid #1f2937" }} />
                <Bar dataKey="count" fill="#22d3ee" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-panel border border-panelBorder rounded-xl p-4">
          <h2 className="text-sm font-semibold text-slate-300 mb-3">Active Model Versions</h2>
          <table className="w-full text-sm">
            <thead className="text-slate-500 text-xs uppercase">
              <tr className="text-left">
                <th className="py-1">Component</th>
                <th>Name</th>
                <th>Version</th>
                <th>Baseline?</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.component} className="border-t border-panelBorder">
                  <td className="py-1.5 text-slate-300">{m.component}</td>
                  <td className="text-slate-400">{m.name}</td>
                  <td className="text-slate-400">{m.version}</td>
                  <td className="text-slate-400">{m.is_baseline ? "Yes" : "No"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-panel border border-panelBorder rounded-xl p-4">
        <h2 className="text-sm font-semibold text-slate-300 mb-2">Calls by Status</h2>
        <p className="text-sm text-slate-400">
          {calls.length} total calls logged · {calls.filter((c) => c.status === "active").length} active ·{" "}
          {calls.filter((c) => c.status !== "active").length} ended
        </p>
      </div>
    </div>
  );
}
