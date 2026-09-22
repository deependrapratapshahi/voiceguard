import { useEffect, useState } from "react";
import { api } from "../services/api";
import RiskBadge from "../components/RiskBadge";
import type { Alert } from "../types";

export default function Alerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listAlerts().then(setAlerts).finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-slate-100">Alerts</h1>
        <p className="text-sm text-slate-500">
          Alerts fire automatically when a call's risk score crosses a configured threshold.
        </p>
      </header>

      <div className="bg-panel border border-panelBorder rounded-xl overflow-hidden">
        {loading ? (
          <p className="p-4 text-sm text-slate-500">Loading...</p>
        ) : alerts.length === 0 ? (
          <p className="p-4 text-sm text-slate-500">No alerts yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-slate-500 text-xs uppercase bg-bg/40">
              <tr className="text-left">
                <th className="py-2 px-4">Alert</th>
                <th>Call</th>
                <th>Severity</th>
                <th>Message</th>
                <th>Recommended Action</th>
                <th>Status</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((a) => (
                <tr key={a.alert_id} className="border-t border-panelBorder">
                  <td className="py-2 px-4 text-slate-300">{a.alert_id}</td>
                  <td className="text-slate-400">{a.call_id}</td>
                  <td><RiskBadge level={a.severity} /></td>
                  <td className="text-slate-400 max-w-xs">{a.message}</td>
                  <td className="text-slate-400">{a.recommended_action}</td>
                  <td className="text-slate-400">{a.status}</td>
                  <td className="text-slate-500 text-xs">{new Date(a.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
