import { useEffect, useState } from "react";
import { api } from "../services/api";
import StatCard from "../components/StatCard";
import RiskBadge from "../components/RiskBadge";
import RiskExplainer from "../components/RiskExplainer";
import type { Alert, Call, HealthStatus } from "../types";

export default function Dashboard() {
  const [calls, setCalls] = useState<Call[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    api.listCalls().then(setCalls).catch(() => setCalls([]));
    api.listAlerts().then(setAlerts).catch(() => setAlerts([]));
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  const activeCalls = calls.filter((c) => c.status === "active").length;
  const criticalAlerts = alerts.filter((a) => a.severity === "CRITICAL").length;

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold text-slate-100">Security Overview</h1>
        <p className="text-sm text-slate-500">
          Real-time impersonation risk monitoring across active calls.
        </p>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Active Calls" value={activeCalls} />
        <StatCard label="Critical Alerts" value={criticalAlerts} accent="text-riskcrit" />
        <StatCard label="Total Calls Logged" value={calls.length} />
        <StatCard
          label="Backend Status"
          value={health?.status?.toUpperCase() ?? "CHECKING..."}
          accent={health?.status === "ok" ? "text-risk-low" : "text-slate-400"}
        />
      </div>

      <RiskExplainer />

      <section className="bg-panel border border-panelBorder rounded-xl p-4">
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Recent Alerts</h2>
        {alerts.length === 0 ? (
          <p className="text-sm text-slate-500">No alerts yet. Run the Demo Mode to generate sample events.</p>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-slate-500 text-xs uppercase">
              <tr className="text-left">
                <th className="py-2">Call</th>
                <th>Severity</th>
                <th>Message</th>
                <th>Recommended Action</th>
              </tr>
            </thead>
            <tbody>
              {alerts.slice(0, 8).map((a) => (
                <tr key={a.alert_id} className="border-t border-panelBorder">
                  <td className="py-2 text-slate-300">{a.call_id}</td>
                  <td><RiskBadge level={a.severity} /></td>
                  <td className="text-slate-400">{a.message}</td>
                  <td className="text-slate-400">{a.recommended_action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
