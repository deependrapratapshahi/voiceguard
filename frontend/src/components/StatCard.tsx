export default function StatCard({
  label,
  value,
  accent = "text-slate-100",
}: {
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className="bg-panel border border-panelBorder rounded-xl p-4 flex flex-col gap-1">
      <span className="text-xs uppercase tracking-wider text-slate-400">{label}</span>
      <span className={`text-2xl font-semibold ${accent}`}>{value}</span>
    </div>
  );
}
