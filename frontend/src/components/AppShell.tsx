import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/live", label: "Live Call Monitoring" },
  { to: "/alerts", label: "Alerts" },
  { to: "/speakers", label: "Speaker Registry" },
  { to: "/analytics", label: "Analytics" },
  { to: "/demo", label: "Demo Mode" },
];

export default function AppShell() {
  return (
    <div className="min-h-screen flex bg-bg">
      <aside className="w-60 shrink-0 border-r border-panelBorder bg-panel/60 flex flex-col">
        <div className="px-5 py-5 border-b border-panelBorder">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-accent shadow-[0_0_8px_2px] shadow-accent/60" />
            <span className="font-bold tracking-wide text-slate-100">VOICEGUARD</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Impersonation Detection SOC</p>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `block px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-accent/10 text-accent border border-accent/30"
                    : "text-slate-400 hover:text-slate-100 hover:bg-white/5"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="px-4 py-3 border-t border-panelBorder text-[11px] text-slate-500">
          Demo / baseline models active
        </div>
      </aside>
      <main className="flex-1 p-6 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
