import { Bell, Moon, SunMedium } from 'lucide-react';

function Header() {
  return (
    <header className="mb-6 flex flex-col gap-4 rounded-3xl border border-slate-800 bg-slate-950/80 p-5 shadow-glass lg:flex-row lg:items-center lg:justify-between">
      <div>
        <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Live Climate Intelligence</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">ClimateGuard AI Dashboard</h2>
        <p className="mt-1 text-sm text-slate-400">Real-time weather, forecast, risk, and explainability metrics for informed decisions.</p>
      </div>
      <div className="flex items-center gap-3">
        <button className="inline-flex items-center gap-2 rounded-2xl border border-slate-700 bg-slate-900/80 px-4 py-3 text-sm text-slate-200 transition hover:border-cyan-500 hover:text-white">
          <Bell size={18} /> Alerts
        </button>
        <button className="inline-flex items-center gap-2 rounded-2xl border border-slate-700 bg-slate-900/80 px-4 py-3 text-sm text-slate-200 transition hover:border-cyan-500 hover:text-white">
          <Moon size={18} /> Dark
        </button>
      </div>
    </header>
  );
}

export default Header;
