import { NavLink } from 'react-router-dom';
import { Gauge, CloudRain, Thermometer, Sparkles, ShieldAlert, PieChart, Settings, Info } from 'lucide-react';

const navItems = [
  { label: 'Dashboard', path: '/dashboard', icon: <Gauge size={18} /> },
  { label: 'Weather', path: '/weather', icon: <CloudRain size={18} /> },
  { label: 'Forecasting', path: '/forecasting', icon: <Thermometer size={18} /> },
  { label: 'Risk Analysis', path: '/risk', icon: <ShieldAlert size={18} /> },
  { label: 'Explainable AI', path: '/explainable-ai', icon: <Sparkles size={18} /> },
  { label: 'Feature Engineering', path: '/feature-engineering', icon: <PieChart size={18} /> },
  { label: 'Data Explorer', path: '/data-explorer', icon: <CloudRain size={18} /> },
  { label: 'Model Performance', path: '/model-performance', icon: <Settings size={18} /> },
  { label: 'About', path: '/about', icon: <Info size={18} /> },
];

function Sidebar() {
  return (
    <aside className="hidden min-h-screen w-72 flex-col border-r border-slate-800 bg-slate-950/90 p-4 shadow-glow lg:flex">
      <div className="mb-8 px-2">
        <div className="mb-4 rounded-3xl border border-slate-700 bg-slate-900/80 p-5 text-slate-100 shadow-glass">
          <p className="text-xs uppercase tracking-[0.32em] text-cyan-300">ClimateGuard AI</p>
          <h1 className="mt-3 text-2xl font-semibold text-white">Risk Intelligence</h1>
          <p className="mt-2 text-sm text-slate-400">AI-powered climate insights across India.</p>
        </div>
      </div>
      <nav className="space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-3xl px-4 py-3 text-sm font-medium transition ${
                isActive ? 'bg-cyan-500/15 text-cyan-200' : 'text-slate-300 hover:bg-slate-800/80 hover:text-white'
              }`
            }
          >
            {item.icon}
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}

export default Sidebar;
