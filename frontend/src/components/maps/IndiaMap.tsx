import { MapPin, Droplet, Wind, Thermometer, Cloud } from 'lucide-react';

function IndiaMap() {
  return (
    <div className="h-full rounded-[32px] border border-slate-800 bg-slate-950/80 p-6 shadow-glow">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">India Climate Map</p>
          <h3 className="mt-2 text-xl font-semibold text-white">State risk overlays</h3>
        </div>
        <div className="rounded-3xl bg-slate-900/80 px-3 py-2 text-xs text-slate-300">Heatmap</div>
      </div>
      <div className="relative h-[320px] rounded-[28px] bg-slate-900/90">
        <div className="absolute inset-0 rounded-[28px] border border-slate-800 bg-[radial-gradient(circle_at_center,_rgba(56,189,248,0.16),_transparent_55%)]" />
        <div className="absolute left-6 top-6 space-y-3 text-sm text-slate-100">
          <div className="flex items-center gap-2 rounded-2xl bg-slate-950/75 px-3 py-2">
            <MapPin size={16} /> Western India
          </div>
          <div className="flex items-center gap-2 rounded-2xl bg-slate-950/75 px-3 py-2">
            <Droplet size={16} /> Rainfall risk
          </div>
          <div className="flex items-center gap-2 rounded-2xl bg-slate-950/75 px-3 py-2">
            <Wind size={16} /> Wind forecast
          </div>
        </div>
        <div className="absolute bottom-6 right-6 grid gap-2 rounded-3xl border border-slate-800 bg-slate-950/90 p-4 text-xs text-slate-300">
          <div className="flex items-center gap-2"><Thermometer size={14} /> Temp boost</div>
          <div className="flex items-center gap-2"><Cloud size={14} /> Cloud cover</div>
          <div className="flex items-center gap-2"><Droplet size={14} /> Flood alert</div>
        </div>
      </div>
    </div>
  );
}

export default IndiaMap;
