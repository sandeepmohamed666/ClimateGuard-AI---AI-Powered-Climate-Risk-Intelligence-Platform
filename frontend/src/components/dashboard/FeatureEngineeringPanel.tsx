function FeatureEngineeringPanel() {
  return (
    <div className="rounded-[32px] border border-slate-800 bg-slate-950/80 p-6 shadow-glow">
      <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Feature Engineering Summary</p>
      <div className="mt-5 space-y-4 text-slate-300">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-4">
          <p className="font-semibold text-white">Time-based features</p>
          <p className="mt-2 text-sm">Season, weekend, cyclic hour, month start, and year end.</p>
        </div>
        <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-4">
          <p className="font-semibold text-white">Weather physics</p>
          <p className="mt-2 text-sm">Heat index, dew point, apparent temperature, and feels like.</p>
        </div>
      </div>
    </div>
  );
}

export default FeatureEngineeringPanel;
