function ModelPerformance() {
  return (
    <div className="glass-card rounded-[32px] p-8 shadow-glow">
      <div className="mb-4">
        <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Model Performance</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">Rainfall, heatwave, clustering, and anomaly metrics</h2>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">Classification metrics</div>
        <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">Anomaly and clustering scores</div>
      </div>
    </div>
  );
}

export default ModelPerformance;
