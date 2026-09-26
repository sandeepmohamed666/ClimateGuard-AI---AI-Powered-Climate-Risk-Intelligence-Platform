function Forecasting() {
  return (
    <div className="glass-card rounded-[32px] p-8 shadow-glow">
      <div className="mb-4">
        <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Forecasting</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">24-hour, 7-day, and seasonal outlooks</h2>
      </div>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">24 Hours</div>
        <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">7 Days</div>
        <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">30 Days</div>
      </div>
    </div>
  );
}

export default Forecasting;
