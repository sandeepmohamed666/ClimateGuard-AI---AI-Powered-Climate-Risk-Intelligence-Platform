function RiskGauge({ score }: { score: number }) {
  const color = score > 80 ? 'text-rose-400' : score > 60 ? 'text-amber-300' : score > 40 ? 'text-emerald-300' : 'text-sky-300';
  return (
    <div className="rounded-[32px] border border-slate-800 bg-slate-950/80 p-6 shadow-glow">
      <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Composite risk gauge</p>
      <p className="mt-5 text-6xl font-semibold tracking-tight text-white">{score}</p>
      <p className={`mt-2 text-sm font-semibold ${color}`}>{score > 80 ? 'Critical' : score > 60 ? 'High' : score > 40 ? 'Moderate' : 'Low'}</p>
      <div className="mt-6 h-2 overflow-hidden rounded-full bg-slate-800">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
    </div>
  );
}

export default RiskGauge;
