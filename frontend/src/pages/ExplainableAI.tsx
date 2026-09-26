import { useShap } from '@/hooks/useShap';

function ExplainableAI() {
  const { data, isLoading, error } = useShap();

  return (
    <div className="glass-card rounded-[32px] p-8 shadow-glow">
      <div className="mb-4">
        <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Explainable AI</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">SHAP, feature importance, and decision plots</h2>
      </div>

      {isLoading ? (
        <p className="text-slate-300">Loading SHAP insights...</p>
      ) : error ? (
        <p className="text-rose-300">Unable to load SHAP insights.</p>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">
            <p className="text-sm text-slate-400">SHAP Summary Plot</p>
            <div className="mt-4 space-y-3 text-sm text-slate-300">
              {data.summary.map((item: any) => (
                <div key={item.feature} className="rounded-3xl border border-slate-800 bg-slate-950/70 p-4">
                  <p className="font-semibold text-white">{item.feature}</p>
                  <p>Impact: {item.impact}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">
            <p className="text-sm text-slate-400">Waterfall / Force Plot</p>
            <div className="mt-4 h-72 rounded-3xl border border-slate-800 bg-slate-950/80" />
          </div>
        </div>
      )}
    </div>
  );
}

export default ExplainableAI;
