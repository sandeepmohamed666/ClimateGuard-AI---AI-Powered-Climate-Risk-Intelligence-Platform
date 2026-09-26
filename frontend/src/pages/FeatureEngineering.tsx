import { useFeatureEngineering } from '@/hooks/useFeatureEngineering';

function FeatureEngineering() {
  const { data, isLoading, error } = useFeatureEngineering();

  return (
    <div className="glass-card rounded-[32px] p-8 shadow-glow">
      <div className="mb-4">
        <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Feature Engineering</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">Original, time, physics, risk, and statistical features</h2>
      </div>

      {isLoading ? (
        <p className="text-slate-300">Loading feature engineering data...</p>
      ) : error ? (
        <p className="text-rose-300">Unable to load feature engineering data.</p>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">
            <h3 className="mb-4 text-xl font-semibold text-white">Feature categories</h3>
            <div className="space-y-3 text-sm text-slate-300">
              {Object.entries({
                'Original Features': data.original_features,
                'Time Features': data.time_features,
                'Weather Physics': data.weather_physics,
                'Risk Features': data.risk_features,
              }).map(([title, values]) => (
                <div key={title}>
                  <p className="mb-2 font-semibold text-slate-100">{title}</p>
                  <div className="flex flex-wrap gap-2">
                    {values.map((feature: string) => (
                      <span key={feature} className="rounded-full border border-slate-700 bg-slate-950/80 px-3 py-1 text-xs text-slate-300">
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">
            <h3 className="mb-4 text-xl font-semibold text-white">Statistical and forecast features</h3>
            <div className="space-y-3 text-sm text-slate-300">
              {Object.entries({
                'Interaction Features': data.interaction_features,
                'Statistical Features': data.statistical_features,
                'Lag Features': data.lag_features,
                'Trend Features': data.trend_features,
                'Forecast Features': data.forecast_features,
              }).map(([title, values]) => (
                <div key={title}>
                  <p className="mb-2 font-semibold text-slate-100">{title}</p>
                  <div className="flex flex-wrap gap-2">
                    {values.map((feature: string) => (
                      <span key={feature} className="rounded-full border border-slate-700 bg-slate-950/80 px-3 py-1 text-xs text-slate-300">
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default FeatureEngineering;
