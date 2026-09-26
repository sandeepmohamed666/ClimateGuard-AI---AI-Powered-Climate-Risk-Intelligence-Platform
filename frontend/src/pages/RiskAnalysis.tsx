import { useRisk } from '@/hooks/useRisk';

function RiskAnalysis() {
  const { data, isLoading, error } = useRisk();

  return (
    <div className="glass-card rounded-[32px] p-8 shadow-glow">
      <div className="mb-4">
        <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Risk Analysis</p>
        <h2 className="mt-2 text-3xl font-semibold text-white">Heatwave, flood, rainfall, wind, and AQI risk</h2>
      </div>

      {isLoading ? (
        <p className="text-slate-300">Loading risk summary...</p>
      ) : error ? (
        <p className="text-rose-300">Unable to load risk summary.</p>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">
            <p className="text-sm text-slate-400">Composite Climate Risk Score</p>
            <p className="mt-4 text-4xl font-semibold text-white">{data.composite_score}</p>
            <div className="mt-6 space-y-3 text-sm text-slate-300">
              <p>Heatwave Risk: {data.heatwave_risk}</p>
              <p>Flood Risk: {data.flood_risk}</p>
              <p>Rainfall Risk: {data.rainfall_risk}</p>
              <p>Wind Risk: {data.wind_risk}</p>
              <p>AQI Risk: {data.aqi_risk}</p>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/80 p-6">
            <p className="text-sm text-slate-400">Regional Rankings</p>
            <div className="mt-4 space-y-4">
              <div>
                <p className="text-slate-200">District Ranking</p>
                <ul className="mt-2 space-y-2 text-sm text-slate-300">
                  {data.district_ranking.map((item: any) => (
                    <li key={item.district} className="rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3">
                      <p className="font-semibold text-white">{item.district}</p>
                      <p>Rank: {item.rank} · Risk: {item.risk}</p>
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <p className="text-slate-200">State Ranking</p>
                <ul className="mt-2 space-y-2 text-sm text-slate-300">
                  {data.state_ranking.map((item: any) => (
                    <li key={item.state} className="rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3">
                      <p className="font-semibold text-white">{item.state}</p>
                      <p>Rank: {item.rank} · Risk: {item.risk}</p>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default RiskAnalysis;
