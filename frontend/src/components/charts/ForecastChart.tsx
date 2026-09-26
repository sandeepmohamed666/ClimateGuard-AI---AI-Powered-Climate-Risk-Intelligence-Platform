import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useForecast } from '@/hooks/useForecast';

function ForecastChart() {
  const { data, isLoading, error } = useForecast();
  const forecastData = data?.forecast ?? [];

  return (
    <div className="h-[340px] rounded-[32px] border border-slate-800 bg-slate-950/80 p-4 shadow-glow">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">7 Day Forecast</p>
          <h3 className="text-xl font-semibold text-white">Temperature & Rainfall</h3>
        </div>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">{data?.horizon ?? '7 days'}</div>
      </div>

      {isLoading ? (
        <div className="flex h-[260px] items-center justify-center text-slate-400">Loading forecast...</div>
      ) : error ? (
        <div className="flex h-[260px] items-center justify-center text-rose-300">Unable to load forecast.</div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={forecastData.map((item: any) => ({ name: item.date, temp: item.temperature, rain: item.rainfall }))} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
            <Line type="monotone" dataKey="temp" stroke="#38bdf8" strokeWidth={3} dot={false} />
            <Line type="monotone" dataKey="rain" stroke="#22c55e" strokeWidth={3} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

export default ForecastChart;
