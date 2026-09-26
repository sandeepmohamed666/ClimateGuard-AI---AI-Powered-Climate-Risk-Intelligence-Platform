import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const trendData = [
  { name: 'Day 1', value: 68 },
  { name: 'Day 2', value: 72 },
  { name: 'Day 3', value: 75 },
  { name: 'Day 4', value: 73 },
  { name: 'Day 5', value: 76 },
  { name: 'Day 6', value: 78 },
  { name: 'Day 7', value: 80 },
];

function TrendChart() {
  return (
    <div className="h-[260px] rounded-[32px] border border-slate-800 bg-slate-950/80 p-4 shadow-glow">
      <div className="mb-4">
        <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Recent Trends</p>
        <h3 className="text-xl font-semibold text-white">Weekly climate index</h3>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={trendData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
          <XAxis dataKey="name" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
          <Area type="monotone" dataKey="value" stroke="#38bdf8" strokeWidth={3} fill="url(#trendGradient)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export default TrendChart;
