import { motion } from 'framer-motion';
import { useDashboardData } from '@/hooks/useDashboardData';
import IndiaMap from '@/components/maps/IndiaMap';
import ForecastChart from '@/components/charts/ForecastChart';
import TrendChart from '@/components/charts/TrendChart';
import ClimateSummary from '@/components/dashboard/ClimateSummary';
import RiskGauge from '@/components/dashboard/RiskGauge';
import FeatureEngineeringPanel from '@/components/dashboard/FeatureEngineeringPanel';
import DashboardModelPerformance from '@/components/dashboard/ModelPerformance';

function Dashboard() {
  const { data, isLoading } = useDashboardData();

  const weather = data?.weather ?? {};
  const risk = data?.risk ?? {};
  const alerts = data?.alerts ?? [];
  const metrics = [
    { label: 'Temp', value: weather.temperature ? `${weather.temperature}°C` : '--' },
    { label: 'Humidity', value: weather.humidity ? `${weather.humidity}%` : '--' },
    { label: 'Rainfall', value: weather.rainfall ? `${weather.rainfall} mm` : '--' },
    { label: 'Pressure', value: weather.pressure ? `${weather.pressure} hPa` : '--' },
    { label: 'Wind', value: weather.wind ? `${weather.wind} km/h` : '--' },
    { label: 'Cloud', value: weather.cloud ? `${weather.cloud}%` : '--' },
    { label: 'UV', value: weather.uv ?? '--' },
    { label: 'Visibility', value: weather.visibility ? `${weather.visibility} km` : '--' },
    { label: 'AQI', value: weather.aqi ?? '--' },
  ];

  const riskCards = [
    { label: 'Overall Risk', value: risk.overall },
    { label: 'Heatwave Risk', value: risk.heatwave },
    { label: 'Flood Risk', value: risk.flood },
    { label: 'Rainfall Risk', value: risk.rainfall },
    { label: 'Air Quality Risk', value: risk.air_quality },
    { label: 'Storm Risk', value: risk.storm },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      <section className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
        <div className="glass-card rounded-[32px] p-6 shadow-glow">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Today's Weather</p>
                <h3 className="mt-2 text-2xl font-semibold text-white">Live summary & climate conditions</h3>
              </div>
              <span className="rounded-3xl bg-emerald-400/15 px-4 py-2 text-sm font-semibold text-emerald-300">
                {isLoading ? 'Loading...' : 'Updated just now'}
              </span>
            </div>
            <ClimateSummary weather={weather} />
          </div>
        </div>

        <div className="space-y-6">
          <RiskGauge score={data?.risk?.overall_score ?? 68} />

          <div className="glass-card rounded-[32px] p-6 shadow-glow">
            <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Current Alerts</p>
            <div className="mt-5 grid gap-3">
              {alerts.length > 0 ? (
                alerts.map((alert: any) => (
                  <div key={alert.title} className="rounded-3xl border border-amber-400/10 bg-amber-500/10 p-4 text-sm text-amber-100">
                    <p className="font-semibold text-white">{alert.title}</p>
                    <p>{alert.description}</p>
                  </div>
                ))
              ) : (
                <p className="text-slate-400">No active alerts</p>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.4fr_0.6fr]">
        <IndiaMap />

        <div className="space-y-6">
          <ForecastChart />
          <div className="glass-card rounded-[32px] p-6 shadow-glow">
            <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">AI Insights</p>
            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              {['Clusters', 'Anomalies', 'Forecast', 'Explainable AI'].map((insight) => (
                <div key={insight} className="rounded-3xl border border-slate-800 bg-slate-900/80 p-4 text-white">{insight}</div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-3">
        <TrendChart />
        <FeatureEngineeringPanel />
        <DashboardModelPerformance
          metrics={
            data?.model_metrics ?? {
              rainfall: { accuracy: '--', precision: '--', recall: '--' },
              heatwave: { accuracy: '--', roc_auc: '--', f1_score: '--' },
              anomaly: { accuracy: '--', precision: '--', support: '--' },
              risk: { mae: '--', rmse: '--', r2: '--' },
            }
          }
        />
      </section>
    </motion.div>
  );
}

export default Dashboard;
