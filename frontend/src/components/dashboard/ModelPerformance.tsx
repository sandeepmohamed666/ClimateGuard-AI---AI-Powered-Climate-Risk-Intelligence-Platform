function ModelPerformance({ metrics }: { metrics: Record<string, any> }) {
  const rainfall = metrics?.rainfall ?? {};
  const heatwave = metrics?.heatwave ?? {};
  const anomaly = metrics?.anomaly ?? {};
  const risk = metrics?.risk ?? {};

  const metricCards = [
    {
      title: 'Rainfall',
      values: [
        ['Accuracy', rainfall.accuracy],
        ['Precision', rainfall.precision],
        ['Recall', rainfall.recall],
      ],
    },
    {
      title: 'Heatwave',
      values: [
        ['Accuracy', heatwave.accuracy],
        ['ROC AUC', heatwave.roc_auc],
        ['F1 Score', heatwave.f1_score],
      ],
    },
    {
      title: 'Anomaly',
      values: [
        ['Accuracy', anomaly.accuracy],
        ['Precision', anomaly.precision],
        ['Support', anomaly.support],
      ],
    },
    {
      title: 'Risk Score',
      values: [
        ['MAE', risk.mae],
        ['RMSE', risk.rmse],
        ['R2', risk.r2],
      ],
    },
  ];

  return (
    <div className="rounded-[32px] border border-slate-800 bg-slate-950/80 p-6 shadow-glow">
      <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Model Performance</p>
      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        {metricCards.map((card) => (
          <div key={card.title} className="rounded-3xl border border-slate-800 bg-slate-900/80 p-4">
            <p className="font-semibold text-white">{card.title}</p>
            <div className="mt-3 space-y-1 text-sm text-slate-300">
              {card.values.map(([label, value]) => (
                <p key={label}>
                  <span className="text-slate-400">{label}:</span>{' '}
                  <span className="text-white">{value ?? '—'}</span>
                </p>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ModelPerformance;
