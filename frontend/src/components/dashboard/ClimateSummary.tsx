function ClimateSummary({ weather }: { weather: Record<string, any> }) {
  const cards = [
    { label: 'Temperature', value: weather.temperature ? `${weather.temperature}°C` : '--' },
    { label: 'Humidity', value: weather.humidity ? `${weather.humidity}%` : '--' },
    { label: 'Rainfall', value: weather.rainfall ? `${weather.rainfall} mm` : '--' },
    { label: 'Pressure', value: weather.pressure ? `${weather.pressure} hPa` : '--' },
    { label: 'Wind', value: weather.wind ? `${weather.wind} km/h` : '--' },
    { label: 'AQI', value: weather.aqi ?? '--' },
  ];

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {cards.map((card) => (
        <div key={card.label} className="rounded-3xl border border-slate-800 bg-slate-900/80 p-4">
          <p className="text-sm text-slate-400">{card.label}</p>
          <p className="mt-3 text-2xl font-semibold text-white">{card.value}</p>
        </div>
      ))}
    </div>
  );
}

export default ClimateSummary;
