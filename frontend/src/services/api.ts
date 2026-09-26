const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api';

export async function fetchDashboard() {
  const response = await fetch(`${baseUrl}/dashboard`);
  if (!response.ok) {
    throw new Error('Failed to load dashboard data');
  }
  return response.json();
}

export async function fetchWeather() {
  const response = await fetch(`${baseUrl}/weather/current`);
  if (!response.ok) {
    throw new Error('Failed to load weather data');
  }
  return response.json();
}

export async function fetchFeatureEngineering() {
  const response = await fetch(`${baseUrl}/feature-engineering`);
  if (!response.ok) {
    throw new Error('Failed to load feature engineering data');
  }
  return response.json();
}
