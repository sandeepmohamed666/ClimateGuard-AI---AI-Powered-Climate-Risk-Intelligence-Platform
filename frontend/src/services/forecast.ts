const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api';

export async function fetchForecast() {
  const response = await fetch(`${baseUrl}/forecast`);
  if (!response.ok) {
    throw new Error('Failed to load forecast data');
  }
  return response.json();
}
