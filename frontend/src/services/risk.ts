const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api';

export async function getRiskSummary() {
  const response = await fetch(`${baseUrl}/risk`);
  if (!response.ok) {
    throw new Error('Failed to load risk data');
  }
  return response.json();
}
