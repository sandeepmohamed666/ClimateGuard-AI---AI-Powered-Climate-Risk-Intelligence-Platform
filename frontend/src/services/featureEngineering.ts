const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api';

export async function fetchFeatureEngineering() {
  const response = await fetch(`${baseUrl}/feature-engineering`);
  if (!response.ok) {
    throw new Error('Failed to load feature engineering data');
  }
  return response.json();
}
