const baseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api';

export async function getShapInsights() {
  const response = await fetch(`${baseUrl}/shap`);
  if (!response.ok) {
    throw new Error('Failed to load SHAP insights');
  }
  return response.json();
}
