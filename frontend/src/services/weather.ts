import { fetchDashboard } from './api';

export async function getCurrentWeather() {
  return fetchDashboard().then((data) => data.weather);
}
