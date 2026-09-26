import { useQuery } from '@tanstack/react-query';
import { fetchForecast } from '@/services/forecast';

export function useForecast() {
  return useQuery(['forecastData'], fetchForecast, {
    staleTime: 1000 * 60 * 5,
  });
}
