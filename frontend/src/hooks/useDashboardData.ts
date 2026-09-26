import { useQuery } from '@tanstack/react-query';
import { fetchDashboard } from '@/services/api';

export function useDashboardData() {
  return useQuery(['dashboardData'], fetchDashboard, {
    staleTime: 1000 * 60,
  });
}
