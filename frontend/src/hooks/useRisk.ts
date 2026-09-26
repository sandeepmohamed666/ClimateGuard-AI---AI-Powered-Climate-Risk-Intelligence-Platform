import { useQuery } from '@tanstack/react-query';
import { getRiskSummary } from '@/services/risk';

export function useRisk() {
  return useQuery(['riskSummary'], getRiskSummary, {
    staleTime: 1000 * 60,
  });
}
