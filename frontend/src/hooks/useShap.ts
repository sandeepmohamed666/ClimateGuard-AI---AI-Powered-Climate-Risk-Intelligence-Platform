import { useQuery } from '@tanstack/react-query';
import { getShapInsights } from '@/services/shap';

export function useShap() {
  return useQuery(['shapInsights'], getShapInsights, {
    staleTime: 1000 * 60,
  });
}
