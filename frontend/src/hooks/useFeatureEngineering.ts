import { useQuery } from '@tanstack/react-query';
import { fetchFeatureEngineering } from '@/services/featureEngineering';

export function useFeatureEngineering() {
  return useQuery(['featureEngineering'], fetchFeatureEngineering, {
    staleTime: 1000 * 60 * 5,
  });
}
