import { createContext, useContext } from 'react';

export const DashboardDataContext = createContext(null);
export function useDashboardData() {
  const data = useContext(DashboardDataContext);
  if (!data) throw new Error('Evaluation data has not loaded.');
  return data;
}
