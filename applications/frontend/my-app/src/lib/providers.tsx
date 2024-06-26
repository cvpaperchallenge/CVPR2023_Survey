'use client'

import { ThemeProvider as NextThemesProvider } from 'next-themes'
import { type ThemeProviderProps } from 'next-themes/dist/types'
import { createContext, useContext, useState } from 'react';

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>
}

export interface SimplePaperState {
  paperId: number,
  paperTitle: string,
}

export interface SimplePaperStates {
  previousPaperState: SimplePaperState | null,
  currentPaperState: SimplePaperState,
  nextPaperState: SimplePaperState | null,
}

export interface SimplePaperContextType {
  simplePaperStates: SimplePaperStates,
  setSimplePaperStates: (state: SimplePaperStates) => void,
}

const SimplePaperContext = createContext<SimplePaperContextType | null>(null);

export const SimplePaperStateProvider = ({ children }: { children: React.ReactNode }) => {
  const [simplePaperStates, setSimplePaperStates] = useState<SimplePaperStates>({
    previousPaperState: null,
    currentPaperState: { paperId: 1, paperTitle: 'Paper 1' },
    nextPaperState: { paperId: 2, paperTitle: 'Paper 2' },
  });

  return (
    <SimplePaperContext.Provider value={{ simplePaperStates, setSimplePaperStates }}>
      {children}
    </SimplePaperContext.Provider>
  );
}

export const useSimplePaperContext = (): SimplePaperContextType => {
  const context = useContext(SimplePaperContext);
  if (!context) {
    throw new Error('useSimplePaperContext must be used within a PaperIdProvider');
  }
  return context;
};