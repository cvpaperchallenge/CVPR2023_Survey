'use client'

import * as React from 'react'
import { useTheme } from 'next-themes'

import { Button } from '@/components/ui/button'

export default function ModeToggle() {
  const { setTheme } = useTheme()

  return (
    <Button variant="outline" size="default" onClick={() => setTheme('dark')}>
      Dark
      <span className="sr-only">Toggle theme</span>
    </Button>
  )
}
