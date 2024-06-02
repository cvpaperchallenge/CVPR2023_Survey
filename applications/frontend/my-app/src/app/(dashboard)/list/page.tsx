'use client'

import { useTheme } from 'next-themes'
import * as React from 'react'

import { Button } from '@/components/ui/button'

export default function ModeToggle() {
  const { setTheme } = useTheme()

  return (
    <Button onClick={() => setTheme('dark')} size="default" variant="outline">
      Dark
      <span className="sr-only">Toggle theme</span>
    </Button>
  )
}
