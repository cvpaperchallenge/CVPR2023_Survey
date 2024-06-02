'use client'

import { useTheme } from 'next-themes'
import * as React from 'react'

import { Button } from '@/components/ui/button'

export default function ModeToggle() {
  const { setTheme } = useTheme()

  return (
    <Button onClick={() => setTheme('light')} size="default" variant="outline">
      light
      <span className="sr-only">Toggle theme</span>
    </Button>
  )
}
