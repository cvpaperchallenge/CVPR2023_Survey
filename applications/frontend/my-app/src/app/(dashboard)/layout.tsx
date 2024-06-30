'use client'
import Footer from '@/components/footer'
import Header from '@/components/header'
import { Suspense } from 'react'

export default function PaperListLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex min-h-screen w-screen flex-col items-center">
      <div className="h-24 w-full flex-none border-b bg-[var(--teal-4)] dark:bg-card">
        <Header />
      </div>
      <div className="flex w-full grow flex-col">
        <Suspense>
          {children}
        </Suspense>
      </div>
      <div className="h-44 w-full flex-none border-t bg-[var(--teal-4)] dark:bg-card">
        <Footer />
      </div>
    </div>
  )
}
