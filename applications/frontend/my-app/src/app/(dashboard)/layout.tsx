'use client'
import Footer from '@/components/footer'
import Header from '@/components/header'

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
      <div className="flex flex-col grow w-full">{children}</div>
      <div className="h-44 w-full flex-none border-t bg-[var(--teal-4)] dark:bg-card">
        <Footer />
      </div>
    </div>
  )
}
