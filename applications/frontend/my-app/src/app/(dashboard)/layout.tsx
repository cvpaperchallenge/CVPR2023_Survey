'use client'
import Footer from '@/components/footer'
import Header from '@/components/header'

export default function PaperListLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex min-h-screen w-screen flex-col items-center gap-8 ">
      <div className="h-24 w-full flex-none border-b dark:bg-card bg-teal-100">
        <Header />
      </div>
      <div className="flex-1 grow">{children}</div>
      <div className="h-44 w-full flex-none border-t dark:bg-card bg-teal-100">
        <Footer />
      </div>
    </div>
  )
}
