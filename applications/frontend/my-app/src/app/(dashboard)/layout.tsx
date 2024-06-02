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
    <div className='flex flex-col items-center min-h-screen w-screen gap-8'>
      <div className='flex-none w-full h-24'>
        <Header/>
      </div>
      <div className='grow flex-1'>
          {children}
      </div>
      <div className='flex-none w-full h-44'>
        <Footer/>
      </div>
    </div>
    )
}