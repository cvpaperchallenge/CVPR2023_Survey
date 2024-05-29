'use client'
import { Flex } from '@radix-ui/themes'

// import Footer from '@/components/footer'
// import Header from '@/components/header'
import Footer from '@/components/footer2'
import Header from '@/components/header2'
import { Suspense } from 'react'

export default function PaperListLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <Flex direction="column" align="stretch" style={{ minHeight: '100vh' }} gap="6">
      <Header/>
      <Flex direction="column" align="center" gap="0" p="3" style={{ flex: '1 0 auto' }}>
        <Suspense>
          {children}
        </Suspense>
      </Flex>
      <Footer/>
    </Flex>
    )
}