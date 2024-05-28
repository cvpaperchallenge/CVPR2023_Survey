'use client'
import {Box, Flex, Text, IconButton, Separator} from '@radix-ui/themes'

import Footer from '@/components/footer2'
import Header from '@/components/header2'

export default function PaperListLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const year = new Date().getFullYear()
  return (
    <Flex direction="column" align="stretch" style={{ minHeight: '100vh' }} gap="6">
      <Header/>
      <Flex direction="column" align="center" gap="0" p="3" style={{ flex: '1 0 auto' }}>
        {children}
      </Flex>
      <Footer/>
    </Flex>
    )
}