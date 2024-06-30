'use client'
import { Inter as FontSans } from 'next/font/google'
import './globals.css'
import { Toaster } from 'sonner'

import { ThemeProvider } from '@/lib/providers'
import { cn } from '@/lib/utils'

const fontSans = FontSans({
  subsets: ['latin'],
  variable: '--font-sans',
})

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <title>CVPR/ICCV 2023 Summary</title>
      <meta
        content="Automated summaries of papers accepted at CVPR 2023 and ICCV 2023 using OpenAI's language model."
        name="description"
      />
      <body
        className={cn(
          'min-h-screen bg-background font-sans antialiased',
          fontSans.variable
        )}
      >
        <Toaster richColors />
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          disableTransitionOnChange
          enableSystem
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
