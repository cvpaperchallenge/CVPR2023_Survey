'use client'
import { Theme } from '@radix-ui/themes';
import { Inter } from "next/font/google";
import { Toaster } from 'sonner';
import "@/app/globals.css";
import '@radix-ui/themes/styles.css';

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <title>CVPR/ICCV 2023 Summary</title>
      <meta name="description" content="Automated summaries of papers accepted at CVPR 2023 and ICCV 2023 using OpenAI's language model." />
      <body className={inter.className}>
        <Toaster richColors />
          <Theme
            accentColor="teal"
            grayColor="gray"
            appearance="dark"
            panelBackground="translucent"
          >
            {children}
          </Theme>
      </body>
    </html>
  );
}
