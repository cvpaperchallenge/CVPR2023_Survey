'use client'
import { Theme } from '@radix-ui/themes';
import '@radix-ui/themes/styles.css';
import { Inter } from "next/font/google";
import { Toaster } from 'sonner';
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CVPR/ICCV 2023 Summary",
  description: "Automated summaries of papers accepted at CVPR 2023 and ICCV 2023 using OpenAI's language model.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Toaster />
          <Theme accentColor="teal" grayColor="gray" appearance="dark">
            {children}
          </Theme>
      </body>
    </html>
  );
}
