import type { Metadata } from "next";
import "./globals.css";
import { Inter } from "next/font/google";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { AppToaster } from "@/components/ui/toaster";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Body Measurement API - AI-Powered Size Recommendations",
  description: "B2B SaaS platform for AI-powered body measurements and clothing size recommendations",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} antialiased`} style={{ fontFamily: 'var(--font-inter)' }}>
        <QueryProvider>
          <div className="container mx-auto">
            <header className="flex items-center justify-between py-4">
              <h1 className="text-xl font-semibold text-foreground">Body Measurement Platform</h1>
              <span className="text-sm text-gray-500">AI-powered sizing</span>
            </header>
            <main>{children}</main>
          </div>
          <AppToaster />
        </QueryProvider>
      </body>
    </html>
  );
}
