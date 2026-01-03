import type { Metadata } from "next";
import "./globals.css";

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
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
