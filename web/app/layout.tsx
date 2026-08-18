import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MemoryVerse — Your AI-Powered Memory Vault",
  description:
    "Capture, preserve, and relive your most precious moments with AI-powered organization, smart timelines, and beautiful vaults.",
  keywords: ["memories", "photos", "AI", "journal", "timeline", "vault"],
  openGraph: {
    title: "MemoryVerse — Your AI-Powered Memory Vault",
    description: "Capture and relive your most precious moments.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
