import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Amazon Launch Viability | Product Evaluation Console',
  description: 'Evaluate prospective Amazon products for launch viability using multimodal AI analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-text antialiased">
        {children}
      </body>
    </html>
  )
}
