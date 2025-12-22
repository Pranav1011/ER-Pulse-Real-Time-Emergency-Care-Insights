import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Healthcare Analytics',
  description: 'Multi-dimensional anomaly detection for healthcare operations',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-900 text-white`}>
        <nav className="border-b border-gray-800 px-6 py-4">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center gap-2">
              <span className="text-2xl">🏥</span>
              <span className="font-bold text-xl">Healthcare Analytics</span>
            </div>
            <div className="flex gap-6">
              <a href="/" className="hover:text-blue-400 transition">Dashboard</a>
              <a href="/alerts" className="hover:text-blue-400 transition">Alerts</a>
              <a href="/predict" className="hover:text-blue-400 transition">Predict</a>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-6 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}
