import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Link from 'next/link';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'ER Pulse | Healthcare Analytics',
  description: 'Real-time emergency care insights and multi-dimensional anomaly detection',
};

const navLinks = [
  { href: '/', label: 'Dashboard', icon: '📊' },
  { href: '/predict', label: 'Predict', icon: '🔮' },
  { href: '/forecast', label: 'Forecast', icon: '📈' },
  { href: '/models', label: 'Models', icon: '🤖' },
  { href: '/alerts', label: 'Alerts', icon: '🚨' },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-950 text-white min-h-screen`}>
        {/* Navigation */}
        <nav className="sticky top-0 z-50 backdrop-blur-xl bg-gray-950/80 border-b border-gray-800/50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6">
            <div className="flex items-center justify-between h-16">
              {/* Logo */}
              <Link href="/" className="flex items-center gap-3 group">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-blue-500/20 group-hover:shadow-blue-500/40 transition-shadow">
                  <span className="text-xl">⚡</span>
                </div>
                <div>
                  <span className="font-bold text-lg text-white">ER Pulse</span>
                  <span className="hidden sm:block text-xs text-gray-500">Healthcare Analytics</span>
                </div>
              </Link>

              {/* Navigation Links */}
              <div className="flex items-center gap-1">
                {navLinks.map((link) => (
                  <Link
                    key={link.href}
                    href={link.href}
                    className="px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-gray-800/50 transition-all duration-200"
                  >
                    <span className="hidden sm:inline mr-1.5">{link.icon}</span>
                    {link.label}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-800/50 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-500">
              <div className="flex items-center gap-2">
                <span className="text-blue-400">⚡</span>
                <span>ER Pulse - Real-time Emergency Care Insights</span>
              </div>
              <div>
                Built with Next.js & FastAPI
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
