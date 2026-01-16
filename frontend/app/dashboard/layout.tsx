'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { authHelpers } from '@/lib/auth';
import type { Brand } from '@/lib/types';
import { NavigationGuardProvider } from '@/contexts/NavigationGuardContext';
import { GuardedLink } from '@/components/ui/guarded-link';
import {
  LayoutDashboard,
  ImagePlus,
  ShoppingBag,
  LineChart,
  KeySquare,
  Menu,
  X,
  LogOut,
  ChevronRight,
  Sparkles,
  Bell,
  Settings,
  History,
  Code2,
} from 'lucide-react';

interface NavItem {
  name: string;
  href: string;
  Icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

const navItems: NavItem[] = [
  { name: 'Dashboard', href: '/dashboard', Icon: LayoutDashboard },
  { name: 'Upload Image', href: '/dashboard/upload', Icon: ImagePlus },
  { name: 'Products', href: '/dashboard/products', Icon: ShoppingBag },
  { name: 'Analytics', href: '/dashboard/analytics', Icon: LineChart },
  { name: 'History', href: '/dashboard/history', Icon: History, badge: 'New' },
  { name: 'SDK & APIs', href: '/dashboard/sdk', Icon: Code2 },
  { name: 'API Keys', href: '/dashboard/api-keys', Icon: KeySquare },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [brand, setBrand] = useState<Brand | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    if (!authHelpers.isAuthenticated()) {
      router.push('/auth/login');
      return;
    }
    const brandData = authHelpers.getBrand();
    setBrand(brandData);
  }, [router]);

  useEffect(() => {
    setIsMenuOpen(false);
  }, [pathname]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsMenuOpen(false);
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  useEffect(() => {
    if (isMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isMenuOpen]);

  const handleLogout = useCallback(() => {
    authHelpers.logout();
    router.push('/');
  }, [router]);

  const isActiveRoute = useCallback(
    (href: string) => {
      if (href === '/dashboard') {
        return pathname === '/dashboard';
      }
      return pathname.startsWith(href);
    },
    [pathname]
  );

  if (!isMounted || !brand) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-500 text-sm">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <NavigationGuardProvider>
    <div className="min-h-screen bg-gray-50">
      {/* Desktop Sidebar */}
      <aside className="fixed inset-y-0 left-0 w-64 bg-white border-r border-gray-200 hidden lg:flex lg:flex-col z-40">
        {/* Logo */}
        <div className="flex items-center gap-3 p-5 border-b border-gray-200">
          <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center shadow-sm">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-base font-bold text-gray-900 truncate">FitWhisperer</h1>
            <p className="text-xs text-gray-500 truncate">AI-Powered Sizing</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const isActive = isActiveRoute(item.href);
            return (
              <GuardedLink
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-indigo-50 text-indigo-700'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
              >
                <item.Icon className="w-5 h-5 flex-shrink-0" />
                <span className="flex-1 truncate">{item.name}</span>
                {item.badge && (
                  <span className="px-2 py-0.5 text-xs font-medium bg-indigo-100 text-indigo-700 rounded-full">
                    {item.badge}
                  </span>
                )}
                {isActive && <ChevronRight className="w-4 h-4 opacity-50" />}
              </GuardedLink>
            );
          })}
        </nav>

        {/* User Section */}
        <div className="p-3 border-t border-gray-200 space-y-2">
          <div className="flex items-center gap-3 p-2 rounded-lg bg-gray-50">
            <div className="w-10 h-10 rounded-full bg-indigo-600 flex items-center justify-center">
              <span className="text-white font-semibold text-sm">
                {brand.name.charAt(0).toUpperCase()}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">{brand.name}</p>
              <p className="text-xs text-gray-500 truncate">{brand.email}</p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-red-600 hover:bg-red-50 transition-colors"
          >
            <LogOut className="w-5 h-5" />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>

      {/* Mobile Header */}
      <header className="lg:hidden fixed top-0 left-0 right-0 h-16 bg-white border-b border-gray-200 z-40">
        <div className="flex items-center justify-between h-full px-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-gray-900">FitWhisperer</span>
          </div>
          <div className="flex items-center gap-2">
            <button className="w-10 h-10 rounded-lg flex items-center justify-center text-gray-500 hover:bg-gray-100 hover:text-gray-900 transition-colors">
              <Bell className="w-5 h-5" />
            </button>
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="w-10 h-10 rounded-lg flex items-center justify-center text-gray-900 hover:bg-gray-100 transition-colors"
              aria-label={isMenuOpen ? 'Close menu' : 'Open menu'}
            >
              {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Menu Overlay */}
      <div
        className={`lg:hidden fixed inset-0 bg-black/50 z-50 transition-opacity duration-200 ${
          isMenuOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => setIsMenuOpen(false)}
      />

      {/* Mobile Menu */}
      <div
        className={`lg:hidden fixed top-16 right-0 bottom-0 w-72 max-w-[85vw] bg-white border-l border-gray-200 z-50 transform transition-transform duration-300 ${
          isMenuOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="flex flex-col h-full">
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-indigo-600 flex items-center justify-center">
                <span className="text-white font-semibold">
                  {brand.name.charAt(0).toUpperCase()}
                </span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium text-gray-900 truncate">{brand.name}</p>
                <p className="text-sm text-gray-500 truncate">{brand.email}</p>
              </div>
            </div>
          </div>

          <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
            {navItems.map((item) => {
              const isActive = isActiveRoute(item.href);
              return (
                <GuardedLink
                  key={item.href}
                  href={item.href}
                  onClick={() => setIsMenuOpen(false)}
                  className={`flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-indigo-50 text-indigo-700'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  <item.Icon className="w-5 h-5 flex-shrink-0" />
                  <span className="flex-1">{item.name}</span>
                  {item.badge && (
                    <span className="px-2 py-0.5 text-xs font-medium bg-indigo-100 text-indigo-700 rounded-full">
                      {item.badge}
                    </span>
                  )}
                </GuardedLink>
              );
            })}
          </nav>

          <div className="p-3 border-t border-gray-200 space-y-1">
            <GuardedLink
              href="/dashboard/settings"
              onClick={() => setIsMenuOpen(false)}
              className="flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-100"
            >
              <Settings className="w-5 h-5" />
              <span>Settings</span>
            </GuardedLink>
            <button
              onClick={handleLogout}
              className="w-full flex items-center gap-3 px-3 py-3 rounded-lg text-sm font-medium text-red-600 hover:bg-red-50"
            >
              <LogOut className="w-5 h-5" />
              <span>Sign Out</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="lg:ml-64 pt-16 lg:pt-0 min-h-screen">
        <div className="p-4 sm:p-6 lg:p-8 max-w-screen-2xl mx-auto">
          {children}
        </div>
      </main>
    </div>
    </NavigationGuardProvider>
  );
}
