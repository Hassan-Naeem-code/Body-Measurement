'use client';

import { useEffect, useState } from 'react';
import { offlineManager } from '@/lib/offline';

export function OfflineBanner() {
  const [isOffline, setIsOffline] = useState(false);

  useEffect(() => {
    offlineManager.init();
    setIsOffline(offlineManager.isOffline);

    return offlineManager.subscribe(setIsOffline);
  }, []);

  if (!isOffline) return null;

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-yellow-500 text-yellow-950 text-center py-2 px-4 text-sm font-medium shadow-md">
      You are offline. Some features may be unavailable.
    </div>
  );
}
