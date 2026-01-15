'use client';

import { useEffect, useCallback, useRef } from 'react';

interface UseNavigationGuardOptions {
  /** Whether navigation should be blocked */
  shouldBlock: boolean;
  /** Message to show in the confirmation dialog */
  message?: string;
  /** Callback when user confirms they want to leave */
  onConfirmLeave?: () => void;
}

/**
 * Hook to prevent accidental navigation away from a page
 * Handles both browser navigation (refresh, close tab) and in-app navigation
 */
export function useNavigationGuard({
  shouldBlock,
  message = 'You have unsaved changes. Are you sure you want to leave?',
  onConfirmLeave,
}: UseNavigationGuardOptions) {
  const shouldBlockRef = useRef(shouldBlock);
  shouldBlockRef.current = shouldBlock;

  // Handle browser navigation (refresh, close tab, back button)
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (shouldBlockRef.current) {
        e.preventDefault();
        e.returnValue = message;
        return message;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [message]);

  // Function to check if navigation should proceed
  const confirmNavigation = useCallback((): boolean => {
    if (!shouldBlockRef.current) {
      return true;
    }

    const confirmed = window.confirm(message);
    if (confirmed && onConfirmLeave) {
      onConfirmLeave();
    }
    return confirmed;
  }, [message, onConfirmLeave]);

  return {
    confirmNavigation,
    isBlocking: shouldBlock,
  };
}
