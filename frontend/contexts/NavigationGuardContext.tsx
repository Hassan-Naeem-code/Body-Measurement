'use client';

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';

interface NavigationGuardContextType {
  /** Register a guard that should block navigation */
  setNavigationBlocked: (blocked: boolean, message?: string) => void;
  /** Check if navigation is currently blocked */
  isNavigationBlocked: boolean;
  /** The message to show when navigation is blocked */
  blockMessage: string;
  /** Try to navigate - returns true if allowed, false if blocked and user cancelled */
  tryNavigate: (callback: () => void) => boolean;
}

const NavigationGuardContext = createContext<NavigationGuardContextType | null>(null);

const DEFAULT_MESSAGE = 'Your image is still being processed. If you leave now, you will lose your results. Are you sure you want to leave?';

export function NavigationGuardProvider({ children }: { children: React.ReactNode }) {
  const [isBlocked, setIsBlocked] = useState(false);
  const [message, setMessage] = useState(DEFAULT_MESSAGE);
  const isBlockedRef = useRef(isBlocked);
  const messageRef = useRef(message);

  // Keep refs in sync
  useEffect(() => {
    isBlockedRef.current = isBlocked;
    messageRef.current = message;
  }, [isBlocked, message]);

  const setNavigationBlocked = useCallback((blocked: boolean, customMessage?: string) => {
    setIsBlocked(blocked);
    if (customMessage) {
      setMessage(customMessage);
    } else {
      setMessage(DEFAULT_MESSAGE);
    }
  }, []);

  const tryNavigate = useCallback((callback: () => void): boolean => {
    if (!isBlockedRef.current) {
      callback();
      return true;
    }

    const confirmed = window.confirm(messageRef.current);
    if (confirmed) {
      callback();
      return true;
    }
    return false;
  }, []);

  // Handle browser back/forward navigation
  useEffect(() => {
    const handlePopState = (e: PopStateEvent) => {
      if (isBlockedRef.current) {
        const confirmed = window.confirm(messageRef.current);
        if (!confirmed) {
          // Push the current state back to prevent navigation
          window.history.pushState(null, '', window.location.href);
        }
      }
    };

    // Push initial state
    window.history.pushState(null, '', window.location.href);
    window.addEventListener('popstate', handlePopState);

    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  // Handle browser refresh/close (beforeunload)
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isBlockedRef.current) {
        e.preventDefault();
        e.returnValue = messageRef.current;
        return messageRef.current;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  return (
    <NavigationGuardContext.Provider
      value={{
        setNavigationBlocked,
        isNavigationBlocked: isBlocked,
        blockMessage: message,
        tryNavigate,
      }}
    >
      {children}
    </NavigationGuardContext.Provider>
  );
}

export function useNavigationGuardContext() {
  const context = useContext(NavigationGuardContext);
  if (!context) {
    throw new Error('useNavigationGuardContext must be used within NavigationGuardProvider');
  }
  return context;
}
