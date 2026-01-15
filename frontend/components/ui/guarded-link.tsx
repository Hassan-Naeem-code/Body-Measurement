'use client';

import { useRouter } from 'next/navigation';
import { useNavigationGuardContext } from '@/contexts/NavigationGuardContext';
import { ComponentProps, MouseEvent, forwardRef } from 'react';

interface GuardedLinkProps extends Omit<ComponentProps<'a'>, 'href'> {
  href: string;
  children: React.ReactNode;
}

/**
 * A link component that respects the navigation guard context.
 * Shows a confirmation dialog if navigation is blocked.
 */
export const GuardedLink = forwardRef<HTMLAnchorElement, GuardedLinkProps>(
  function GuardedLink({ href, children, onClick, ...props }, ref) {
    const router = useRouter();
    const { tryNavigate } = useNavigationGuardContext();

    const handleClick = (e: MouseEvent<HTMLAnchorElement>) => {
      e.preventDefault();

      // Call the original onClick if provided
      if (onClick) {
        onClick(e);
      }

      // Try to navigate - this will show confirmation if blocked
      tryNavigate(() => {
        router.push(href);
      });
    };

    return (
      <a ref={ref} href={href} onClick={handleClick} {...props}>
        {children}
      </a>
    );
  }
);
