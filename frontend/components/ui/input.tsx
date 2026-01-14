"use client";

import { forwardRef, InputHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

export type InputProps = InputHTMLAttributes<HTMLInputElement>;

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { className, type, ...props },
  ref
) {
  return (
    <input
      ref={ref}
      type={type}
      className={cn(
        // Base styles
        "flex h-11 w-full rounded-lg border border-input bg-card px-4 py-2 text-sm",
        // Text colors
        "text-foreground placeholder:text-muted-foreground",
        // Focus styles
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        // Transition
        "transition-colors duration-200",
        // Hover
        "hover:border-border-hover",
        // Disabled
        "disabled:cursor-not-allowed disabled:opacity-50 disabled:bg-muted",
        // File input specific
        type === "file" &&
          "cursor-pointer file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-primary hover:file:text-primary-hover",
        className
      )}
      {...props}
    />
  );
});
