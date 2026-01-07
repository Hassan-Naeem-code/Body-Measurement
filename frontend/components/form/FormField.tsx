"use client";

import { ReactNode } from "react";
import { FieldErrors, useFormContext } from "react-hook-form";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/cn";

export function FormField({
  name,
  label,
  children,
  className,
}: {
  name: string;
  label?: string;
  children: ReactNode;
  className?: string;
}) {
  const { formState } = useFormContext();
  const errors = formState.errors as FieldErrors<Record<string, unknown>>;
  const error = (errors?.[name] as { message?: string } | undefined)?.message;

  return (
    <div className={cn("space-y-1", className)}>
      {label && <Label htmlFor={name}>{label}</Label>}
      {children}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
