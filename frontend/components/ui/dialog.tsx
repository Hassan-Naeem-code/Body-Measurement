"use client";

import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { cn } from "@/lib/cn";

export const Dialog = DialogPrimitive.Root;
export const DialogTrigger = DialogPrimitive.Trigger;
export const DialogPortal = DialogPrimitive.Portal;

export function DialogOverlay({ className, ...props }: DialogPrimitive.DialogOverlayProps) {
  return (
    <DialogPrimitive.Overlay
      className={cn("fixed inset-0 bg-black/40 backdrop-blur-sm", className)}
      {...props}
    />
  );
}

export function DialogContent({ className, ...props }: DialogPrimitive.DialogContentProps) {
  return (
    <DialogPortal>
      <DialogOverlay />
      <DialogPrimitive.Content
        className={cn(
          "fixed left-1/2 top-1/2 w-[95vw] max-w-lg -translate-x-1/2 -translate-y-1/2",
          "rounded-lg border bg-background p-4 shadow-lg",
          className
        )}
        {...props}
      />
    </DialogPortal>
  );
}

export function DialogHeader({ title }: { title: string }) {
  return (
    <div className="mb-2 flex items-center justify-between">
      <h3 className="text-lg font-semibold">{title}</h3>
      <DialogPrimitive.Close asChild>
        <button className="rounded p-1 hover:bg-muted" aria-label="Close">
          <X size={16} />
        </button>
      </DialogPrimitive.Close>
    </div>
  );
}
