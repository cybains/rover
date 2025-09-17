import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { cn } from "@/lib/utils";

export const Sheet = DialogPrimitive.Root;
export const SheetTrigger = DialogPrimitive.Trigger;

export const SheetContent = React.forwardRef<React.ElementRef<typeof DialogPrimitive.Content>, DialogPrimitive.DialogContentProps & { side?: "left" | "right" }>(
  ({ className, side = "right", ...props }, ref) => (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/20" />
      <DialogPrimitive.Content
        ref={ref}
        className={cn(
          "fixed z-50 h-full w-80 bg-white shadow-lg",
          side === "right" ? "right-0 top-0" : "left-0 top-0",
          className
        )}
        {...props}
      />
    </DialogPrimitive.Portal>
  )
);
SheetContent.displayName = "SheetContent";

export function SheetHeader(props: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("border-b px-4 py-3", props.className)} {...props} />;
}
export function SheetTitle(props: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h2 className={cn("text-base font-medium", props.className)} {...props} />;
}