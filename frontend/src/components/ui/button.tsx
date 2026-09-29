import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const buttonStyles = cva(
  "button inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "button-primary",
        secondary: "button-secondary",
        outline: "button-outline",
        ghost: "button-ghost",
        destructive: "button-destructive",
      },
      size: {
        default: "button-md",
        sm: "button-sm",
        lg: "button-lg",
        icon: "button-icon",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonStyles>;

export function Button({ className, variant, size, type = "button", ...props }: ButtonProps) {
  return <button className={cn(buttonStyles({ variant, size }), className)} type={type} {...props} />;
}
