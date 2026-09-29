import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number) {
  if (bytes < 1_000) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1_000;
  let unit = units[0];
  for (let index = 0; value >= 1_000 && index < units.length - 1; index += 1) {
    value /= 1_000;
    unit = units[index + 1];
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${unit}`;
}

export function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function formatRelativeDate(value: string) {
  const elapsed = Date.now() - new Date(value).getTime();
  const minutes = Math.floor(elapsed / 60_000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(value));
}

export function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
