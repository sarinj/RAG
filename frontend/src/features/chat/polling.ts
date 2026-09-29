export function hasPendingAssistant(messages: readonly { role: string; status: string }[]) {
  return messages.some((message) => message.role === "assistant" && message.status === "pending");
}
