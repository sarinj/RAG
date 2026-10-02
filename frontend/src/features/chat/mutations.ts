import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createConversation } from "./api";
import { sendMessageStream, type StreamMessageInput } from "./streaming";
import { chatKeys } from "./queries";

export function useCreateConversation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: createConversation,
    onSuccess: async (conversation) => {
      queryClient.setQueryData(chatKeys.conversation(conversation.id), conversation);
      await queryClient.invalidateQueries({ queryKey: chatKeys.conversations() });
    },
  });
}

export function useSendMessage(conversationId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: StreamMessageInput) => sendMessageStream(conversationId, input),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: chatKeys.messages(conversationId) }),
        queryClient.invalidateQueries({ queryKey: chatKeys.conversation(conversationId) }),
        queryClient.invalidateQueries({ queryKey: chatKeys.conversations() }),
      ]);
    },
  });
}
