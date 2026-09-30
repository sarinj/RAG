from __future__ import annotations

import re
from enum import StrEnum


class ChatIntent(StrEnum):
    SMALL_TALK = "small_talk"
    DOCUMENT_CATALOG = "document_catalog"
    DOCUMENT_QUESTION = "document_question"


_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9\s]")
_CATALOG_ASK = re.compile(r"\b(?:what|which|list|show|name)\b")
_DOCUMENT_WORD = re.compile(r"\b(?:documents?|files?|sources?)\b")
_CATALOG_SCOPE = re.compile(
    r"\b(?:can|could)\b.{0,15}\b(?:you|u|i|we)\b.{0,15}\b(?:access|see|use)\b"
    r"|\b(?:do|did)\s+(?:you|u|i|we)\s+have\b"
    r"|\bhave\s+access\b"
    r"|\b(?:available|accessible|loaded|selected|uploaded)\b"
    r"|\b(?:in|for)\s+(?:this|the)\s+(?:chat|conversation)\b"
)

_SMALL_TALK_ANSWERS = {
    "hi": "Hi! How can I help you with your documents?",
    "hello": "Hello! How can I help you with your documents?",
    "hey": "Hey! What would you like to know about your documents?",
    "hi there": "Hi! How can I help you with your documents?",
    "hello there": "Hello! How can I help you with your documents?",
    "hey there": "Hey! What would you like to know about your documents?",
    "good morning": "Good morning! What would you like to know about your documents?",
    "good afternoon": "Good afternoon! What would you like to know about your documents?",
    "good evening": "Good evening! What would you like to know about your documents?",
    "how are you": "I’m here and ready to help with your documents. What would you like to find out?",
    "hows it going": "I’m here and ready to help with your documents. What would you like to find out?",
    "how s it going": "I’m here and ready to help with your documents. What would you like to find out?",
    "what s up": "I’m here and ready to help with your documents. What would you like to find out?",
    "thanks": "You’re welcome! I can help with another question about your documents.",
    "thank you": "You’re welcome! I can help with another question about your documents.",
    "thanks so much": "You’re welcome! I can help with another question about your documents.",
    "bye": "Goodbye! Come back whenever you need help with your documents.",
    "goodbye": "Goodbye! Come back whenever you need help with your documents.",
}


def normalize_question(question: str) -> str:
    return " ".join(_NON_ALPHANUMERIC.sub(" ", question.casefold()).split())


def classify_intent(question: str) -> ChatIntent:
    normalized = normalize_question(question)
    if normalized in _SMALL_TALK_ANSWERS:
        return ChatIntent.SMALL_TALK
    if (
        _CATALOG_ASK.search(normalized)
        and _DOCUMENT_WORD.search(normalized)
        and _CATALOG_SCOPE.search(normalized)
    ):
        return ChatIntent.DOCUMENT_CATALOG
    # Keep document QA broad so summaries, explanations, comparisons, and questions that
    # connect ideas are not rejected by a narrow classifier.
    return ChatIntent.DOCUMENT_QUESTION


def small_talk_answer(question: str) -> str | None:
    return _SMALL_TALK_ANSWERS.get(normalize_question(question))
