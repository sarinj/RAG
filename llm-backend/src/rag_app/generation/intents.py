from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from rag_app.models import QueryRoute

MAX_RETRIEVAL_HOPS = 3


class ChatIntent(StrEnum):
    CONVERSATION = "conversation"
    SOURCE_CATALOG = "source_catalog"
    SOURCE_QUESTION = "source_question"
    ANSWER_REVISION = "answer_revision"


class ChatPlan(BaseModel):
    """Validated interpretation of a message, produced by the configured LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: ChatIntent
    query: StrictStr = Field(min_length=1, max_length=12_000)
    max_words: StrictInt | None = Field(ge=1, le=10_000)
    definition_subject: StrictStr | None = Field(min_length=1, max_length=200)
    topic_message_index: StrictInt | None = Field(ge=0)
    direct_answer: StrictStr | None = Field(min_length=1, max_length=1000)
    previous_intent: ChatIntent | None = None
    query_route: QueryRoute = QueryRoute.LOCAL_SEARCH
    subqueries: list[StrictStr] = Field(default_factory=list, max_length=MAX_RETRIEVAL_HOPS)
    document_ids: list[UUID] = Field(default_factory=lambda: list[UUID](), max_length=100)
    shorten: StrictBool = False
    retrieval_queries: list[StrictStr] = Field(default_factory=list, max_length=4)
    subquestions: list[StrictStr] = Field(default_factory=list, max_length=4)
    reason: StrictStr | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_preferences(self) -> ChatPlan:
        if self.intent is ChatIntent.CONVERSATION:
            if self.direct_answer is None:
                raise ValueError("Small talk requires a conversational response")
        else:
            # Some models draft a factual answer while planning. Discard it;
            # document answers must be generated from fresh, scoped evidence.
            self.direct_answer = None
        if any(
            not query.strip() or len(query) > 1000
            for query in [
                *self.subqueries,
                *self.retrieval_queries,
                *self.subquestions,
            ]
        ):
            raise ValueError("Subqueries must be nonempty and at most 1000 characters")
        if len(self.document_ids) != len(set(self.document_ids)):
            raise ValueError("Document selection must be unique")
        if self.intent is not ChatIntent.ANSWER_REVISION:
            self.shorten = False
        if self.intent is not ChatIntent.SOURCE_QUESTION:
            self.definition_subject = None
        return self


# High-confidence language signals augment the multilingual planner. Bare "all" alone
# is deliberately ambiguous ("all I need is the refund date", "all else equal").
_SUMMARY_SIGNALS = (
    r"\b(?:summari[sz]e|summary|overview)\b.*"
    r"\b(?:entire|whole|all sections|everything|complete document)\b",
    r"\b(?:entire|whole)\b.*\b(?:summary|overview)\b",
    r"สรุป.*(?:ทั้งหมด|ทั้งเอกสาร|ทุกส่วน)",
    r"(?:总结|總結|概述).*(?:全部|所有|整个|整個|全文)",
    r"(?:全文|全体|すべて).*(?:要約|まとめ)",
    r"\b(?:resumir|resumen|résumer|résumé)\b.*\b(?:todo|todos|entero|entera|tout|entier)\b",
    r"\b(?:zusammenfassung|zusammenfassen)\b.*\b(?:gesamte|ganze|alles)\b",
)
_LIST_SIGNALS = (
    r"\b(?:list|extract|find|show|identify|enumerate|give|count)\b.*"
    r"\b(?:all|every|each|complete list)\b",
    r"\b(?:all mentions|every occurrence|complete list|all occurrences)\b",
    r"(?:แสดง|รายการ|ดึง|ค้นหา|นับ).*(?:ทั้งหมด|ทุก|ครบ)",
    r"(?:列出|提取|抽取|统计|統計).*(?:所有|全部|每个|每個)",
    r"(?:すべて|全部|全件).*(?:抽出|一覧|列挙|数)",
    r"\b(?:lista|listar|extraer|enumerar|lister|extraire)\b.*\b(?:todos|todas|cada|tous|toutes|chaque)\b",
    r"\b(?:auflisten|extrahieren|zählen)\b.*\b(?:alle|jede|jeden)\b",
)
_CATALOG_SIGNAL = re.compile(
    r"\b(?:list|show|which|what)\b.*(?:"
    r"\b(?:files|filenames|documents|sources)\b.*"
    r"\b(?:accessible|available|selected|uploaded|can (?:access|read))\b"
    r"|\b(?:accessible|available|selected|uploaded)\b.*\b(?:files|filenames|documents|sources)\b)"
)


def exhaustive_route(question: str) -> QueryRoute | None:
    text = unicodedata.normalize("NFKC", question).casefold()
    if _CATALOG_SIGNAL.search(text):
        return None
    if any(re.search(pattern, text) for pattern in _SUMMARY_SIGNALS):
        return QueryRoute.GLOBAL_SUMMARY
    if any(re.search(pattern, text) for pattern in _LIST_SIGNALS):
        return QueryRoute.AGGREGATION_LIST
    return None


_SOCIAL_MESSAGES = re.compile(
    r"^(?:hi|hello|hey|hello there|hi there|good morning|good afternoon|good evening|"
    r"thanks|thank you|thanks a lot|thank you very much|thanks for your help|"
    r"thank you for your help|bye|goodbye|see you|see you later|"
    r"สวัสดี(?:ครับ|ค่ะ|คะ)?|ขอบคุณ(?:ครับ|ค่ะ|คะ|มาก)?|ลาก่อน|"
    r"你好|您好|谢谢|謝謝|再见|再見|こんにちは|ありがとう|さようなら|"
    r"bonjour|salut|merci|merci beaucoup|au revoir|hola|gracias|muchas gracias|adiós|"
    r"hallo|danke|vielen dank|tschüss|ciao|grazie|arrivederci|"
    r"안녕하세요|감사합니다|안녕|مرحبا|شكرا|مع السلامة)$"
)


def social_message_only(question: str) -> bool:
    """Conservative bypass: unknown phrasing stays with scoped document QA."""
    text = unicodedata.normalize("NFKC", question).casefold()
    text = "".join(char for char in text if not unicodedata.category(char).startswith(("P", "S")))
    text = " ".join(text.split())
    return bool(_SOCIAL_MESSAGES.fullmatch(text))


def catalog_message_only(question: str) -> bool:
    """Direct catalog replies must concern only source availability/names, not their facts."""
    text = unicodedata.normalize("NFKC", question).casefold()
    text = "".join(char for char in text if not unicodedata.category(char).startswith(("P", "S")))
    text = " ".join(text.split())
    subjects = {
        "documents",
        "document",
        "files",
        "file",
        "sources",
        "source",
        "filenames",
        "fichiers",
        "archivos",
        "documentos",
        "quellen",
        "dateien",
        "dokumente",
    }
    words = set(text.split())
    catalog_words = subjects | set(
        "which what are is the a all every complete list show give name names filename enumerate "
        "selected available accessible uploaded attached in this chat conversation to me my you "
        "your can access read see please only how many do have get tell currently "
        "quels quelles les des sont disponibles sélectionnés sélectionnées afficher liste "
        "qué cuáles los las están disponibles seleccionados mostrar lista "
        "welche sind verfügbar ausgewählt auflisten alle".split()
    )
    return bool(words & subjects and words <= catalog_words) or text in {
        "what can you access",
        "what can you read",
        "which sources can you access",
        "มีเอกสารอะไรบ้าง",
        "มีไฟล์อะไรบ้าง",
        "แสดงรายการเอกสารทั้งหมด",
        "เอกสารที่เลือกมีอะไรบ้าง",
        "有哪些文档",
        "列出所有文件",
        "哪些文件可用",
        "有哪些文件",
        "どの文書が利用できますか",
        "利用できるファイル一覧",
        "quelles sources sont disponibles",
    }
