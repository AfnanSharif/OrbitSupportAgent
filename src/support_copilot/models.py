from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Intent(StrEnum):
    KNOWLEDGE = "knowledge"
    ACCOUNT = "account"
    BILLING = "billing"
    SECURITY = "security"
    HUMAN = "human"
    RESEARCH = "research"


@dataclass(slots=True)
class KnowledgeHit:
    title: str
    text: str
    score: float
    source: str = "knowledge"


@dataclass(slots=True)
class SupportResponse:
    message: str
    intent: Intent
    sources: list[KnowledgeHit] = field(default_factory=list)
    ticket_id: int | None = None
    trace: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CallReport:
    transcript: str
    summary: str
    transcription_provider: str
    summary_provider: str
