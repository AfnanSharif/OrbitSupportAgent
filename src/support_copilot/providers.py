from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .models import CallReport


class AnswerProvider(Protocol):
    name: str
    def answer(self, question: str, context: str, customer: dict[str, str] | None) -> str: ...


class ExtractiveProvider:
    name = "offline-extractive"

    def answer(self, question: str, context: str, customer: dict[str, str] | None) -> str:
        if not context:
            return "I do not have enough verified information to answer that. I can open a support ticket for a specialist."
        prefix = f"For {customer['name']} on the {customer['plan']} plan: " if customer else ""
        return prefix + context.split("\n---\n", 1)[0]


@dataclass(frozen=True, slots=True)
class ReActDecision:
    action: str | None = None
    action_input: str = ""
    final_answer: str | None = None


class SupportReActOutputParser:
    """Parse the deliberately small ReAct contract used by the support tools."""

    allowed_actions = {"knowledge_search", "customer_lookup"}

    def parse(self, value) -> ReActDecision:
        text = str(getattr(value, "content", value)).strip()
        final = re.search(r"Final Answer\s*:\s*(.+)", text, re.I | re.S)
        if final:
            answer = final.group(1).strip()
            if not answer:
                raise ValueError("Final Answer cannot be empty")
            return ReActDecision(final_answer=answer)
        action = re.search(r"Action\s*:\s*([a-z_]+)", text, re.I)
        action_input = re.search(r"Action Input\s*:\s*(.+)", text, re.I | re.S)
        if not action or not action_input:
            raise ValueError("Expected either Final Answer or Action plus Action Input")
        name = action.group(1).lower()
        if name not in self.allowed_actions:
            raise ValueError(f"Unsupported ReAct tool: {name}")
        return ReActDecision(action=name, action_input=action_input.group(1).strip())


class LangChainReActProvider:
    name = "langchain-react-openai"

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        *,
        chain=None,
        parser: SupportReActOutputParser | None = None,
        max_steps: int = 3,
    ) -> None:
        if max_steps < 1 or max_steps > 8:
            raise ValueError("max_steps must be between 1 and 8")
        self.parser = parser or SupportReActOutputParser()
        self.max_steps = max_steps
        self.chain = chain or self._build_chain(api_key, model)

    def _build_chain(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableLambda
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install langchain, langchain-openai, and their dependencies for the ReAct provider") from exc
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a careful SaaS support ReAct agent. Use only tool observations. "
                "Never request secrets or promise refunds. Choose exactly one format: "
                "'Action: knowledge_search\\nAction Input: <query>', "
                "'Action: customer_lookup\\nAction Input: <customer question>', or "
                "'Final Answer: <grounded concise answer>'.",
            ),
            ("human", "QUESTION:\n{question}\n\nPRELOADED CONTEXT:\n{context}\n\nSCRATCHPAD:\n{scratchpad}"),
        ])
        return prompt | ChatOpenAI(api_key=api_key, model=model, temperature=0.1) | RunnableLambda(self.parser.parse)

    def answer(self, question: str, context: str, customer: dict[str, str] | None) -> str:
        safe_customer = {key: value for key, value in (customer or {}).items() if key not in {"email"}}
        tools = {
            "knowledge_search": lambda _query: context or "No verified knowledge evidence was retrieved.",
            "customer_lookup": lambda _query: json.dumps(safe_customer or {"status": "anonymous"}, sort_keys=True),
        }
        scratchpad = ""
        for _ in range(self.max_steps):
            decision = self.chain.invoke({"question": question, "context": context, "scratchpad": scratchpad})
            if not isinstance(decision, ReActDecision):
                decision = self.parser.parse(decision)
            if decision.final_answer:
                return decision.final_answer
            observation = tools[decision.action](decision.action_input)
            scratchpad += f"\nAction: {decision.action}\nAction Input: {decision.action_input}\nObservation: {observation}\n"
        raise RuntimeError("ReAct provider reached its tool-step limit without a final answer")


# Backward-compatible import name for existing callers.
LangChainOpenAIProvider = LangChainReActProvider


def _openai_client(api_key: str, client=None):
    if client is not None:
        return client
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai to use voice-call processing") from exc
    return OpenAI(api_key=api_key)


def transcribe_audio(path: str, api_key: str, model: str = "whisper-1", *, client=None) -> str:
    client = _openai_client(api_key, client)
    with open(path, "rb") as audio:
        response = client.audio.transcriptions.create(model=model, file=audio)
    text = response.get("text", "") if isinstance(response, dict) else getattr(response, "text", "")
    if not str(text).strip():
        raise RuntimeError("Voice transcription returned no text")
    return str(text).strip()


def transcribe_and_summarize_call(
    path: str,
    api_key: str,
    transcription_model: str = "whisper-1",
    summary_model: str = "gpt-4o-mini",
    *,
    client=None,
) -> CallReport:
    client = _openai_client(api_key, client)
    transcript = transcribe_audio(path, api_key, transcription_model, client=client)
    response = client.responses.create(
        model=summary_model,
        instructions=(
            "Summarize this customer-support call using only the transcript. Include the issue, "
            "confirmed facts, actions already tried, requested next step, and unresolved risks. "
            "Do not invent customer details or commitments."
        ),
        input=transcript,
    )
    summary = response.get("output_text", "") if isinstance(response, dict) else getattr(response, "output_text", "")
    if not str(summary).strip():
        raise RuntimeError("Call summarization returned no text")
    return CallReport(transcript, str(summary).strip(), f"openai:{transcription_model}", f"openai:{summary_model}")
