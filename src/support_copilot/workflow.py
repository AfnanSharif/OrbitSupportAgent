from __future__ import annotations

import re
from typing import TypedDict

from .database import SupportDatabase
from .models import Intent, KnowledgeHit, SupportResponse
from .providers import AnswerProvider, ExtractiveProvider
from .retrieval import Retriever


class WorkflowState(TypedDict, total=False):
    question: str
    customer_id: str | None
    intent: Intent
    route: str
    customer: dict[str, str] | None
    hits: list[KnowledgeHit]
    context: str
    message: str
    ticket_id: int | None
    trace: list[str]


def classify_intent(question: str) -> Intent:
    normalized = question.lower()
    if any(term in normalized for term in ("human", "person", "agent", "representative")):
        return Intent.HUMAN
    if any(term in normalized for term in ("hacked", "breach", "password", "recovery code", "compromised")):
        return Intent.SECURITY
    if any(term in normalized for term in ("refund", "invoice", "charge", "billing", "price")):
        return Intent.BILLING
    if any(term in normalized for term in ("my plan", "my account", "subscription", "customer id")):
        return Intent.ACCOUNT
    if any(term in normalized for term in ("current incident", "status", "outage", "research", "across sources")):
        return Intent.RESEARCH
    return Intent.KNOWLEDGE


class SupportWorkflow:
    """Shared nodes executed by either LangGraph or the deterministic local runner."""

    def __init__(
        self,
        knowledge: Retriever,
        database: SupportDatabase,
        provider: AnswerProvider | None = None,
        *,
        top_k: int = 4,
        escalation_threshold: float = 0.05,
        workflow_engine: str = "auto",
        graph_factory=None,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 <= escalation_threshold <= 1:
            raise ValueError("escalation_threshold must be between 0 and 1")
        if workflow_engine not in {"auto", "langgraph", "local"}:
            raise ValueError("workflow_engine must be auto, langgraph, or local")
        self.knowledge, self.database = knowledge, database
        self.provider = provider or ExtractiveProvider()
        self.top_k = top_k
        self.escalation_threshold = escalation_threshold
        self.workflow_engine = workflow_engine
        self.graph_factory = graph_factory

    @staticmethod
    def _with_trace(state: WorkflowState, event: str) -> list[str]:
        return [*state.get("trace", []), event]

    def router_agent(self, state: WorkflowState) -> WorkflowState:
        intent = classify_intent(state["question"])
        route = "escalation_agent" if intent == Intent.HUMAN else "research_agent" if intent == Intent.RESEARCH else "knowledge_agent"
        return {"intent": intent, "route": route, "trace": self._with_trace(state, f"agent:router:{intent.value}")}

    def database_agent(self, state: WorkflowState) -> WorkflowState:
        customer_id = state.get("customer_id")
        customer = self.database.customer(customer_id)
        return {"customer": customer, "trace": self._with_trace(state, "agent:database")}

    def knowledge_agent(self, state: WorkflowState) -> WorkflowState:
        hits = self.knowledge.search(state["question"], self.top_k)
        context = "\n---\n".join(f"[{hit.source}] {hit.title}: {hit.text}" for hit in hits)
        return {"hits": hits, "context": context, "trace": self._with_trace(state, "agent:knowledge")}

    def research_agent(self, state: WorkflowState) -> WorkflowState:
        hits = self.knowledge.search(state["question"], self.top_k * 2)
        selected: list[KnowledgeHit] = []
        represented: set[str] = set()
        for hit in hits:
            if hit.source not in represented:
                selected.append(hit)
                represented.add(hit.source)
            if len(selected) == self.top_k:
                break
        for hit in hits:
            if hit not in selected:
                selected.append(hit)
            if len(selected) == self.top_k:
                break
        context = "\n---\n".join(f"[{hit.source}] {hit.title}: {hit.text}" for hit in selected)
        return {"hits": selected, "context": context, "trace": self._with_trace(state, "agent:research")}

    def answer_agent(self, state: WorkflowState) -> WorkflowState:
        message = self.provider.answer(state["question"], state.get("context", ""), state.get("customer"))
        return {"message": message, "trace": self._with_trace(state, f"agent:answer:{self.provider.name}")}

    def escalation_agent(self, state: WorkflowState) -> WorkflowState:
        intent = state["intent"]
        priority = "urgent" if intent == Intent.SECURITY else "normal"
        ticket_id = self.database.create_ticket(
            f"{intent.value.title()} assistance",
            state["question"],
            customer_id=state.get("customer_id"),
            priority=priority,
        )
        message = state.get("message") or "I’m routing this request to a human specialist."
        message += f"\n\nI created ticket **#{ticket_id}** for a human specialist."
        return {"message": message, "ticket_id": ticket_id, "trace": self._with_trace(state, "agent:escalation")}

    def _route_after_database(self, state: WorkflowState) -> str:
        return state["route"]

    def _route_after_answer(self, state: WorkflowState) -> str:
        intent = state["intent"]
        hits = state.get("hits", [])
        needs_ticket = intent == Intent.SECURITY or not hits or hits[0].score < self.escalation_threshold
        if intent == Intent.BILLING and "refund" in state["question"].lower():
            needs_ticket = True
        return "escalation_agent" if needs_ticket else "end"

    def build_langgraph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("Install langgraph to use the graph workflow engine") from exc
        builder = StateGraph(WorkflowState)
        builder.add_node("router_agent", self.router_agent)
        builder.add_node("database_agent", self.database_agent)
        builder.add_node("knowledge_agent", self.knowledge_agent)
        builder.add_node("research_agent", self.research_agent)
        builder.add_node("answer_agent", self.answer_agent)
        builder.add_node("escalation_agent", self.escalation_agent)
        builder.add_edge(START, "router_agent")
        builder.add_edge("router_agent", "database_agent")
        builder.add_conditional_edges(
            "database_agent",
            self._route_after_database,
            {
                "knowledge_agent": "knowledge_agent",
                "research_agent": "research_agent",
                "escalation_agent": "escalation_agent",
            },
        )
        builder.add_edge("knowledge_agent", "answer_agent")
        builder.add_edge("research_agent", "answer_agent")
        builder.add_conditional_edges("answer_agent", self._route_after_answer, {"escalation_agent": "escalation_agent", "end": END})
        builder.add_edge("escalation_agent", END)
        return builder.compile()

    @staticmethod
    def _merge(state: WorkflowState, update: WorkflowState) -> WorkflowState:
        return {**state, **update}

    def _run_local(self, state: WorkflowState) -> WorkflowState:
        state = self._merge(state, self.router_agent(state))
        state = self._merge(state, self.database_agent(state))
        if state["route"] == "escalation_agent":
            return self._merge(state, self.escalation_agent(state))
        worker = self.research_agent if state["route"] == "research_agent" else self.knowledge_agent
        state = self._merge(state, worker(state))
        state = self._merge(state, self.answer_agent(state))
        if self._route_after_answer(state) == "escalation_agent":
            state = self._merge(state, self.escalation_agent(state))
        return state

    def _graph(self):
        if self.graph_factory is not None:
            return self.graph_factory(self)
        if self.workflow_engine == "local":
            return None
        try:
            return self.build_langgraph()
        except RuntimeError:
            if self.workflow_engine == "langgraph":
                raise
            return None

    def run(self, question: str, customer_id: str | None = None) -> SupportResponse:
        if not isinstance(question, str):
            raise ValueError("question must be a string")
        clean = re.sub(r"\s+", " ", question).strip()
        if not clean:
            raise ValueError("question cannot be empty")
        if len(clean) > 4_000:
            raise ValueError("question cannot exceed 4,000 characters")
        graph = self._graph()
        initial: WorkflowState = {
            "question": clean,
            "customer_id": customer_id,
            "trace": ["engine:langgraph" if graph is not None else "engine:local"],
            "ticket_id": None,
        }
        final = graph.invoke(initial) if graph is not None else self._run_local(initial)
        response = SupportResponse(final["message"], final["intent"], final.get("hits", []), final.get("ticket_id"), final.get("trace", []))
        self.database.record_interaction(clean, response.message, response.intent.value, customer_id)
        return response
