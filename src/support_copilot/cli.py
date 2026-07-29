from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .database import SupportDatabase
from .providers import ExtractiveProvider, LangChainReActProvider, transcribe_and_summarize_call
from .retrieval import build_retriever, sections_from_file
from .workflow import SupportWorkflow


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run the multi-agent customer-support workflow")
    parser.add_argument("question", nargs="*")
    parser.add_argument("--audio", type=Path, help="Voice-call recording to transcribe and summarize")
    parser.add_argument("--customer")
    parser.add_argument("--provider", choices=["offline", "react", "openai"], default=os.getenv("SUPPORT_PROVIDER", "offline"))
    parser.add_argument("--knowledge", type=Path, action="append", help="Repeat for multiple Markdown/TXT/PDF sources")
    parser.add_argument("--retrieval", choices=["lexical", "chroma"], default=os.getenv("SUPPORT_RETRIEVAL_BACKEND", "lexical"))
    parser.add_argument("--workflow-engine", choices=["auto", "langgraph", "local"], default=os.getenv("SUPPORT_WORKFLOW_ENGINE", "auto"))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("SUPPORT_TOP_K", "4")))
    parser.add_argument("--escalation-threshold", type=float, default=float(os.getenv("SUPPORT_ESCALATION_THRESHOLD", "0.05")))
    parser.add_argument("--json", action="store_true", help="Print the full response, trace, and optional call report")
    args = parser.parse_args()
    if not args.question and args.audio is None:
        parser.error("provide a question or --audio recording")

    call_report = None
    if args.audio is not None:
        call_report = transcribe_and_summarize_call(
            str(args.audio),
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1"),
            os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        )
    typed_question = " ".join(args.question).strip()
    question = typed_question or (call_report.summary if call_report else "")
    if typed_question and call_report:
        question = f"{typed_question}\n\nCall summary: {call_report.summary}"

    knowledge_paths = args.knowledge or [Path("knowledge/help_center.md")]
    sources = {
        f"{path.stem}-{index}": sections_from_file(path, path.name)
        for index, path in enumerate(knowledge_paths, 1)
    }
    retriever = build_retriever(
        sources,
        backend=args.retrieval,
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "data/chroma"),
        embedding_model=os.getenv("CHROMA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    provider = (
        LangChainReActProvider(os.getenv("OPENAI_API_KEY", ""), os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
        if args.provider in {"react", "openai"}
        else ExtractiveProvider()
    )
    workflow = SupportWorkflow(
        retriever,
        SupportDatabase(os.getenv("SUPPORT_DB_PATH", "data/support.sqlite3")),
        provider,
        top_k=args.top_k,
        escalation_threshold=args.escalation_threshold,
        workflow_engine=args.workflow_engine,
    )
    result = workflow.run(question, args.customer)
    if args.json:
        payload = {
            "message": result.message,
            "intent": result.intent.value,
            "sources": [hit.__dict__ if hasattr(hit, "__dict__") else {"title": hit.title, "text": hit.text, "score": hit.score, "source": hit.source} for hit in result.sources],
            "ticket_id": result.ticket_id,
            "trace": result.trace,
            "call": None if call_report is None else {
                "transcript": call_report.transcript,
                "summary": call_report.summary,
                "transcription_provider": call_report.transcription_provider,
                "summary_provider": call_report.summary_provider,
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        if call_report:
            print("call summary:", call_report.summary)
        print(result.message)
        print("trace:", " → ".join(result.trace))


if __name__ == "__main__":
    main()
