from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from support_copilot.database import SupportDatabase
from support_copilot.providers import ExtractiveProvider, LangChainReActProvider, transcribe_and_summarize_call
from support_copilot.retrieval import build_retriever, sections_from_file
from support_copilot.workflow import SupportWorkflow

ROOT = Path(__file__).parent
st.set_page_config(page_title="Orbit Support", page_icon="◎", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 20% 0,#312e81,#111827 38%,#030712);color:#f5f3ff}
.orbit{padding:1.6rem 2rem;border-radius:25px;background:linear-gradient(110deg,#7c3aed44,#ec489933);border:1px solid #c4b5fd55;animation:orbit 4s infinite alternate}
@keyframes orbit{to{box-shadow:0 0 46px #a855f744}}
@media (prefers-reduced-motion: reduce){.orbit{animation:none!important}}
[data-testid=stChatMessage]{background:#ffffff0a;border:1px solid #ffffff14;border-radius:18px}
</style><div class="orbit"><h1>◎ Orbit Support</h1><p>Grounded answers, visible multi-agent routing, voice-call briefs, and accountable escalation.</p></div>
""", unsafe_allow_html=True)

configured_db = Path(os.getenv("SUPPORT_DB_PATH", "data/support.sqlite3"))
db = SupportDatabase(configured_db if configured_db.is_absolute() else ROOT / configured_db)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_sources" not in st.session_state:
    st.session_state.knowledge_sources = {
        "help-center": sections_from_file(ROOT / "knowledge/help_center.md", "help_center.md")
    }

voice_question = None
with st.sidebar:
    st.header("Session")
    customer_id = st.text_input("Customer ID (optional)", placeholder="C-1001")
    provider_options = {"Offline extractive": "offline"}
    if os.getenv("OPENAI_API_KEY"):
        provider_options["LangChain ReAct + OpenAI"] = "react"
    configured_provider = os.getenv("SUPPORT_PROVIDER", "offline").lower()
    default_provider = next((label for label, value in provider_options.items() if value == configured_provider), next(iter(provider_options)))
    provider_label = st.selectbox("Answer engine", list(provider_options), index=list(provider_options).index(default_provider))
    workflow_engine = st.selectbox("Workflow engine", ["auto", "langgraph", "local"], index=["auto", "langgraph", "local"].index(os.getenv("SUPPORT_WORKFLOW_ENGINE", "auto")) if os.getenv("SUPPORT_WORKFLOW_ENGINE", "auto") in {"auto", "langgraph", "local"} else 0)
    retrieval_backend = st.selectbox("Retrieval backend", ["lexical", "chroma"], index=1 if os.getenv("SUPPORT_RETRIEVAL_BACKEND", "lexical") == "chroma" else 0)

    knowledge_uploads = st.file_uploader("Add knowledge sources", type=["md", "txt", "pdf"], accept_multiple_files=True)
    if knowledge_uploads and st.button("Index knowledge", use_container_width=True):
        added = 0
        max_bytes = int(os.getenv("SUPPORT_MAX_KNOWLEDGE_MB", "10")) * 1024 * 1024
        for upload in knowledge_uploads:
            if upload.size > max_bytes:
                st.error(f"{upload.name} exceeds the configured knowledge-file limit")
                continue
            with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix, delete=False) as handle:
                handle.write(upload.getvalue())
                knowledge_path = Path(handle.name)
            try:
                sections = sections_from_file(knowledge_path, upload.name)
                st.session_state.knowledge_sources[upload.name] = sections
                added += len(sections)
            finally:
                knowledge_path.unlink(missing_ok=True)
        st.success(f"Indexed {added} section(s) across {len(knowledge_uploads)} source(s)")
    st.caption(f"{len(st.session_state.knowledge_sources)} sources · {sum(map(len, st.session_state.knowledge_sources.values()))} sections")

    audio = st.file_uploader("Voice-call recording", type=["wav", "mp3", "m4a", "webm"])
    if audio and st.button("Transcribe + summarize call", use_container_width=True):
        with tempfile.NamedTemporaryFile(suffix=Path(audio.name).suffix, delete=False) as handle:
            handle.write(audio.getvalue())
            temp_path = handle.name
        try:
            report = transcribe_and_summarize_call(
                temp_path,
                os.getenv("OPENAI_API_KEY", ""),
                os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1"),
                os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            )
            st.session_state.call_report = report
            voice_question = report.summary
        except Exception as exc:
            st.error(str(exc))
        finally:
            Path(temp_path).unlink(missing_ok=True)
    if report := st.session_state.get("call_report"):
        st.success(report.summary)
        with st.expander("Call transcript"):
            st.write(report.transcript)
            st.caption(f"{report.transcription_provider} → {report.summary_provider}")

    st.divider()
    tickets = db.tickets()
    st.metric("Open tickets", sum(row["status"] == "open" for row in tickets))
    with st.expander("Recent tickets"):
        for ticket in tickets[:8]:
            st.caption(f"#{ticket['id']} · {ticket['priority']} · {ticket['subject']}")

try:
    retriever = build_retriever(
        st.session_state.knowledge_sources,
        backend=retrieval_backend,
        persist_directory=ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma"),
        embedding_model=os.getenv("CHROMA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
except Exception as exc:
    st.error(f"Could not initialize {retrieval_backend} retrieval: {exc}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("trace"):
            st.caption(" → ".join(message["trace"]))

question = st.chat_input("Ask about billing, plans, security, an incident, or request a person…") or voice_question
if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    try:
        provider = (
            LangChainReActProvider(os.getenv("OPENAI_API_KEY", ""), os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
            if provider_options[provider_label] == "react"
            else ExtractiveProvider()
        )
        workflow = SupportWorkflow(
            retriever,
            db,
            provider,
            top_k=int(os.getenv("SUPPORT_TOP_K", "4")),
            escalation_threshold=float(os.getenv("SUPPORT_ESCALATION_THRESHOLD", "0.05")),
            workflow_engine=workflow_engine,
        )
        result = workflow.run(question, customer_id or None)
        with st.chat_message("assistant"):
            st.markdown(result.message)
            st.caption(" → ".join(result.trace))
            with st.expander("Evidence across sources"):
                for source in result.sources:
                    st.markdown(f"**{source.title}** · `{source.source}` · `{source.score:.3f}`\n\n{source.text}")
        st.session_state.messages.append({"role": "assistant", "text": result.message, "trace": result.trace})
    except Exception as exc:
        st.error(str(exc))
