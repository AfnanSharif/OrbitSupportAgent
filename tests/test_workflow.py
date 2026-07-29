import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from support_copilot.database import SupportDatabase
from support_copilot.models import Intent, KnowledgeHit
from support_copilot.providers import LangChainReActProvider, ReActDecision, SupportReActOutputParser, transcribe_and_summarize_call
from support_copilot.retrieval import ChromaKnowledgeBase, KnowledgeBase, MultiSourceRetriever
from support_copilot.workflow import SupportWorkflow, classify_intent


class WorkflowTests(unittest.TestCase):
    def test_intent_routing(self) -> None:
        self.assertEqual(classify_intent("My password was compromised"), Intent.SECURITY)
        self.assertEqual(classify_intent("I need a refund"), Intent.BILLING)
        self.assertEqual(classify_intent("Talk to a person"), Intent.HUMAN)

    def test_grounded_answer_without_ticket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = SupportDatabase(Path(directory) / "test.sqlite3")
            knowledge = KnowledgeBase([("Plans", "The Pro plan includes audit logs and priority support.")])
            result = SupportWorkflow(knowledge, database).run("Does Pro include audit logs?", "C-1001")
            self.assertEqual(result.intent, Intent.KNOWLEDGE)
            self.assertTrue(result.sources)
            self.assertIsNone(result.ticket_id)

    def test_security_query_opens_urgent_ticket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = SupportDatabase(Path(directory) / "test.sqlite3")
            knowledge = KnowledgeBase([("Safety", "Reset a compromised password immediately.")])
            result = SupportWorkflow(knowledge, database).run("My password is compromised", "C-1001")
            self.assertIsNotNone(result.ticket_id)
            self.assertEqual(database.tickets()[0]["priority"], "urgent")

    def test_workflow_validates_retrieval_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = SupportDatabase(Path(directory) / "test.sqlite3")
            knowledge = KnowledgeBase([("Plans", "The Pro plan includes audit logs.")])
            with self.assertRaises(ValueError):
                SupportWorkflow(knowledge, database, top_k=0)
            with self.assertRaises(ValueError):
                SupportWorkflow(knowledge, database, escalation_threshold=2)

    def test_named_agents_route_knowledge_research_database_and_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = SupportDatabase(Path(directory) / "test.sqlite3")
            retriever = MultiSourceRetriever({
                "help": KnowledgeBase([("Incidents", "Current incident status is posted on the status page.")], "help"),
                "runbook": KnowledgeBase([("Outages", "Regional outages require a request identifier.")], "runbook"),
            })
            workflow = SupportWorkflow(retriever, database, workflow_engine="local")
            normal = workflow.run("Where is current incident status?", "C-1001")
            human = workflow.run("I need a human representative", "C-1001")
        self.assertIn("agent:router:research", normal.trace)
        self.assertIn("agent:database", normal.trace)
        self.assertIn("agent:research", normal.trace)
        self.assertGreaterEqual(len({hit.source for hit in normal.sources}), 1)
        self.assertIn("agent:escalation", human.trace)

    def test_injected_graph_factory_is_invoked(self) -> None:
        calls = []

        class FakeGraph:
            def __init__(self, workflow):
                self.workflow = workflow

            def invoke(self, state):
                calls.append("invoke")
                return self.workflow._run_local(state)

        with tempfile.TemporaryDirectory() as directory:
            workflow = SupportWorkflow(
                KnowledgeBase([("Plans", "Pro includes audit logs.")]),
                SupportDatabase(Path(directory) / "test.sqlite3"),
                workflow_engine="langgraph",
                graph_factory=lambda current: FakeGraph(current),
            )
            workflow.run("Does Pro include audit logs?")
        self.assertEqual(calls, ["invoke"])

    def test_custom_react_parser_and_tool_loop(self) -> None:
        parser = SupportReActOutputParser()
        self.assertEqual(parser.parse("Action: knowledge_search\nAction Input: audit logs").action, "knowledge_search")
        self.assertEqual(parser.parse("Final Answer: Pro includes audit logs.").final_answer, "Pro includes audit logs.")

        class FakeChain:
            def __init__(self):
                self.requests = []

            def invoke(self, payload):
                self.requests.append(payload)
                if len(self.requests) == 1:
                    return ReActDecision(action="knowledge_search", action_input="audit logs")
                return ReActDecision(final_answer="Pro includes audit logs according to the help center.")

        chain = FakeChain()
        provider = LangChainReActProvider(chain=chain)
        answer = provider.answer("Does Pro include audit logs?", "Pro includes audit logs.", None)
        self.assertIn("Pro includes", answer)
        self.assertIn("Observation: Pro includes audit logs.", chain.requests[1]["scratchpad"])

    def test_chroma_backend_upserts_and_queries_semantic_results(self) -> None:
        class Collection:
            def upsert(self, **kwargs):
                self.upserted = kwargs

            def query(self, **kwargs):
                self.queried = kwargs
                return {
                    "documents": [["Pro includes audit logs."]],
                    "metadatas": [[{"title": "Plans", "source": "help"}]],
                    "distances": [[0.12]],
                }

        class Client:
            def get_or_create_collection(self, **kwargs):
                self.kwargs = kwargs
                self.collection = Collection()
                return self.collection

        client = Client()
        retriever = ChromaKnowledgeBase([("Plans", "Pro includes audit logs.")], source="help", client=client, embedding_function=lambda rows: rows)
        hits = retriever.search("audit logs", 2)
        self.assertEqual(client.collection.upserted["metadatas"][0]["source"], "help")
        self.assertEqual(hits[0].source, "help")
        self.assertAlmostEqual(hits[0].score, 0.88)

    def test_voice_call_is_transcribed_then_summarized(self) -> None:
        class Transcriptions:
            def create(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(text="Customer reports duplicate billing and requests review.")

        class Responses:
            def create(self, **kwargs):
                self.request = kwargs
                return SimpleNamespace(output_text="Duplicate charge requires billing review; no refund was promised.")

        client = SimpleNamespace(audio=SimpleNamespace(transcriptions=Transcriptions()), responses=Responses())
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "call.wav"
            audio.write_bytes(b"fake")
            report = transcribe_and_summarize_call(str(audio), "", client=client)
        self.assertIn("duplicate billing", report.transcript)
        self.assertIn("billing review", report.summary)
        self.assertEqual(client.responses.request["input"], report.transcript)

    def test_multi_source_retrieval_keeps_source_provenance(self) -> None:
        retriever = MultiSourceRetriever({
            "manual": KnowledgeBase([("Plans", "Pro audit logs")], "manual"),
            "faq": KnowledgeBase([("Audit", "Audit logs are retained")], "faq"),
        })
        hits = retriever.search("audit logs", 4)
        self.assertEqual({hit.source for hit in hits}, {"manual", "faq"})


if __name__ == "__main__":
    unittest.main()
