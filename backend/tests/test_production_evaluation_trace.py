import json
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock, patch


# Keep these focused unit tests runnable in lightweight environments where
# the API/runtime integrations are not installed. Installed packages are
# used normally; stubs cover import-time names only and are never executed.
if importlib.util.find_spec("fastapi") is None:
    fastapi_stub = ModuleType("fastapi")

    class _Router:
        def post(self, *args, **kwargs):
            return lambda function: function

    class _HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    fastapi_stub.APIRouter = _Router
    fastapi_stub.HTTPException = _HTTPException
    sys.modules["fastapi"] = fastapi_stub

if importlib.util.find_spec("ddgs") is None:
    ddgs_stub = ModuleType("ddgs")
    ddgs_stub.DDGS = object
    sys.modules["ddgs"] = ddgs_stub

if importlib.util.find_spec("sentence_transformers") is None:
    sentence_transformers_stub = ModuleType("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            raise AssertionError("The optional model stub should not be used by these tests")

    sentence_transformers_stub.SentenceTransformer = _SentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers_stub

from app.services import community as community_module
with patch.object(community_module, "CommunityService", return_value=SimpleNamespace()):
    from app.api import analysis
from app.models.analysis import AnalyzeRequest
from app.models.reddit import RedditPost
from app.services.community import CommunityService
from app.services.evaluation_trace import TRACE_DIRECTORY_ENV
from app.services.ranking import CommunityCandidate, CommunityRanker


class ProductionEvaluationTraceTests(unittest.TestCase):
    def test_production_ranker_trace_records_all_post_scores_and_order(self):
        import numpy as np

        class EqualSimilarityEmbeddings:
            def encode(self, texts):
                return np.ones((len(texts), 2), dtype=float) / (2 ** 0.5)

        ranker = CommunityRanker(embeddings=EqualSimilarityEmbeddings())
        candidate = CommunityCandidate(
            subreddit="study",
            posts=[RedditPost(
                id="p1",
                subreddit="study",
                title="Looking for study partners",
                body="I need accountability and a reliable study partner.",
                score=5,
                num_comments=2,
            )],
        )
        trace = {}
        result = ranker.rank(
            "An app helping students find reliable study partners",
            [candidate],
            trace=trace,
        )

        self.assertEqual(trace["candidate_diagnostics"][0]["input_post_count"], 1)
        self.assertEqual(trace["candidate_diagnostics"][0]["posts"][0]["post_id"], "p1")
        self.assertGreater(trace["candidate_diagnostics"][0]["posts"][0]["score"], 0)
        self.assertEqual(trace["ranked_output"][0]["subreddit"], result[0]["subreddit"])

    def test_service_trace_preserves_pre_limit_and_actual_ranker_inputs(self):
        service = CommunityService.__new__(CommunityService)
        service.discovery = SimpleNamespace(
            discover_communities=lambda query, limit: ["study"]
        )
        posts = [
            RedditPost(id=f"p{i}", subreddit="study", title=f"Post {i}", body="body")
            for i in range(3)
        ]
        service.reddit = SimpleNamespace(search_posts=lambda subreddit, limit: posts)

        class FakeFilter:
            def rerank(self, description, candidates, queries=None, trace=None):
                trace.update({"config": {"test": True}, "output_order": [c.subreddit for c in candidates]})
                return candidates

        class FakeRanker:
            embeddings = SimpleNamespace(model=SimpleNamespace(name_or_path="test-embedding"))

            def rank(self, description, candidates, trace=None):
                trace.update({"candidate_diagnostics": [{"subreddit": "study", "posts": ["p0"]}]})
                return [{
                    "subreddit": "study",
                    "score": 0.7,
                    "status": "potential",
                    "signals": {"problem_relevant_posts": 1},
                    "top_posts": candidates[0].posts[:1],
                }]

        service.community_filter = FakeFilter()
        service.ranker = FakeRanker()
        trace = {}
        with patch("app.services.community.extract_search_queries", return_value=["study partners"]):
            result = service.analyze(
                "exact user description",
                posts_per_community=1,
                trace=trace,
            )

        self.assertEqual([row["id"] for row in trace["retrieval"][0]["retrieved_posts"]], ["p0", "p1", "p2"])
        self.assertEqual([row["id"] for row in trace["retrieval"][0]["usable_before_candidate_limit"]], ["p0", "p1", "p2"])
        self.assertEqual(trace["retrieval"][0]["ranker_input_ids"], ["p0"])
        self.assertEqual(len(trace["ranker_input"][0]["posts"]), 1)
        self.assertEqual(trace["community_ranker"]["candidate_diagnostics"][0]["subreddit"], "study")
        self.assertEqual(trace["returned_output"][0]["status"], "potential")
        self.assertEqual(result[0]["status"], "potential")

    def test_api_forwards_status_signals_and_fallback_contract(self):
        signal_data = {
            "problem_relevant_posts": 2,
            "evidence_mean": 0.5,
            "prevalence": 0.4,
            "avg_recency": 0.8,
        }
        result = {
            "subreddit": "study",
            "score": 0.4,
            "status": "potential",
            "signals": signal_data,
            "top_posts": [RedditPost(id="p1", subreddit="study", title="Study", body="")],
        }
        fake_service = SimpleNamespace(analyze=lambda *args, **kwargs: [result])

        with patch.object(analysis, "community_service", fake_service):
            response = analysis.analyze(AnalyzeRequest(description="problem"))

        self.assertEqual(response.communities[0].status, "potential")
        self.assertEqual(response.communities[0].signals.problem_relevant_posts, 2)
        self.assertTrue(response.fallback_used)

    def test_opt_in_trace_is_saved_and_trace_id_is_returned(self):
        signal_data = {
            "problem_relevant_posts": 2,
            "evidence_mean": 0.5,
            "prevalence": 0.4,
            "avg_recency": 0.8,
        }

        def analyze_service(description, **kwargs):
            kwargs["trace"].update({"description": description, "ranked_output": []})
            return [{
                "subreddit": "study",
                "score": 0.4,
                "status": "relevant",
                "signals": signal_data,
                "top_posts": [RedditPost(id="p1", subreddit="study", title="Study", body="")],
            }]

        fake_service = SimpleNamespace(analyze=analyze_service)
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {TRACE_DIRECTORY_ENV: temp_dir}):
                with patch.object(analysis, "community_service", fake_service):
                    response = analysis.analyze(
                        AnalyzeRequest(description="exact user problem", capture_trace=True)
                    )

            self.assertIsNotNone(response.evaluation_trace_id)
            trace_path = Path(temp_dir) / f"{response.evaluation_trace_id}.json"
            saved = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["description"], "exact user problem")
            self.assertEqual(saved["final_response"]["evaluation_trace_id"], response.evaluation_trace_id)

    def test_capture_request_requires_configured_trace_directory(self):
        from fastapi import HTTPException

        fake_analyze = Mock(return_value=[])
        fake_service = SimpleNamespace(analyze=fake_analyze)
        with patch.dict(os.environ, {TRACE_DIRECTORY_ENV: ""}):
            with patch.object(analysis, "community_service", fake_service):
                with self.assertRaises(HTTPException) as raised:
                    analysis.analyze(
                        AnalyzeRequest(description="problem", capture_trace=True)
                    )

        self.assertEqual(raised.exception.status_code, 503)
        fake_analyze.assert_not_called()

    def test_opt_in_trace_is_written_when_ranker_path_raises(self):
        def fail_after_capture(description, **kwargs):
            kwargs["trace"].update({"description": description, "ranker_input": [{"subreddit": "study"}]})
            raise ValueError("ranker failed")

        fake_service = SimpleNamespace(analyze=fail_after_capture)
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {TRACE_DIRECTORY_ENV: temp_dir}):
                with patch.object(analysis, "community_service", fake_service):
                    with self.assertRaisesRegex(ValueError, "ranker failed"):
                        analysis.analyze(
                            AnalyzeRequest(description="exact user problem", capture_trace=True)
                        )

            saved_path = next(Path(temp_dir).glob("*.json"))
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["pipeline_error"]["error_type"], "ValueError")
            self.assertEqual(saved["ranker_input"], [{"subreddit": "study"}])


if __name__ == "__main__":
    unittest.main()
