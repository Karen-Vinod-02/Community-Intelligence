import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import (  # noqa: E402
    evaluate_agent_reach_safety_scope,
    evaluate_communities,
    load_annotations,
)


class EvaluatorScopeTests(unittest.TestCase):
    def test_only_pipeline_ranked_mcp_list_receives_community_metrics(self):
        metrics = evaluate_communities(load_annotations())

        self.assertEqual(set(metrics), {"mcp_ranked_baseline"})
        result = metrics["mcp_ranked_baseline"]["study_accountability"]
        self.assertEqual(result["n_items"], 14)
        self.assertTrue(result["judgments_complete"])
        self.assertAlmostEqual(result["ndcg@10"], 0.8904958081358094)

    def test_agent_reach_scope_keeps_safety_separate_from_relevance(self):
        scope = evaluate_agent_reach_safety_scope(load_annotations())["study_accountability"]

        self.assertEqual(scope["raw_post_rows"], 75)
        self.assertEqual(scope["distinct_communities"], 32)
        self.assertEqual(scope["safety_eligible_communities"], 13)
        self.assertEqual(scope["safety_ineligible_communities"], 19)
        self.assertEqual(scope["eligible_communities_with_relevance_judgment"], 13)
        self.assertIsNone(scope["ranking_metrics"])


if __name__ == "__main__":
    unittest.main()
