import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_evaluation import build_audit, ranked_metrics  # noqa: E402


class EvaluationAuditTests(unittest.TestCase):
    def test_active_thread_list_is_fully_judged_and_scoped(self):
        audit = build_audit()
        result = audit["thread_evaluation"]

        self.assertTrue(result["ranking_order_verified"])
        self.assertFalse(result["duplicate_post_ids"])
        self.assertEqual(result["metrics"]["annotated_items"], 15)
        self.assertEqual(result["metrics"]["unique_ranked_items"], 15)
        self.assertAlmostEqual(result["metrics"]["ndcg@10"], 0.8356420858776415)
        self.assertEqual(result["metrics"]["mrr"], 1.0)
        self.assertIsNone(result["retrieved_candidate_count_available"])

    def test_agent_reach_safety_is_not_relevance(self):
        audit = build_audit()
        result = audit["community_evaluation"]["agent_reach_safety_scoped_findings"]

        self.assertEqual(result["safety_eligible_communities"], 13)
        self.assertEqual(result["eligible_communities_with_relevance_judgment"], 13)
        self.assertIn("getstudying", result["ineligible_ids"])
        self.assertIsNone(result["rank_metrics"])
        self.assertEqual(audit["community_evaluation"]["excluded_annotation_archive_rows"], 18)

    def test_unjudged_and_duplicate_items_do_not_create_metric_labels(self):
        items = [
            {"post_id": "abc"},
            {"post_id": "abc"},
            {"post_id": "missing"},
        ]
        result = ranked_metrics(items, "thread", {("thread", "abc"): 3})

        self.assertEqual(result["unique_ranked_items"], 2)
        self.assertEqual(result["duplicate_rows_removed_for_metric"], 1)
        self.assertEqual(result["annotated_items"], 1)
        self.assertFalse(result["judgments_complete"])
        self.assertIsNone(result["ndcg@10"])
        self.assertIsNone(result["mrr"])


if __name__ == "__main__":
    unittest.main()
