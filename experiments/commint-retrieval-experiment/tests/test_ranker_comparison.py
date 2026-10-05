import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_ranker_comparison import build_comparison, evaluate_order  # noqa: E402


class RankerComparisonTests(unittest.TestCase):
    def test_thread_ranker_comparison_uses_identical_complete_candidate_pool(self):
        result = build_comparison()["thread_comparison"]

        self.assertEqual(result["baseline"]["candidate_count"], 15)
        self.assertEqual(len(result["comment_engagement_reranker"]["ranked_ids"]), 15)
        self.assertEqual(set(result["baseline"]["ranked_ids"]), set(result["comment_engagement_reranker"]["ranked_ids"]))
        self.assertEqual(result["baseline"]["judged_count"], 15)
        self.assertEqual(result["comment_engagement_reranker"]["judged_count"], 15)
        self.assertAlmostEqual(result["baseline"]["nDCG@10"], 0.8356420858776415)
        self.assertAlmostEqual(result["comment_engagement_reranker"]["nDCG@10"], 0.821686241587554)
        self.assertEqual(result["baseline"]["MRR"], 1.0)
        self.assertEqual(result["comment_engagement_reranker"]["MRR"], 1.0)
        self.assertEqual(result["candidate_coverage"]["reranker_retained_candidate_fraction"], 1.0)

    def test_community_comparison_is_explicitly_limited_to_three_shared_items(self):
        result = build_comparison()["community_comparison"]

        self.assertEqual(result["baseline"]["ranked_ids"], ["studypartner", "studying", "getstudying"])
        self.assertEqual(result["comment_aggregation_ranker"]["ranked_ids"], ["getstudying", "studying", "studypartner"])
        self.assertEqual(result["baseline"]["judged_count"], 3)
        self.assertEqual(result["comment_aggregation_ranker"]["judged_count"], 3)
        self.assertAlmostEqual(result["baseline"]["nDCG@10"], result["comment_aggregation_ranker"]["nDCG@10"])

    def test_metrics_are_suppressed_for_incomplete_or_duplicate_rankings(self):
        incomplete = evaluate_order(["judged", "missing"], "thread", {("thread", "judged"): 3})
        self.assertFalse(incomplete["judgments_complete"])
        self.assertIsNone(incomplete["nDCG@10"])
        self.assertIsNone(incomplete["MRR"])

        with self.assertRaises(ValueError):
            evaluate_order(["same", "same"], "thread", {("thread", "same"): 2})


if __name__ == "__main__":
    unittest.main()
