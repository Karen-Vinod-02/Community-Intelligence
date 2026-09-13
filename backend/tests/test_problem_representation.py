import unittest

from app.services.query_extraction import extract_problem_hypotheses


class ProblemRepresentationTests(unittest.TestCase):

    def test_connection_description_preserves_explicit_and_inferred_fields(self):
        hypothesis = extract_problem_hypotheses(
            "An app connecting local farmers directly with customers"
        )[0]

        self.assertEqual(hypothesis.actor, "local farmers")
        self.assertIn(
            "local farmers connect directly with customers",
            hypothesis.interaction,
        )
        self.assertIn("friction", hypothesis.inferred_fields)
        self.assertLess(hypothesis.confidence, 1.0)

    def test_generic_helping_patterns_work_across_domains(self):
        examples = (
            (
                "An app helping students find reliable study partners",
                "students",
                "find reliable study partners",
            ),
            (
                "A platform helping small businesses find affordable designers",
                "small businesses",
                "find affordable designers",
            ),
            (
                "A tool helping developers understand large unfamiliar codebases",
                "developers",
                "understand large unfamiliar codebases",
            ),
        )

        for description, actor, activity in examples:
            with self.subTest(description=description):
                hypothesis = extract_problem_hypotheses(description)[0]

                self.assertEqual(hypothesis.actor, actor)
                self.assertEqual(hypothesis.activity, activity)
                self.assertIsNotNone(hypothesis.friction)
                self.assertIsNotNone(hypothesis.goal)


if __name__ == "__main__":
    unittest.main()