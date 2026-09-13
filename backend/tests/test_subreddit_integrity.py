import unittest
from types import SimpleNamespace

from app.services.community import _belongs_to_subreddit


class SubredditIntegrityTests(unittest.TestCase):

    def test_rejects_user_profile_post_for_requested_subreddit(self):
        post = SimpleNamespace(
            subreddit="u_Consistent-Mud-9521"
        )

        self.assertFalse(
            _belongs_to_subreddit(post, "smallfarms")
        )

    def test_accepts_case_insensitive_subreddit_match(self):
        post = SimpleNamespace(
            subreddit="SmallFarms"
        )

        self.assertTrue(
            _belongs_to_subreddit(post, "smallfarms")
        )


if __name__ == "__main__":
    unittest.main()