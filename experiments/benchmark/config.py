# Discovery test queries (used against PullPush, which does Reddit-wide FTS) ---
TEST_QUERIES = [
    "AI startup",
    "CRM for freelancers",
    "fitness app",
    "budget tracker",
    "no code app builder",
]

# Retrieval test subreddits (used against Arctic Shift, which needs a known subreddit) 
KNOWN_SUBREDDITS = [
    "startups",
    "SaaS",
    "Entrepreneur",
    "indiehackers",
    "SideProject",
]

# Date window for the 'last ~90 days' requirement 
DAYS_WINDOW = 90

# Endpoints
PULLPUSH_BASE = "https://api.pullpush.io/reddit"
ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"

# Networking behavior
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0
RESULTS_PER_QUERY = 50  

# PRAW credentials, read from environment ---
REDDIT_CLIENT_ID_ENV = "REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET_ENV = "REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT_ENV = "REDDIT_USER_AGENT"
REDDIT_DEFAULT_USER_AGENT = "community-intelligence-benchmark/0.1 (university capstone project)"

RESULTS_DIR = "results"
