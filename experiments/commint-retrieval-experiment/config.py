import os

from dotenv import load_dotenv


load_dotenv()


ARCTIC_SHIFT_BASE = os.getenv(
    "ARCTIC_SHIFT_BASE",
    "https://arctic-shift.photon-reddit.com/api",
)

REQUEST_TIMEOUT = int(
    os.getenv(
        "REQUEST_TIMEOUT",
        "30",
    )
)

ARCTIC_MAX_RETRIES = int(
    os.getenv(
        "ARCTIC_MAX_RETRIES",
        "4",
    )
)

ARCTIC_COMMENT_LIMIT = int(
    os.getenv(
        "ARCTIC_COMMENT_LIMIT",
        "50",
    )
)