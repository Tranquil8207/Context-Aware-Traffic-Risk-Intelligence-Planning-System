"""Supabase client factory. Reads credentials from environment variables.

Copy .env.example to .env and fill in your project's values before use.
"""

import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set. Copy .env.example to "
            ".env and fill in your project's values."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


if __name__ == "__main__":
    get_client()
    print(f"Connected to Supabase project: {SUPABASE_URL}")
