"""Live API credential verification test with minimal token consumption."""

import sys
import os
import site

# Link project root & venv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
venv_site = os.path.join(BASE_DIR, ".venv", "Lib", "site-packages")
if os.path.exists(venv_site):
    site.addsitedir(venv_site)

from src.config import settings
from src.storage.database import get_db_session, init_db
from src.storage.repository import DataRepository
from src.ingestion.exa_client import ExaNewsService
from apify_client import ApifyClient


def verify_apify_credentials():
    """Verify Apify token validity without consuming actor compute units."""
    print("\n--- 1. Testing Apify API Token ---")
    token = settings.APIFY_API_TOKEN
    assert token, "APIFY_API_TOKEN is empty!"
    print(f"Token: {token[:12]}...{token[-4:]}")

    client = ApifyClient(token=token)
    user_info = client.user().get()
    print(" Apify Authentication SUCCESSFUL!")
    print(f"• User ID: {user_info.get('id')}")
    print(f"• Username: {user_info.get('username')}")
    print(f"• Plan: {user_info.get('plan', {}).get('id', 'Free')}")
    return True


def verify_exa_credentials():
    """Verify Exa API key by executing 1 minimal query (num_results=2)."""
    print("\n--- 2. Testing Exa.ai API Key ---")
    key = settings.EXA_API_KEY
    assert key, "EXA_API_KEY is empty!"
    print(f"Key: {key[:8]}...{key[-4:]}")

    init_db()
    with get_db_session() as session:
        repo = DataRepository(session)
        service = ExaNewsService(repo, api_key=key)

        asset = repo.get_asset_by_ticker("0700.HK")
        assert asset, "Asset 0700.HK not found in database!"

        print(f"Executing minimal live Exa search for {asset.ticker} ({asset.name})...")
        inserted = service.fetch_and_store_news(asset, lookback_days=5, num_results=2)
        print(f" Exa.ai Live Search SUCCESSFUL! Ingested {inserted} real news articles.")
    return True


if __name__ == "__main__":
    try:
        apify_ok = verify_apify_credentials()
        exa_ok = verify_exa_credentials()
        print("\n=======================================================")
        print(" ALL CREDENTIALS VALIDATED SUCCESSFULLY WITH LIVE APIS!")
        print("=======================================================")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)
