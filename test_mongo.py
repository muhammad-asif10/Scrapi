import os
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

MONGO_URI = os.getenv("MONGO_URI")
print(f"Testing: {MONGO_URI[:60]}...")

try:
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=10000
    )
    client.admin.command("ping")
    print("✓ Connection successful!")
except Exception as e:
    print(f"✗ Connection failed: {e}")