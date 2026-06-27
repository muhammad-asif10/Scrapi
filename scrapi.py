from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from fake_useragent import UserAgent, FakeUserAgentError
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from pymongo import MongoClient
import google.generativeai as genai
import spacy
import numpy as np
import pandas as pd
import datetime
import time
import random
import os
from dotenv import load_dotenv
from pathlib import Path


# ============================================================
# SETUP — Load everything once
# ============================================================
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

print("Loading models...")

# ---- Required assets checks ----
# spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. Run:\n"
        "python -m spacy download en_core_web_sm"
    )

# NLTK vader lexicon
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Secrets from environment (DO NOT hardcode in source) ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI environment variable.")

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-3.1-flash-lite")  # stable model name

client = MongoClient(MONGO_URI)
collection = client["news_db"]["headlines"]

print("Ready.\n")

# ============================================================
# CORE FUNCTIONS
# ============================================================

def get_user_agent():
    try:
        return UserAgent().random
    except FakeUserAgentError:
        return (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )

def fetch_page(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)  # True for stable scraping
            context = browser.new_context(user_agent=get_user_agent())
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=45000)

            # wait for at least one headline link
            page.wait_for_selector("a.story__link", timeout=15000)

            html = page.content()
            browser.close()
            time.sleep(random.uniform(1.0, 2.0))
            return html
    except PlaywrightTimeoutError:
        print("  Fetch error: timeout while loading Dawn.")
        return None
    except Exception as e:
        print(f"  Fetch error: {e}")
        return None

def parse_headlines(html):
    soup = BeautifulSoup(html, "html.parser")

    # FIX 1: correct selector (class is on <a>, not <h2>)
    link_tags = soup.find_all("a", class_="story__link")

    headlines = []
    for a in link_tags:
        text = a.get_text(strip=True)
        href = a.get("href")

        if not text or not href:
            continue

        # FIX 2: normalize URL
        if href.startswith("/"):
            href = "https://www.dawn.com" + href

        headlines.append({
            "text": text,
            "url": href,
            "source": "Dawn",
            "scraped_at": str(datetime.date.today())
        })

    return headlines

def extract_nlp(text):
    doc = nlp(text)
    entities = {
        "persons": [e.text for e in doc.ents if e.label_ == "PERSON"],
        "locations": [e.text for e in doc.ents if e.label_ == "GPE"],
        "organizations": [e.text for e in doc.ents if e.label_ == "ORG"],
        "dates": [e.text for e in doc.ents if e.label_ == "DATE"],
    }

    keywords = [
        token.text.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]

    compound = sia.polarity_scores(text)["compound"]
    sentiment = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    embedding = embedder.encode(text).tolist()

    return {
        "entities": entities,
        "keywords": keywords,
        "sentiment": sentiment,
        "sentiment_score": round(compound, 3),
        "embedding": embedding
    }

def clean(headlines):
    if not headlines:
        return []
    df = pd.DataFrame(headlines)
    df = df.dropna(subset=["text", "url"])
    df = df.drop_duplicates(subset=["text", "url"])
    return df.to_dict("records")

def store(headlines):
    new_count = 0
    for h in headlines:
        # dedupe on text+url (better than text only)
        if not collection.find_one({"text": h["text"], "url": h["url"]}):
            nlp_data = extract_nlp(h["text"])
            collection.insert_one({**h, **nlp_data})
            new_count += 1
    return new_count

def retrieve_semantic(query, limit=5):
    query_embedding = embedder.encode(query)
    all_docs = list(collection.find({"embedding": {"$exists": True}}))
    if not all_docs:
        return []

    scores = []
    qnorm = np.linalg.norm(query_embedding) or 1.0

    for doc in all_docs:
        emb = doc.get("embedding")
        if not emb:
            continue
        doc_embedding = np.array(emb)
        dnorm = np.linalg.norm(doc_embedding) or 1.0
        similarity = float(np.dot(query_embedding, doc_embedding) / (qnorm * dnorm))
        scores.append((similarity, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:limit]]

def build_prompt(question, context_docs):
    context = ""
    for i, doc in enumerate(context_docs, 1):
        context += f"{i}. [{doc.get('scraped_at','unknown')}] {doc.get('text','')}\n"
        context += f"   Sentiment: {doc.get('sentiment', 'unknown')}\n"

    return f"""You are an AI assistant analyzing Pakistani news headlines.

CONTEXT — Recent headlines from Dawn News:
{context}

Based ONLY on the headlines above, answer this question:
{question}

Rules:
- Only use information from the provided headlines
- If headlines lack sufficient information, say so clearly
- Be concise and cite relevant headline dates
"""

def generate_answer(prompt):
    response = gemini.generate_content(prompt)
    return getattr(response, "text", "No response text returned.")

# ============================================================
# MENU ACTIONS
# ============================================================

def action_scrape():
    print("\n  Scraping Dawn News...")
    html = fetch_page("https://www.dawn.com")
    if not html:
        print("  Failed to fetch page.")
        return

    raw = parse_headlines(html)
    cleaned = clean(raw)

    if not cleaned:
        print("  No headlines parsed. Check selectors/site structure.")
        return

    new = store(cleaned)
    print(f"  Done. Parsed {len(raw)} | Cleaned {len(cleaned)} | Stored new {new}")
    print(f"  Total in database: {collection.count_documents({})}")

def action_nlp_stats():
    total = collection.count_documents({})
    if total == 0:
        print("\n  Database is empty. Run scraper first.")
        return

    pos = collection.count_documents({"sentiment": "positive"})
    neg = collection.count_documents({"sentiment": "negative"})
    neu = collection.count_documents({"sentiment": "neutral"})

    all_docs = list(collection.find({"entities.locations": {"$exists": True}}, {"entities.locations": 1}))
    all_locs = []
    for doc in all_docs:
        all_locs.extend(doc.get("entities", {}).get("locations", []))

    if all_locs:
        loc_series = pd.Series(all_locs)
        top_locs = loc_series.value_counts().head(5).to_string()
    else:
        top_locs = "No locations found."

    print(f"""
  NLP Summary
  ───────────────────────────────
  Total headlines:  {total}

  Sentiment breakdown:
    Positive  {pos:>4}
    Negative  {neg:>4}
    Neutral   {neu:>4}

  Top locations mentioned:
{top_locs}
    """)

def action_ask():
    question = input("\n  Your question: ").strip()
    if not question:
        print("  No question entered.")
        return

    print("  Searching database...")
    docs = retrieve_semantic(question, limit=5)

    if not docs:
        print("  No relevant documents found. Run scraper first.")
        return

    print(f"  Found {len(docs)} relevant headlines")
    print("  Generating answer...\n")

    prompt = build_prompt(question, docs)
    answer = generate_answer(prompt)

    print("  Answer:")
    print("  " + "─" * 50)
    for line in answer.split("\n"):
        print(f"  {line}")

def action_export():
    total = collection.count_documents({})
    if total == 0:
        print("\n  Database is empty. Run scraper first.")
        return

    docs = list(collection.find({}, {"_id": 0, "embedding": 0}))
    df = pd.DataFrame(docs)

    if "entities" in df.columns:
        df["persons"] = df["entities"].apply(lambda x: ", ".join(x.get("persons", [])) if isinstance(x, dict) else "")
        df["locations"] = df["entities"].apply(lambda x: ", ".join(x.get("locations", [])) if isinstance(x, dict) else "")
        df["organizations"] = df["entities"].apply(lambda x: ", ".join(x.get("organizations", [])) if isinstance(x, dict) else "")
        df = df.drop(columns=["entities"])

    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")

    filename = f"news_report_{datetime.date.today()}.csv"
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"\n  Exported {len(df)} records to {filename}")

# ============================================================
# MAIN LOOP
# ============================================================

def print_header():
    print("""
╔══════════════════════════════════════════╗
║     Pakistan News Intelligence CLI       ║
║     Powered by Dawn + Gemini + MongoDB   ║
╚══════════════════════════════════════════╝""")

def print_menu():
    print("""
  [1]  Scrape latest headlines
  [2]  Show NLP & sentiment stats
  [3]  Ask a question about the news
  [4]  Export full report to CSV
  [0]  Exit
""")

def main():
    print_header()
    while True:
        print_menu()
        choice = input("  Choose: ").strip()

        if choice == "1":
            action_scrape()
        elif choice == "2":
            action_nlp_stats()
        elif choice == "3":
            action_ask()
        elif choice == "4":
            action_export()
        elif choice == "0":
            print("\n  Exiting. Goodbye.\n")
            break
        else:
            print("\n  Invalid choice. Enter 0-4.")

if __name__ == "__main__":
    main()