import streamlit as st
import asyncio
import hashlib
import json
import re
import psycopg2
from psycopg2.extras import RealDictCursor
import difflib
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from playwright_stealth import stealth
import nest_asyncio

# Initialize nested asyncio for Streamlit
nest_asyncio.apply()

# --- CONFIGURATION & DATABASE CONNECTION ---
# Using the Neon Connection String provided
DB_URL = "postgresql://neondb_owner:npg_XU5awkQ1FOZW@ep-falling-field-adfv4ciy-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

# --- CORE SCRAPER (DITTO V3 LOGIC) ---
async def get_chictr_data(chictr_id):
    url = f"https://www.chictr.org.cn/showprojEN.html?proj={chictr_id}"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        page = await context.new_page()
        await stealth(page)
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            html_content = await page.content()
            await browser.close()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            # Clean Chinese text
            for cn_tag in soup.find_all(class_='cn'): 
                cn_tag.decompose()

            # Extract text for hashing and storage
            raw_text = soup.get_text("\n", strip=True)
            content_hash = hashlib.sha256(raw_text.encode()).hexdigest()
            
            return raw_text, content_hash
        except Exception as e:
            return None, str(e)

# --- DATABASE OPERATIONS ---
def handle_sync(chictr_id):
    """Scrapes data and saves to Neon if hash is different."""
    raw_content, new_hash = asyncio.run(get_chictr_data(chictr_id))
    
    if not raw_content:
        st.error(f"Scrape failed: {new_hash}")
        return False

    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get the record ID from monitored_trials
            cur.execute("SELECT id FROM monitored_trials WHERE chictr_id = %s", (chictr_id,))
            trial = cur.fetchone()
            
            if not trial:
                st.error("This ID is not in your monitored list. Please add it first.")
                return False
            
            internal_id = trial['id']

            # Check the latest snapshot in the database
            cur.execute("""
                SELECT content_hash FROM trial_snapshots 
                WHERE trial_id = %s ORDER BY scraped_at DESC LIMIT 1
            """, (internal_id,))
            last_snap = cur.fetchone()

            if last_snap and last_snap['content_hash'] == new_hash:
                st.info("✅ Database is already up to date. No changes found.")
                return False
            else:
                cur.execute("""
                    INSERT INTO trial_snapshots (trial_id, raw_content, content_hash)
                    VALUES (%s, %s, %s)
                """, (internal_id, raw_content, new_hash))
                conn.commit()
                st.success("🔔 Protocol update detected! New version saved.")
                return True

def fetch_comparison(chictr_id):
    """Returns the two most recent versions for diffing."""
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT s.raw_content, s.scraped_at 
                FROM trial_snapshots s
                JOIN monitored_trials m ON s.trial_id = m.id
                WHERE m.chictr_id = %s
                ORDER BY s.scraped_at DESC LIMIT 2
            """, (chictr_id,))
            return cur.fetchall()

# --- STREAMLIT UI ---
st.set_page_config(page_title="ChiCTR HEOR Intel", layout="wide")

st.markdown("# 🧪 ChiCTR Clinical Trial Monitor")
st.write("Real-time monitoring and Red/Green diffing for Chinese registries.")

# Target ID Input
target_id = st.text_input("Enter ChiCTR ID (e.g., 297646)", value="297646")

action_cols = st.columns([1, 1, 4])

with action_cols[0]:
    if st.button("🔍 Sync Data"):
        handle_sync(target_id)

with action_cols[1]:
    if st.button("📜 Show Diff"):
        history = fetch_comparison(target_id)
        if len(history) < 2:
            st.warning("Comparison requires at least two historical snapshots.")
        else:
            st.subheader(f"Changes Detected since {history[1]['scraped_at'].strftime('%Y-%m-%d')}")
            
            # Generate the Red/Green HTML
            diff_engine = difflib.HtmlDiff()
            diff_html = diff_engine.make_file(
                history[1]['raw_content'].splitlines(),
                history[0]['raw_content'].splitlines(),
                context=True,
                numlines=3
            )
            st.components.v1.html(diff_html, height=800, scrolling=True)

# Sidebar: Monitored Trials List
with st.sidebar:
    st.header("Monitored Trials")
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT chictr_id, trial_name FROM monitored_trials")
                for trial in cur.fetchall():
                    st.markdown(f"**{trial['chictr_id']}** \n*{trial['trial_name']}*")
                    st.divider()
    except:
        st.write("Database connecting...")
