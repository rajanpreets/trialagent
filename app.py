import streamlit as st
import asyncio
import hashlib
import json
import re
import os
import subprocess
import psycopg2
from psycopg2.extras import RealDictCursor
import difflib
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import nest_asyncio

# Initialize nested asyncio for Streamlit's environment
nest_asyncio.apply()

# --- CONFIGURATION ---
DB_URL = "postgresql://neondb_owner:npg_XU5awkQ1FOZW@ep-falling-field-adfv4ciy-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

def ensure_playwright_browsers():
    """Ensures Chromium is installed in the cloud environment."""
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception:
        pass

async def get_chictr_data_async(chictr_id):
    """Scraper logic with Manual Stealth to avoid TypeErrors."""
    ensure_playwright_browsers()
    url = f"https://www.chictr.org.cn/showprojEN.html?proj={chictr_id}"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, 
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        # Manual Stealth
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
        """)
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            html_content = await page.content()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract Title for the monitored_trials table
            title_tag = soup.find('p', class_='en')
            trial_name = title_tag.get_text(strip=True) if title_tag else "Unknown Trial"
            
            # Clean Chinese text tags
            for cn_tag in soup.find_all(class_='cn'): 
                cn_tag.decompose()

            raw_text = soup.get_text("\n", strip=True)
            content_hash = hashlib.sha256(raw_text.encode()).hexdigest()
            
            await browser.close()
            return {"raw_content": raw_text, "hash": content_hash, "name": trial_name, "url": url}, None
        except Exception as e:
            return None, str(e)

# --- DB HELPERS ---

def upsert_monitored_trial(chictr_id, trial_name, trial_url):
    """Adds a new ID to monitored_trials if it doesn't exist."""
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO monitored_trials (chictr_id, trial_name, trial_url)
                VALUES (%s, %s, %s)
                ON CONFLICT (chictr_id) DO UPDATE SET trial_name = EXCLUDED.trial_name
                RETURNING id;
            """, (chictr_id, trial_name, trial_url))
            return cur.fetchone()[0]

def save_snapshot_if_changed(trial_internal_id, raw_content, content_hash):
    """Saves to trial_snapshots only if the hash is new."""
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content_hash FROM trial_snapshots WHERE trial_id = %s ORDER BY scraped_at DESC LIMIT 1", (trial_internal_id,))
            last_snap = cur.fetchone()

            if not last_snap or last_snap[0] != content_hash:
                cur.execute("INSERT INTO trial_snapshots (trial_id, raw_content, content_hash) VALUES (%s, %s, %s)", 
                            (trial_internal_id, raw_content, content_hash))
                conn.commit()
                return True
            return False

def get_all_monitored_ids():
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT chictr_id FROM monitored_trials")
            return [row['chictr_id'] for row in cur.fetchall()]

# --- UI LOGIC ---

async def run_sync_for_id(chictr_id):
    data, error = await get_chictr_data_async(chictr_id)
    if error:
        st.error(f"Error syncing {chictr_id}: {error}")
        return False
    
    # 1. Ensure it is in monitored_trials
    internal_id = upsert_monitored_trial(chictr_id, data['name'], data['url'])
    
    # 2. Save snapshot
    changed = save_snapshot_if_changed(internal_id, data['raw_content'], data['hash'])
    return changed

# --- STREAMLIT UI ---
st.set_page_config(page_title="ChiCTR HEOR Intel", layout="wide")
st.title("🧪 Clinical Trial Protocol Intelligence")

# Single ID Sync
st.subheader("Add or Update Single ID")
target_id = st.text_input("Enter ChiCTR ID (e.g., 297646)")

if st.button("🔍 Sync Single ID"):
    if target_id:
        with st.spinner(f"Scraping {target_id}..."):
            changed = asyncio.run(run_sync_for_id(target_id))
            if changed:
                st.success(f"✅ Changes detected and saved for {target_id}!")
            else:
                st.info(f"ℹ️ No changes for {target_id}.")
    else:
        st.warning("Please enter an ID.")

st.divider()

# Batch Sync
st.subheader("Platform Controls")
if st.button("🔄 Scrape All Monitored IDs"):
    ids = get_all_monitored_ids()
    if not ids:
        st.warning("No IDs are currently monitored.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chictr_id in enumerate(ids):
            status_text.text(f"Syncing {chictr_id} ({i+1}/{len(ids)})...")
            asyncio.run(run_sync_for_id(chictr_id))
            progress_bar.progress((i + 1) / len(ids))
        
        status_text.text("✅ All monitored IDs synced!")
        st.balloons()

st.divider()

# Comparison View
st.subheader("View Protocol Changes")
compare_id = st.selectbox("Select ID to compare", options=[""] + get_all_monitored_ids())

if compare_id:
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT s.raw_content, s.scraped_at 
                FROM trial_snapshots s
                JOIN monitored_trials m ON s.trial_id = m.id
                WHERE m.chictr_id = %s
                ORDER BY s.scraped_at DESC LIMIT 2
            """, (compare_id,))
            history = cur.fetchall()

    if len(history) < 2:
        st.warning("Not enough history for this ID yet. Sync it again later to see changes.")
    else:
        st.write(f"Comparing latest vs. snapshot from {history[1]['scraped_at']}")
        diff = difflib.HtmlDiff().make_file(
            history[1]['raw_content'].splitlines(),
            history[0]['raw_content'].splitlines(),
            context=True
        )
        st.components.v1.html(diff, height=600, scrolling=True)

# Sidebar
with st.sidebar:
    st.header("📋 Monitored Portfolio")
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT chictr_id, trial_name FROM monitored_trials ORDER BY date_added DESC")
                for item in cur.fetchall():
                    st.markdown(f"**{item['chictr_id']}**")
                    st.caption(item['trial_name'])
                    st.divider()
    except:
        st.write("Connecting...")
