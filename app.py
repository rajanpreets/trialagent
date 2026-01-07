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
from playwright_stealth import stealth_async
import nest_asyncio

# Setup for Streamlit's internal event loop
nest_asyncio.apply()

# --- DATABASE CONFIG ---
# Connection string for your Neon Database
DB_URL = "postgresql://neondb_owner:npg_XU5awkQ1FOZW@ep-falling-field-adfv4ciy-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

def ensure_playwright_browsers():
    """Automatically installs Chromium if it's missing in the cloud environment."""
    try:
        # Check if browser binaries are already available
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception as e:
        st.error(f"Playwright Browser Install Failed: {e}")

async def get_chictr_data_async(chictr_id):
    """The 'Ditto' v3 Scraper logic adjusted for async compatibility."""
    ensure_playwright_browsers()
    url = f"https://www.chictr.org.cn/showprojEN.html?proj={chictr_id}"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, 
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        # FIX: Correctly await the async stealth function
        await stealth_async(page)
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            html_content = await page.content()
            await browser.close()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove Chinese text tags
            for cn_tag in soup.find_all(class_='cn'): 
                cn_tag.decompose()

            # Data structure for Neon DB and Diffing
            results = {
                "metadata": {
                    "proj_id": chictr_id,
                    "scraped_at": "2026-01-07",
                    "hash": hashlib.sha256(html_content.encode()).hexdigest()
                },
                "fields": {}
            }

            # Field Extraction Logic
            for row in soup.find_all('tr'):
                label_cell = row.find('td', class_='left_title')
                if label_cell:
                    label = label_cell.get_text(strip=True).replace('：', '').replace(':', '')
                    value_cell = label_cell.find_next_sibling('td')
                    if value_cell:
                        results["fields"][label] = value_cell.get_text(" ", strip=True)

            # Smart Outcome Mapping
            primary_outcomes = []
            secondary_outcomes = []
            for table in soup.select('.subitem table'):
                text = table.get_text(" ", strip=True)
                if "Outcome" in text:
                    clean_val = re.sub(r'Outcome|Type|Measure.*', '', text).replace('：', '').strip()
                    if "Primary" in text: primary_outcomes.append(clean_val)
                    else: secondary_outcomes.append(clean_val)
            
            results["fields"]["Primary outcome"] = "\n".join(primary_outcomes) if primary_outcomes else "Not Found"
            results["fields"]["Secondary outcome"] = "\n".join(secondary_outcomes) if secondary_outcomes else "Not Found"

            # Sample Size Number Extraction
            sample_size_matches = re.findall(r'Sample size\s*：\s*(\d+)', html_content)
            if sample_size_matches:
                results["fields"]["Target Sample Size"] = sample_size_matches[0]

            return results, None
        except Exception as e:
            return None, str(e)

# --- DATABASE HANDLERS ---
def handle_sync_process(chictr_id):
    """Orchestrates scraping and database upsert."""
    with st.spinner("⚡ Fetching live data from ChiCTR..."):
        # Run the async scraper inside the Streamlit loop
        scraped_data, error = asyncio.run(get_chictr_data_async(chictr_id))
        
        if error:
            st.error(f"Scrape error: {error}")
            return False

        try:
            with psycopg2.connect(DB_URL) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Match schema: checked chictr_id
                    cur.execute("SELECT id FROM monitored_trials WHERE chictr_id = %s", (chictr_id,))
                    trial = cur.fetchone()
                    if not trial:
                        st.error("This ID is not in your monitored list. Please add it via SQL first.")
                        return False
                    
                    t_id = trial['id']
                    new_hash = scraped_data['metadata']['hash']
                    raw_text = json.dumps(scraped_data['fields']) # Save structured fields as text for diffing

                    # Check for updates
                    cur.execute("SELECT content_hash FROM trial_snapshots WHERE trial_id = %s ORDER BY scraped_at DESC LIMIT 1", (t_id,))
                    last_snap = cur.fetchone()

                    if last_snap and last_snap['content_hash'] == new_hash:
                        st.info("✅ Database is up to date. No protocol changes detected.")
                        return False
                    else:
                        cur.execute("""
                            INSERT INTO trial_snapshots (trial_id, raw_content, content_hash)
                            VALUES (%s, %s, %s)
                        """, (t_id, raw_text, new_hash))
                        conn.commit()
                        st.success("🔔 New version saved to snapshots!")
                        return True
        except Exception as e:
            st.error(f"Database error: {e}")
            return False

def get_recent_versions(chictr_id):
    """Retrieves the two latest versions for comparison."""
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

# --- STREAMLIT DASHBOARD ---
st.set_page_config(page_title="ChiCTR HEOR Intel", layout="wide")
st.title("🧪 ChiCTR Clinical Trial Intelligence")

selected_id = st.text_input("Enter ChiCTR ID (e.g., 297646)", value="297646")

col_btn1, col_btn2, _ = st.columns([1, 1, 4])

with col_btn1:
    if st.button("🔍 Sync & Save"):
        handle_sync_process(selected_id)

with col_btn2:
    if st.button("📜 View Changes"):
        history = get_recent_versions(selected_id)
        if len(history) < 2:
            st.warning("Insufficient history to generate a Red/Green comparison.")
        else:
            st.subheader(f"Comparison: {history[1]['scraped_at'].strftime('%Y-%m-%d')} vs Today")
            
            # Format JSON text back into lines for the HTML Diff engine
            old_lines = json.dumps(json.loads(history[1]['raw_content']), indent=2).splitlines()
            new_lines = json.dumps(json.loads(history[0]['raw_content']), indent=2).splitlines()

            diff_html = difflib.HtmlDiff().make_file(old_lines, new_lines, context=True)
            st.components.v1.html(diff_html, height=800, scrolling=True)

# Sidebar: Registry Explorer
with st.sidebar:
    st.header("📊 Monitoring Portfolio")
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT chictr_id, trial_name FROM monitored_trials")
                for item in cur.fetchall():
                    st.write(f"**{item['chictr_id']}**")
                    st.caption(item['trial_name'] if item['trial_name'] else "Unnamed Study")
                    st.divider()
    except:
        st.write("Awaiting database connection...")
