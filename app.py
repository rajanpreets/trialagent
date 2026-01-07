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

# Initialize nested asyncio for Streamlit
nest_asyncio.apply()

# --- CONFIGURATION ---
DB_URL = "postgresql://neondb_owner:npg_XU5awkQ1FOZW@ep-falling-field-adfv4ciy-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require"

def ensure_playwright_browsers():
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception:
        pass

async def get_chictr_data_async(chictr_id):
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
        
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            html_content = await page.content()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('p', class_='en')
            trial_name = title_tag.get_text(strip=True) if title_tag else "Unknown Trial"
            
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
            cur.execute("SELECT chictr_id FROM monitored_trials ORDER BY chictr_id ASC")
            return [row['chictr_id'] for row in cur.fetchall()]

async def run_sync_for_id(chictr_id):
    data, error = await get_chictr_data_async(chictr_id)
    if error: return False
    internal_id = upsert_monitored_trial(chictr_id, data['name'], data['url'])
    return save_snapshot_if_changed(internal_id, data['raw_content'], data['hash'])

# --- UI LOGIC ---
st.set_page_config(page_title="ChiCTR Intelligence", layout="wide")
st.title("🧪 Clinical Trial Intelligence System")

# Controls
with st.expander("⚙️ Platform Controls", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        new_id = st.text_input("Add/Sync New ID")
        if st.button("🔍 Sync ID"):
            with st.spinner(f"Syncing {new_id}..."):
                asyncio.run(run_sync_for_id(new_id))
                st.success(f"Trial {new_id} synced.")
    with c2:
        st.write("Update All Monitored Trials")
        if st.button("🔄 Scrape Entire Portfolio"):
            ids = get_all_monitored_ids()
            progress = st.progress(0)
            for i, cid in enumerate(ids):
                asyncio.run(run_sync_for_id(cid))
                progress.progress((i + 1) / len(ids))
            st.balloons()

st.divider()

# Viewer
view_id = st.selectbox("Select Trial to View Analysis", options=[""] + get_all_monitored_ids())

if view_id:
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT s.raw_content, s.scraped_at, s.content_hash
                FROM trial_snapshots s
                JOIN monitored_trials m ON s.trial_id = m.id
                WHERE m.chictr_id = %s
                ORDER BY s.scraped_at DESC LIMIT 2
            """, (view_id,))
            history = cur.fetchall()

    if history:
        st.info(f"📍 Viewing: **{view_id}** | Last updated: {history[0]['scraped_at']}")
        
        # LOGIC: Highlight changes if 2 versions exist, otherwise show normal
        if len(history) >= 2:
            st.subheader("🚩 Protocol Changes Detected")
            st.caption("Additions are highlighted in green, deletions in red.")
            
            # Generate the full document diff
            diff = difflib.HtmlDiff().make_file(
                history[1]['raw_content'].splitlines(),
                history[0]['raw_content'].splitlines(),
                context=False # Set to False to show the FULL document
            )
            st.components.v1.html(diff, height=800, scrolling=True)
        else:
            st.subheader("📄 Protocol Information (First Snapshot)")
            st.warning("No previous version found for comparison. This is the baseline data.")
            st.text_area("Full Scraped Data", value=history[0]['raw_content'], height=600)
    else:
        st.error("No data found for this ID. Please click Sync first.")

# Sidebar
with st.sidebar:
    st.header("📋 Monitoring List")
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT chictr_id, trial_name FROM monitored_trials ORDER BY id DESC")
                for item in cur.fetchall():
                    st.markdown(f"**{item['chictr_id']}**")
                    st.caption(item['trial_name'])
                    st.divider()
    except: pass
