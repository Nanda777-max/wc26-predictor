"""
auto_update.py
--------------
Polls football-data.org every 30 minutes for new WC 2026 results
and pushes them to the FastAPI /result endpoint automatically.

No more manual add_wc26_result() calls.

Setup:
  pip install requests schedule
  python scripts/auto_update.py

Free API key: https://www.football-data.org/  (12 calls/min free tier)
"""

import requests
import schedule
import time
import json
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
FOOTBALL_DATA_API_KEY = os.getenv('FOOTBALL_DATA_API_KEY', 'YOUR_KEY_HERE')
PREDICTOR_API_URL     = os.getenv('PREDICTOR_API_URL', 'http://localhost:8000')
WC2026_COMPETITION_ID = 2000   # football-data.org competition ID for FIFA WC

HEADERS = {
    'X-Auth-Token': FOOTBALL_DATA_API_KEY,
}

# Track which matches we've already added so we don't double-post
posted_match_ids = set()
POSTED_IDS_FILE  = 'scripts/posted_ids.json'

def load_posted_ids():
    global posted_match_ids
    if os.path.exists(POSTED_IDS_FILE):
        with open(POSTED_IDS_FILE) as f:
            posted_match_ids = set(json.load(f))

def save_posted_ids():
    with open(POSTED_IDS_FILE, 'w') as f:
        json.dump(list(posted_match_ids), f)

def fetch_finished_matches():
    """Fetch all FINISHED WC 2026 matches from football-data.org."""
    url = f'https://api.football-data.org/v4/competitions/{WC2026_COMPETITION_ID}/matches'
    params = {'status': 'FINISHED'}
    try:
        res = requests.get(url, headers=HEADERS, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data.get('matches', [])
    except Exception as e:
        print(f'[{datetime.now()}] Fetch error: {e}')
        return []

def push_result(match):
    """Push one finished match to the predictor API."""
    match_id   = match['id']
    home_team  = match['homeTeam']['name']
    away_team  = match['awayTeam']['name']
    home_score = match['score']['fullTime']['home']
    away_score = match['score']['fullTime']['away']
    date       = match['utcDate'][:10]   # 'YYYY-MM-DD'

    if home_score is None or away_score is None:
        return  # score not yet available

    payload = {
        'home_team':  home_team,
        'away_team':  away_team,
        'home_score': home_score,
        'away_score': away_score,
        'date':       date,
    }

    try:
        res = requests.post(
            f'{PREDICTOR_API_URL}/result',
            json=payload, timeout=10
        )
        res.raise_for_status()
        print(f'[{datetime.now()}] ✅ Added: {home_team} {home_score}-{away_score} {away_team}')
        posted_match_ids.add(match_id)
        save_posted_ids()
    except Exception as e:
        print(f'[{datetime.now()}] Push error for match {match_id}: {e}')

def check_and_update():
    print(f'[{datetime.now()}] Checking for new results...')
    matches = fetch_finished_matches()
    new = 0
    for m in matches:
        if m['id'] not in posted_match_ids:
            push_result(m)
            new += 1
    print(f'[{datetime.now()}] Done. {new} new results pushed. Total tracked: {len(posted_match_ids)}')

# ── Scheduler ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('WC 2026 Auto-Updater starting...')
    print(f'  Predictor API: {PREDICTOR_API_URL}')
    print(f'  Polling every 30 minutes')

    load_posted_ids()
    check_and_update()  # run once immediately on start

    schedule.every(30).minutes.do(check_and_update)

    while True:
        schedule.run_pending()
        time.sleep(60)
