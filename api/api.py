"""
WC 2026 Match Outcome Predictor — FastAPI
-----------------------------------------
Endpoints:
  GET  /                        health check
  POST /predict                 predict a single match
  POST /predict/batch           predict multiple matches
  POST /result                  add a completed match result
  GET  /results                 list all added results
  GET  /fixtures                list all WC 2026 fixtures + predictions
  POST /simulate                Monte Carlo tournament simulator

Run locally:
  pip install fastapi uvicorn pandas numpy xgboost scikit-learn
  uvicorn api:app --reload --port 8000

Deploy free:
  Railway → connect GitHub repo → set start command to above
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pickle, json, os, random
import pandas as pd
import numpy as np

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="WC 2026 Match Predictor",
    description="XGBoost model trained on 47k+ international matches",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts ───────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, '..', 'model')

with open(os.path.join(MODEL_DIR, 'calibrated_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
    LE = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'feature_cols.json')) as f:
    FEATURE_COLS = json.load(f)
with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
    METADATA = json.load(f)

# ── Load match data for form computation ──────────────────────────────────────
DATA_DIR = os.path.join(BASE, '..', 'data', 'processed')

all_matches = pd.read_csv(
    os.path.join(DATA_DIR, 'matches_clean.csv'), parse_dates=['date']
).sort_values('date').reset_index(drop=True)

rankings = pd.read_csv(
    os.path.join(DATA_DIR, 'rankings_clean.csv'), parse_dates=['date']
)

# Build ranking lookup
rk_lookup = {}
for team, grp in rankings.groupby('team'):
    rk_lookup[team] = grp.sort_values('date')[['date','fifa_rank','fifa_points']].values

# Added WC 2026 results (in-memory, persists while server is running)
added_results = []

# ── Constants ─────────────────────────────────────────────────────────────────
REF_DATE = pd.Timestamp('2026-06-11')
LAMBDA   = 0.1
COMP_WEIGHTS = {
    'FIFA World Cup': 3.0, 'UEFA Euro': 2.5, 'Copa América': 2.5,
    'Africa Cup of Nations': 2.2, 'AFC Asian Cup': 2.2,
    'CONCACAF Gold Cup': 2.0, 'UEFA Nations League': 1.8,
    'FIFA World Cup qualification': 1.5, 'Friendly': 0.5,
}

CONF_MAP = {
    **{t:'UEFA' for t in ['England','France','Germany','Spain','Italy','Portugal',
                           'Netherlands','Belgium','Croatia','Denmark','Switzerland',
                           'Poland','Sweden','Ukraine','Austria','Czech Republic',
                           'Czechia','Serbia','Hungary','Scotland','Wales','Norway',
                           'Turkey','Turkiye','Kosovo','Bosnia and Herzegovina',
                           'Georgia','Slovakia','Slovenia','Albania','Romania']},
    **{t:'CONMEBOL' for t in ['Brazil','Argentina','Colombia','Uruguay','Chile',
                               'Ecuador','Peru','Venezuela','Paraguay','Bolivia']},
    **{t:'CONCACAF' for t in ['Mexico','USA','Canada','Costa Rica','Honduras',
                               'Jamaica','Panama','Haiti','Curacao']},
    **{t:'CAF' for t in ['Morocco','Senegal','Nigeria','Ghana','Cameroon',
                          'Egypt',"Cote d'Ivoire",'Ivory Coast','Mali','Tunisia',
                          'Algeria','South Africa','Cabo Verde','Cape Verde',
                          'Angola','Congo DR','Kenya']},
    **{t:'AFC' for t in ['Japan','Korea Republic','South Korea','IR Iran','Iran',
                          'Saudi Arabia','Australia','Qatar','Iraq','Uzbekistan',
                          'China','India','Jordan']},
}

# ── Helper functions ──────────────────────────────────────────────────────────
def get_comp_weight(t):
    for k, w in COMP_WEIGHTS.items():
        if k.lower() in str(t).lower(): return w
    return 1.0

def recompute_weights(df):
    df['years_ago']    = (REF_DATE - df['date']).dt.days / 365.25
    df['time_weight']  = np.exp(-LAMBDA * df['years_ago'])
    df['comp_weight']  = df['tournament'].apply(get_comp_weight)
    df['match_weight'] = df['time_weight'] * df['comp_weight']
    return df

all_matches = recompute_weights(all_matches)

def fast_lookup(team, match_date):
    if team not in rk_lookup: return np.nan, np.nan
    records = rk_lookup[team]
    dates   = records[:, 0]
    idx     = np.searchsorted(dates, match_date, side='right') - 1
    if idx < 0: return np.nan, np.nan
    return float(records[idx, 1]), float(records[idx, 2])

def compute_team_form(team, before_date, df, n=10):
    home_mask = (df['home_team'] == team) & (df['date'] < before_date)
    away_mask = (df['away_team'] == team) & (df['date'] < before_date)
    team_matches = df[home_mask | away_mask].tail(n).copy()
    if len(team_matches) == 0:
        return {k: np.nan for k in ['win_rate','draw_rate','loss_rate',
            'goals_scored_avg','goals_conceded_avg','goal_diff_avg','sos']}
    records = []
    for _, row in team_matches.iterrows():
        is_home  = (row['home_team'] == team)
        scored   = row['home_score'] if is_home else row['away_score']
        conceded = row['away_score'] if is_home else row['home_score']
        opp_pts  = row.get('away_fifa_points', np.nan) if is_home else row.get('home_fifa_points', np.nan)
        outcome  = row['outcome']
        win  = 1 if (is_home and outcome=='H') or (not is_home and outcome=='A') else 0
        draw = 1 if outcome == 'D' else 0
        loss = 1 if (is_home and outcome=='A') or (not is_home and outcome=='H') else 0
        w = row['match_weight']
        records.append({'win':w*win,'draw':w*draw,'loss':w*loss,
                        'scored':w*scored,'conceded':w*conceded,'opp_pts':opp_pts,'w':w})
    r = pd.DataFrame(records)
    W = r['w'].sum()
    return {
        'win_rate':           r['win'].sum()/W,
        'draw_rate':          r['draw'].sum()/W,
        'loss_rate':          r['loss'].sum()/W,
        'goals_scored_avg':   r['scored'].sum()/W,
        'goals_conceded_avg': r['conceded'].sum()/W,
        'goal_diff_avg':      (r['scored'].sum()-r['conceded'].sum())/W,
        'sos':                r['opp_pts'].mean(),
    }

def compute_h2h(home, away, before_date, df, n=5):
    mask = (
        ((df['home_team']==home)&(df['away_team']==away)) |
        ((df['home_team']==away)&(df['away_team']==home))
    ) & (df['date'] < before_date)
    meetings = df[mask].tail(n)
    if len(meetings) == 0:
        return {'h2h_home_win_rate':np.nan,'h2h_draw_rate':np.nan,
                'h2h_home_goals_avg':np.nan,'h2h_away_goals_avg':np.nan,'h2h_n':0}
    results = []
    for _, row in meetings.iterrows():
        if row['home_team'] == home:
            win = 1 if row['outcome']=='H' else 0
            draw= 1 if row['outcome']=='D' else 0
            hg, ag = row['home_score'], row['away_score']
        else:
            win = 1 if row['outcome']=='A' else 0
            draw= 1 if row['outcome']=='D' else 0
            hg, ag = row['away_score'], row['home_score']
        results.append({'win':win,'draw':draw,'hg':hg,'ag':ag})
    r = pd.DataFrame(results)
    return {'h2h_home_win_rate':r['win'].mean(),'h2h_draw_rate':r['draw'].mean(),
            'h2h_home_goals_avg':r['hg'].mean(),'h2h_away_goals_avg':r['ag'].mean(),'h2h_n':len(r)}

def build_features(home_team, away_team, match_date, is_neutral, tournament_weight):
    h_rank, h_pts = fast_lookup(home_team, match_date)
    a_rank, a_pts = fast_lookup(away_team, match_date)
    hf  = compute_team_form(home_team, match_date, all_matches)
    af  = compute_team_form(away_team, match_date, all_matches)
    h2h = compute_h2h(home_team, away_team, match_date, all_matches)
    home_conf = CONF_MAP.get(home_team, 'UNKNOWN')
    away_conf = CONF_MAP.get(away_team, 'UNKNOWN')

    row = {
        'home_fifa_rank':              h_rank,
        'away_fifa_rank':              a_rank,
        'home_fifa_points':            h_pts,
        'away_fifa_points':            a_pts,
        'rank_diff':                   (a_rank-h_rank) if (pd.notna(a_rank) and pd.notna(h_rank)) else 0,
        'points_diff':                 (h_pts-a_pts)   if (pd.notna(h_pts)  and pd.notna(a_pts))  else 0,
        'home_form_win_rate':          hf['win_rate'],
        'home_form_draw_rate':         hf['draw_rate'],
        'home_form_loss_rate':         hf['loss_rate'],
        'home_form_goals_scored_avg':  hf['goals_scored_avg'],
        'home_form_goals_conceded_avg':hf['goals_conceded_avg'],
        'home_form_goal_diff_avg':     hf['goal_diff_avg'],
        'home_form_sos':               hf['sos'],
        'away_form_win_rate':          af['win_rate'],
        'away_form_draw_rate':         af['draw_rate'],
        'away_form_loss_rate':         af['loss_rate'],
        'away_form_goals_scored_avg':  af['goals_scored_avg'],
        'away_form_goals_conceded_avg':af['goals_conceded_avg'],
        'away_form_goal_diff_avg':     af['goal_diff_avg'],
        'away_form_sos':               af['sos'],
        'form_win_rate_diff':          (hf['win_rate']-af['win_rate'])               if pd.notna(hf['win_rate'])        else 0,
        'form_goal_diff_diff':         (hf['goal_diff_avg']-af['goal_diff_avg'])     if pd.notna(hf['goal_diff_avg'])   else 0,
        'form_goals_scored_diff':      (hf['goals_scored_avg']-af['goals_scored_avg'])if pd.notna(hf['goals_scored_avg'])else 0,
        'form_goals_conceded_diff':    (hf['goals_conceded_avg']-af['goals_conceded_avg']) if pd.notna(hf['goals_conceded_avg']) else 0,
        'sos_diff':                    (hf['sos']-af['sos']) if (pd.notna(hf['sos']) and pd.notna(af['sos'])) else 0,
        'h2h_home_win_rate':           h2h['h2h_home_win_rate'],
        'h2h_draw_rate':               h2h['h2h_draw_rate'],
        'h2h_home_goals_avg':          h2h['h2h_home_goals_avg'],
        'h2h_away_goals_avg':          h2h['h2h_away_goals_avg'],
        'h2h_n':                       h2h['h2h_n'],
        'is_neutral':                  int(is_neutral),
        'tournament_weight':           tournament_weight,
        'same_conf':                   int(home_conf == away_conf),
        'home_conf_UEFA':              int(home_conf == 'UEFA'),
        'home_conf_CONMEBOL':          int(home_conf == 'CONMEBOL'),
        'home_conf_CONCACAF':          int(home_conf == 'CONCACAF'),
        'home_conf_CAF':               int(home_conf == 'CAF'),
        'home_conf_AFC':               int(home_conf == 'AFC'),
    }
    for k, v in row.items():
        if isinstance(v, float) and np.isnan(v):
            row[k] = 0.33 if 'rate' in k else 0.0
    return pd.DataFrame([row])[FEATURE_COLS]

# ── Pydantic schemas ───────────────────────────────────────────────────────────
class MatchRequest(BaseModel):
    home_team:          str
    away_team:          str
    match_date:         Optional[str] = '2026-06-15'
    is_neutral:         Optional[bool] = True
    tournament_weight:  Optional[float] = 3.0

class BatchRequest(BaseModel):
    matches: list[MatchRequest]



@app.get('/results')
def list_results():
    return {'results': added_results, 'count': len(added_results)}

@app.post('/simulate')
def simulate(req: SimulateRequest):
    """Monte Carlo bracket simulation."""
    if len(req.teams) < 2 or (len(req.teams) & (len(req.teams)-1)) != 0:
        raise HTTPException(400, 'teams must be a power of 2 (2, 4, 8, 16...)')

    wins = {t: 0 for t in req.teams}
    for _ in range(req.n_sims):
        bracket = req.teams.copy()
        random.shuffle(bracket)
        while len(bracket) > 1:
            next_round = []
            for i in range(0, len(bracket), 2):
                a, b = bracket[i], bracket[i+1]
                p = predict(MatchRequest(home_team=a, away_team=b))
                outcome = random.choices(
                    ['home','draw','away'],
                    weights=[p['home_win'], p['draw'], p['away_win']]
                )[0]
                winner = a if outcome == 'home' else (b if outcome == 'away'
                         else (a if random.random() < 0.5 else b))
                next_round.append(winner)
            bracket = next_round
        wins[bracket[0]] += 1

    return {
        'simulations': req.n_sims,
        'win_probabilities': {
            t: round(v/req.n_sims*100, 1)
            for t, v in sorted(wins.items(), key=lambda x: -x[1])
        }
    }
