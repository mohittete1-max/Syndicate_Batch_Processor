import os
import requests
import pandas as pd
import numpy as np
import pulp as lp
from collections import Counter
import plotly.express as px
import webbrowser

WORKSPACE_DIR = r"C:\Users\User\OneDrive\Desktop\Cricket\Syndicate_Batch_Processor"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

API_KEY = "4905f024-424c-4f6c-a2e6-b4e64f41f7bb"
BASE_URL = "https://api.cricapi.com/v1"

NAM_SA_POOL = [
    {"name": "Z Green", "role": "WK", "team": "NAM", "credits": 6.0, "points": 130.0},
    {"name": "A Volschenk", "role": "BAT", "team": "NAM", "credits": 8.0, "points": 198.0},
    {"name": "L Steenkamp", "role": "BAT", "team": "NAM", "credits": 7.5, "points": 115.0},
    {"name": "J Frylinck", "role": "AR", "team": "NAM", "credits": 7.0, "points": 194.0},
    {"name": "G Erasmus", "role": "AR", "team": "NAM", "credits": 8.0, "points": 372.0},
    {"name": "J Smit", "role": "AR", "team": "NAM", "credits": 7.0, "points": 193.0},
    {"name": "Nicol Loftie-Eaton", "role": "AR", "team": "NAM", "credits": 6.5, "points": 97.0},
    {"name": "J Balt", "role": "AR", "team": "NAM", "credits": 8.0, "points": 35.0},
    {"name": "R Trumpelmann", "role": "BOWL", "team": "NAM", "credits": 6.0, "points": 253.0},
    {"name": "J Brassell", "role": "BOWL", "team": "NAM", "credits": 7.0, "points": 111.0},
    {"name": "M Heingo", "role": "BOWL", "team": "NAM", "credits": 6.5, "points": 104.0},
    {"name": "B Scholtz", "role": "BOWL", "team": "NAM", "credits": 6.0, "points": 44.0},
    {"name": "R Hermann", "role": "WK", "team": "SA", "credits": 8.5, "points": 52.0},
    {"name": "C Esterhuizen", "role": "WK", "team": "SA", "credits": 7.0, "points": 38.0},
    {"name": "D Brevis", "role": "BAT", "team": "SA", "credits": 9.0, "points": 417.0},
    {"name": "L Pretorius", "role": "BAT", "team": "SA", "credits": 8.5, "points": 265.0},
    {"name": "de Zorzi", "role": "BAT", "team": "SA", "credits": 8.0, "points": 48.0},
    {"name": "E Bosch", "role": "AR", "team": "SA", "credits": 7.0, "points": 93.0},
    {"name": "P Subrayen", "role": "AR", "team": "SA", "credits": 6.5, "points": 247.0},
    {"name": "B Fortuin", "role": "BOWL", "team": "SA", "credits": 8.0, "points": 314.0},
    {"name": "D Jansen", "role": "BOWL", "team": "SA", "credits": 7.0, "points": 377.0},
    {"name": "K Maphaka", "role": "BOWL", "team": "SA", "credits": 8.0, "points": 71.0},
    {"name": "N Peter", "role": "BOWL", "team": "SA", "credits": 7.5, "points": 31.0}
]

UAE_INA_POOL = [
    {"name": "T Satish", "role": "WK", "team": "UAE-W", "credits": 7.0, "points": 2.0},
    {"name": "R Rajith", "role": "BAT", "team": "UAE-W", "credits": 6.0, "points": 27.0},
    {"name": "U Iyer", "role": "BAT", "team": "UAE-W", "credits": 6.0, "points": 0.0},
    {"name": "E Oza", "role": "AR", "team": "UAE-W", "credits": 6.0, "points": 50.0},
    {"name": "S Dharnidharka", "role": "AR", "team": "UAE-W", "credits": 6.0, "points": 28.0},
    {"name": "H Hotchandani", "role": "AR", "team": "UAE-W", "credits": 7.0, "points": 10.0},
    {"name": "S Maypriani", "role": "BOWL", "team": "UAE-W", "credits": 7.5, "points": 112.0},
    {"name": "V Mahesh", "role": "BOWL", "team": "UAE-W", "credits": 7.0, "points": 8.0},
    {"name": "S Kotte", "role": "BOWL", "team": "UAE-W", "credits": 6.5, "points": 5.0},
    {"name": "Putu Ayu Nanda", "role": "WK", "team": "INA-W", "credits": 6.0, "points": 17.0},
    {"name": "R Pangestuti", "role": "BAT", "team": "INA-W", "credits": 8.0, "points": 47.0},
    {"name": "M Corazon", "role": "BAT", "team": "INA-W", "credits": 6.5, "points": 32.0},
    {"name": "Luh Dewi", "role": "AR", "team": "INA-W", "credits": 8.0, "points": 196.0},
    {"name": "Made Putri", "role": "AR", "team": "INA-W", "credits": 6.0, "points": 87.0},
    {"name": "N Ariani", "role": "BOWL", "team": "INA-W", "credits": 8.0, "points": 194.0},
    {"name": "S Velic", "role": "BOWL", "team": "INA-W", "credits": 8.0, "points": 87.0},
    {"name": "D Paramitha", "role": "BOWL", "team": "INA-W", "credits": 6.0, "points": 24.0}
]

def fetch_venue_meteo(city_name):
    try:
        geo = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1", timeout=5).json()
        if not geo.get("results"):
            return {"temp": 25.0, "wind_speed": 10.0}
        w = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={geo['results'][0]['latitude']}&longitude={geo['results'][0]['longitude']}&current_weather=true", timeout=5).json()
        return {
            "temp": w.get("current_weather", {}).get("temperature", 25.0),
            "wind_speed": w.get("current_weather", {}).get("windspeed", 10.0)
        }
    except:
        return {"temp": 25.0, "wind_speed": 10.0}

def apply_adjustments(df, meteo, venue_type, toss):
    df = df.copy()
    if meteo["wind_speed"] > 22.0:
        df.loc[df['role'] == 'BOWL', 'proj'] *= 1.12
    if meteo["temp"] > 32.0:
        df.loc[df['role'] == 'BAT', 'proj'] *= 1.08
    if venue_type == "spinner":
        df.loc[df['role'].isin(['AR', 'BOWL']), 'proj'] *= 1.12
    elif venue_type == "seamer":
        df.loc[df['role'] == 'BOWL', 'proj'] *= 1.18
        df.loc[df['role'] == 'BAT', 'proj'] *= 0.92
    if toss:
        df.loc[(df['team'] == toss["bowling_first"]) & (df['role'] == 'BOWL'), 'proj'] *= 1.15
        df.loc[(df['team'] == toss["bowling_first"]) & (df['role'] == 'AR'), 'proj'] *= 1.08
    return df

def prepare_data(pool):
    df = pd.DataFrame(pool).rename(columns={"points": "proj"})
    df["proj"] = df["proj"].astype(float)
    mx = df["proj"].max() if not df["proj"].empty and df["proj"].max() > 0 else 100.0
    df["base_pown"] = (df["proj"] / mx) * 85.0
    df["c_pown"] = (df["proj"] / mx) * 20.0
    df["vc_pown"] = (df["proj"] / mx) * 15.0
    mean_proj = df["proj"].mean()
    df["eff_credits"] = df["credits"]
    elite_ar_mask = (df["role"] == "AR") & (df["proj"] > mean_proj * 1.25)
    df.loc[elite_ar_mask, "eff_credits"] *= 0.90
    return df

def classify_match_context(match_name):
    name_lower = match_name.lower()
    if any(term in name_lower for term in ["quadrangular", "associate", "nam vs", "vs sa", "qualifier", "t20i"]):
        return "T2_ASSOCIATE", 0.25
    elif any(term in name_lower for term in ["cpl", "ipl", "the hundred", "bbl", "league"]):
        return "FRANCHISE", 0.15
    else:
        return "TIER_1", 0.10

def detect_blowout_and_favorite(df):
    team_points = df.groupby('team')['proj'].sum()
    if len(team_points) < 2:
        return False, None
    sorted_teams = team_points.sort_values(ascending=False)
    if sorted_teams.iloc[0] > sorted_teams.iloc[1] * 1.35:
        return True, sorted_teams.index[0]
    return False, None

def solve_syndicate_matrix(df, match_name, venue_type, num_sims=300):
    match_tier, sim_noise = classify_match_context(match_name)
    is_blowout, favorite_team = detect_blowout_and_favorite(df)
    is_bowling_deck = venue_type in ["spinner", "seamer"]
    
    sims, counts, c_hist = [], {i: 0 for i in df.index}, Counter()
    max_bat = 4 if is_bowling_deck else 6
    min_bowl = 3 if is_bowling_deck else 2

    for _ in range(num_sims):
        prob = lp.LpProblem("Sim", lp.LpMaximize)
        p = {i: lp.LpVariable(f"p_{i}", cat="Binary") for i in df.index}
        c = {i: lp.LpVariable(f"c_{i}", cat="Binary") for i in df.index}
        vc = {i: lp.LpVariable(f"vc_{i}", cat="Binary") for i in df.index}
        
        noisy = {}
        for i, r in df.iterrows():
            penalty = 0.90 if c_hist[i] > num_sims * 0.30 else 1.0
            val = (r["proj"] + np.random.normal(0, sim_noise * r["proj"])) * penalty
            noisy[i] = max(0.0, val)
            
        obj_expr = []
        for i in df.index:
            r_c_weight = 1.25 if (is_bowling_deck and df.loc[i, 'role'] == 'BOWL') else 1.0
            r_vc_weight = 0.65 if (is_bowling_deck and df.loc[i, 'role'] == 'BOWL') else 0.5
            obj_expr.append(noisy[i] * p[i] + r_c_weight * noisy[i] * c[i] + r_vc_weight * noisy[i] * vc[i])
        
        prob += lp.lpSum(obj_expr)
        prob += lp.lpSum(p.values()) == 11
        prob += lp.lpSum([df.loc[i, "eff_credits"] * p[i] for i in df.index]) <= 100.0
        prob += lp.lpSum([df.loc[i, "base_pown"] * p[i] for i in df.index]) <= 398.0
        prob += lp.lpSum(c.values()) == 1
        prob += lp.lpSum(vc.values()) == 1
        
        for i in df.index:
            prob += c[i] + vc[i] <= p[i]
        
        if is_bowling_deck:
            bowl_indices = df[df['role'] == 'BOWL'].index
            prob += lp.lpSum([c[i] + vc[i] for i in bowl_indices]) >= 1
            
        if is_blowout and favorite_team:
            fav_top_indices = df[(df['team'] == favorite_team) & (df['proj'] >= df['proj'].quantile(0.65))].index
            if len(fav_top_indices) > 0:
                prob += lp.lpSum([c[i] for i in fav_top_indices]) == 1
        
        for role, mn, mx in [("WK", 1, 4), ("BAT", 1, max_bat), ("AR", 1, 6), ("BOWL", min_bowl, 6)]:
            role_indices = df[df["role"] == role].index
            prob += lp.lpSum([p[i] for i in role_indices]) >= mn
            prob += lp.lpSum([p[i] for i in role_indices]) <= mx
            
        prob.solve(lp.PULP_CBC_CMD(msg=False))
        if lp.LpStatus[prob.status] == "Optimal":
            sel = [i for i in df.index if p[i].varValue > 0.5]
            c_pick = [i for i in df.index if c[i].varValue > 0.5][0]
            vc_pick = [i for i in df.index if vc[i].varValue > 0.5][0]
            c_hist[c_pick] += 1
            for i in sel:
                counts[i] += 1
            sims.append((tuple(sorted(sel)), c_pick, vc_pick))
            
    df["optimal_pct"] = [(counts[i] / num_sims) * 100 for i in df.index]
    if not sims:
        return None, 0.0, None, None, df
    best, count = Counter(sims).most_common(1)[0]
    return best[0], (count / num_sims) * 100.0, best[1], best[2], df

def export_results(squad, capt, vcapt, name, df):
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
    csv_path = os.path.join(WORKSPACE_DIR, f"LIVE_398_{safe_name.replace(' ', '_')}.csv")
    html_path = os.path.join(WORKSPACE_DIR, f"CHART_{safe_name.replace(' ', '_')}.html")
    
    squad_list = squad['name'].tolist() + [capt, vcapt]
    columns = [f"Player_{i+1}" for i in range(11)] + ["Captain", "Vice_Captain"]
    pd.DataFrame([squad_list], columns=columns).to_csv(csv_path, index=False)
    
    fig = px.scatter(df, x="base_pown", y="proj", color="proj", hover_name="name", title=f"Live Leverage Profile: {name}", template="plotly_dark")
    fig.write_html(html_path)

if __name__ == "__main__":
    fixtures = [
        {"name": "NAM vs SA", "pool": NAM_SA_POOL, "venue": "Windhoek", "pitch": "balance"},
        {"name": "UAE-W vs INA-W", "pool": UAE_INA_POOL, "venue": "Dubai", "pitch": "spinner"}
    ]
    for fix in fixtures:
        df = apply_adjustments(prepare_data(fix["pool"]), fetch_venue_meteo(fix["venue"]), fix["pitch"], None)
        idx, rate, c_idx, vc_idx, lev_df = solve_syndicate_matrix(df, fix["name"], fix["pitch"])
        if idx:
            squad = df.loc[list(idx)].sort_values('role')
            capt, vcapt = df.loc[c_idx, "name"], df.loc[vc_idx, "name"]
            export_results(squad, capt, vcapt, fix["name"], lev_df)
