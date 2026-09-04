import os
import requests
import pandas as pd
import pulp
import plotly.express as px
import random
import re

# HARDCODED API KEY
CRICDATA_API_KEY = "4905f024-424c-4f6c-a2e6-b4e64f41f7bb"

def get_active_match_id():
    """Automatically fetches the first available valid match ID from CricAPI."""
    print("[*] Fetching today's active match list from CricAPI...")
    matches_url = f"https://api.cricapi.com/v1/currentMatches?apikey={CRICDATA_API_KEY}&offset=0"
    
    response = requests.get(matches_url)
    if response.status_code != 200:
        raise SystemExit(f"Failed to fetch active matches. Status: {response.status_code}")
        
    data = response.json()
    if data.get("status") != "success":
        raise SystemExit(f"API Error: {data.get('reason')}")
        
    match_list = data.get("data", [])
    if not match_list:
        raise SystemExit("No active matches found today.")
        
    # Pick the first match in the list
    selected_match = match_list[0]
    match_id = selected_match.get("id")
    match_name = selected_match.get("name")
    
    print(f"[+] Auto-selected Match: {match_name}")
    print(f"[+] Using Valid GUID: {match_id}\n")
    return match_id

def fetch_live_pool(match_id):
    """Fetches live player data from CricAPI, with a simulated fallback for empty rosters."""
    
    # 1. Intercept invalid IDs (CricAPI requires UUID format)
    if not re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", match_id):
        print(f"[!] The ID '{match_id}' is not a valid 32-digit CricAPI GUID.")
        match_id = get_active_match_id()
        
    url = f"https://api.cricapi.com/v1/match_squad?apikey={CRICDATA_API_KEY}&id={match_id}"
    
    print(f"Fetching live squad roster...")
    response = requests.get(url)
    
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        raise SystemExit(f"API Error! Status: {response.status_code}. Raw: {response.text[:150]}")
        
    if data.get("status") != "success":
        raise ConnectionError(f"API Fetch Failed: {data.get('reason', 'Unknown API Error')}")
    
    all_players = []
    for team_info in data.get("data", []):
        team_name = team_info.get("teamName", "Unknown")
        for player in team_info.get("players", []):
            # Generating DFS metrics for the optimizer since API only returns raw roster profiles
            player_dict = {
                "Player_Name": player.get("name"),
                "Team": team_name,
                "Role": player.get("role", "BAT"),
                "Salary": round(random.uniform(7.5, 10.5), 1), 
                "Projection": round(random.uniform(20, 80), 1), 
                "pOWN%": round(random.uniform(5, 60), 1)       
            }
            all_players.append(player_dict)
            
    df = pd.DataFrame(all_players)
    
    # 2. Failsafe for unannounced/empty squads
    if df.empty:
        print("\n[!] API returned successfully, but no squad data is available for this match yet.")
        print("[*] Generating a simulated 22-player roster to test the ILP pipeline...")
        
        dummy_players = []
        for team_name in ["Team_A", "Team_B"]:
            for i in range(1, 12):
                dummy_players.append({
                    "Player_Name": f"{team_name}_Player_{i}",
                    "Team": team_name,
                    "Role": random.choice(["BAT", "BOWL", "AR", "WK"]),
                    "Salary": round(random.uniform(7.5, 10.5), 1),
                    "Projection": round(random.uniform(20, 80), 1),
                    "pOWN%": round(random.uniform(5, 60), 1)
                })
        df = pd.DataFrame(dummy_players)
    
    print(f"Successfully loaded {len(df)} players into the pipeline.\n")
    return df

def run_ilp_optimization(df):
    """Runs Integer Linear Programming to find the optimal lineup."""
    print("Running ILP Optimization constraints...")
    
    df["Leverage_Score"] = df["Projection"] - (df["pOWN%"] * 0.5)
    
    prob = pulp.LpProblem("Fantasy_Cricket_Optimizer", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Players", df.index, cat="Binary")

    prob += pulp.lpSum([df.loc[i, "Projection"] * player_vars[i] for i in df.index]), "Total_Projection"
    prob += pulp.lpSum([player_vars[i] for i in df.index]) == 11, "Exactly_11_Players"
    prob += pulp.lpSum([df.loc[i, "Salary"] * player_vars[i] for i in df.index]) <= 100.0, "Salary_Cap"

    for team in df['Team'].unique():
        team_players = df[df['Team'] == team].index
        prob += pulp.lpSum([player_vars[i] for i in team_players]) <= 7, f"Max_7_{team}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    selected_indices = [i for i in df.index if player_vars[i].varValue == 1.0]
    optimal_lineup = df.loc[selected_indices].copy()
    
    print(f"Optimization Status: {pulp.LpStatus[prob.status]}")
    print(f"Total Projected Points: {round(pulp.value(prob.objective), 2)}\n")
    return df, optimal_lineup

def create_visualizations(pool_df):
    """Generates a dynamic 3D bubble chart for leverage and salary analysis."""
    print("Generating enhanced HTML visualization...")
    fig = px.scatter(
        pool_df, 
        x="pOWN%", 
        y="Projection", 
        size="Salary", 
        color="Leverage_Score",
        hover_name="Player_Name",
        hover_data=["Team", "Role", "Salary"],
        title="DFS Leverage & Salary Matrix",
        color_continuous_scale="Viridis",
        template="plotly_dark"
    )
    fig.add_shape(type="line", x0=0, y0=20, x1=100, y1=80, line=dict(color="Red", dash="dash"))
    fig.write_html("Enhanced_Leverage_Chart.html")

if __name__ == "__main__":
    # The script will detect this is invalid and auto-fetch a real match.
    MATCH_ID = "398"
    
    player_pool = fetch_live_pool(MATCH_ID)
    processed_pool, lineup = run_ilp_optimization(player_pool)
    lineup.to_csv("Optimized_Lineup.csv", index=False)
    create_visualizations(processed_pool)
    
    print("Pipeline complete! Check your folder for:")
    print(" 1. Optimized_Lineup.csv")
    print(" 2. Enhanced_Leverage_Chart.html")
