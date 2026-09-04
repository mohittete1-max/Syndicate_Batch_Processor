import os
import requests
import pandas as pd
import pulp
import plotly.express as px
import sqlite3

# HARDCODED API KEYS & TELEGRAM CONFIG
CRICDATA_API_KEY = "4905f024-424c-4f6c-a2e6-b4e64f41f7bb"
TELEGRAM_BOT_TOKEN = "8942957322:AAF86-GixapC8Rs88Jcn-wWX6M-o-6SYWKE"
TELEGRAM_CHAT_ID = "8942186617"

def send_telegram_message(message):
    """Dispatches a notification message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("[+] Telegram alert sent successfully.")
        else:
            print(f"[!] Failed to send Telegram alert. Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"[!] Exception occurred while sending Telegram message: {e}")

def apply_vegas_multipliers_from_api(df, odds_mapping):
    """Applies precise projection boosts/penalties based on real bookmaker implied probabilities."""
    print("Applying live Vegas odds projection multipliers...")
    for idx, row in df.iterrows():
        team = row['Team']
        implied_prob = odds_mapping.get(team, 0.50)
        multiplier = 1.0 + (implied_prob - 0.50) * 1.0
        df.loc[idx, 'Projection'] = round(row['Projection'] * multiplier, 1)
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
    
    total_proj = round(pulp.value(prob.objective), 2)
    print(f"Optimization Status: {pulp.LpStatus[prob.status]}")
    print(f"Total Projected Points: {total_proj}\n")
    return df, optimal_lineup, total_proj

def create_visualizations(pool_df, filename="Enhanced_Leverage_Chart.html"):
    """Generates a dynamic 3D bubble chart for leverage and salary analysis."""
    print(f"Generating enhanced HTML visualization: {filename}...")
    fig = px.scatter(
        pool_df, 
        x="pOWN%", 
        y="Projection", 
        size="Salary", 
        color="Leverage_Score",
        hover_name="Player_Name",
        hover_data=["Team", "Role", "Salary"],
        title=f"DFS Leverage & Salary Matrix - {filename}",
        color_continuous_scale="Viridis",
        template="plotly_dark"
    )
    fig.add_shape(type="line", x0=0, y0=20, x1=100, y1=80, line=dict(color="Red", dash="dash"))
    fig.write_html(filename)

if __name__ == "__main__":
    matches_data = {
        "NAM_vs_SA": {
            "odds": {"South Africa": 0.75, "Namibia": 0.25},
            "data": {
                "Player_Name": [
                    "R Hermann", "C Esterhuizen", "Z Green",
                    "D Brevis", "L Pretorius", "J Hermann", "A Volschenk", "de Zorzi", "M Kruger", "J Taanyanda", "L Steenkamp", "J Smith", "J Frylinck", "van Lingen", "D Leicher",
                    "G Erasmus", "J Balt", "Smit", "E Bosch", "P Subrayen", "Nicol Loftie-Eaton", "A Simelane",
                    "K Maphaka", "B Fortuin", "L Sipamla", "N Peter", "W Smith", "D Jansen", "J Brassell", "N Mokoena", "M Heingo", "B Shikongo", "R Trumpelmann", "B Scholtz"
                ],
                "Team": [
                    "South Africa", "South Africa", "Namibia",
                    "South Africa", "South Africa", "Namibia", "Namibia", "South Africa", "Namibia", "Namibia", "Namibia", "South Africa", "Namibia", "Namibia", "Namibia",
                    "Namibia", "Namibia", "Namibia", "South Africa", "South Africa", "Namibia", "South Africa",
                    "South Africa", "South Africa", "South Africa", "South Africa", "Namibia", "South Africa", "Namibia", "Namibia", "Namibia", "Namibia", "Namibia", "Namibia"
                ],
                "Role": [
                    "WK", "WK", "WK",
                    "BAT", "BAT", "BAT", "BAT", "BAT", "BAT", "BAT", "BAT", "BAT", "BAT", "BAT", "BAT",
                    "AR", "AR", "AR", "AR", "AR", "AR", "AR",
                    "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL"
                ],
                "Salary": [
                    8.5, 7.0, 6.0,
                    9.0, 8.5, 8.0, 8.0, 8.0, 8.0, 8.0, 7.5, 7.5, 7.0, 6.5, 6.0,
                    8.0, 8.0, 7.0, 7.0, 6.5, 6.5, 6.5,
                    8.0, 8.0, 8.0, 7.5, 7.5, 7.0, 7.0, 7.0, 6.5, 6.5, 6.0, 6.0
                ],
                "Projection": [
                    52.0, 38.0, 130.0,
                    417.0, 265.0, 129.0, 198.0, 48.0, 0.0, 0.0, 115.0, 0.0, 194.0, 0.0, 0.0,
                    372.0, 35.0, 193.0, 93.0, 247.0, 97.0, 0.0,
                    71.0, 314.0, 7.0, 31.0, 0.0, 377.0, 111.0, 55.0, 104.0, 0.0, 253.0, 44.0
                ],
                "pOWN%": [
                    23.73, 21.95, 69.25,
                    91.93, 87.07, 63.29, 26.40, 1.76, 2.55, 1.99, 23.36, 2.48, 69.31, 2.74, 2.61,
                    92.17, 7.12, 77.21, 31.50, 81.83, 11.73, 2.25,
                    18.02, 85.53, 2.65, 9.17, 2.91, 84.18, 16.12, 2.12, 9.25, 2.80, 70.16, 2.83
                ]
            }
        },
        "UAE_W_vs_INA_W": {
            "odds": {"UAE-W": 0.55, "INA-W": 0.45},
            "data": {
                "Player_Name": [
                    "T Satish", "Putu Ayu Nanda Sakarini", "Winda Prastini",
                    "M Kulkarni", "R Pangestuti", "K Kasse", "D Wulandari", "M Corazon", "R Rajith", "L Keny", "U Iyer",
                    "Luh Dewi", "H Hotchandani", "Elna Yaung", "J Thirukkumaran", "Made Putri Suwandewi", "S Dharnidharka", "Kadek Fitria Rada Rani", "E Oza",
                    "S Velic", "N Ariani", "I Nandakumar", "A Silva", "D Bangi", "S Maypriani", "V Mahesh", "S Gokhale", "S Kotte", "L Qiao", "D Paramitha", "A Supriya"
                ],
                "Team": [
                    "UAE-W", "INA-W", "INA-W",
                    "UAE-W", "INA-W", "INA-W", "INA-W", "INA-W", "UAE-W", "UAE-W", "UAE-W",
                    "INA-W", "UAE-W", "INA-W", "UAE-W", "INA-W", "UAE-W", "INA-W", "UAE-W",
                    "INA-W", "INA-W", "UAE-W", "UAE-W", "INA-W", "INA-W", "UAE-W", "UAE-W", "UAE-W", "INA-W", "INA-W", "UAE-W"
                ],
                "Role": [
                    "WK", "WK", "WK",
                    "BAT", "BAT", "BAT", "BAT", "BAT", "BAT", "BAT", "BAT",
                    "AR", "AR", "AR", "AR", "AR", "AR", "AR", "AR",
                    "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL"
                ],
                "Salary": [
                    7.0, 6.0, 6.0,
                    8.0, 8.0, 8.0, 7.5, 6.5, 6.0, 6.0, 6.0,
                    8.0, 7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0,
                    8.0, 8.0, 8.0, 8.0, 8.0, 7.5, 7.0, 7.0, 6.5, 6.5, 6.0, 6.0
                ],
                "Projection": [
                    2.0, 17.0, 0.0,
                    7.0, 47.0, 0.0, 22.0, 32.0, 27.0, 1.0, 0.0,
                    196.0, 10.0, 0.0, 0.0, 87.0, 28.0, 20.0, 50.0,
                    87.0, 194.0, 12.0, 12.0, 0.0, 112.0, 8.0, 0.0, 5.0, 0.0, 24.0, 0.0
                ],
                "pOWN%": [
                    80.13, 19.40, 9.63,
                    5.95, 70.20, 12.03, 6.08, 21.77, 26.95, 9.05, 11.56,
                    86.43, 56.21, 10.99, 10.97, 75.61, 70.06, 18.41, 81.49,
                    73.55, 82.97, 44.19, 40.52, 10.70, 71.41, 25.86, 10.58, 17.39, 10.73, 18.56, 10.63
                ]
            }
        }
    }

    conn = sqlite3.connect("dfs_history.db")

    for match_id, match_info in matches_data.items():
        print(f"\n================================")
        print(f"Processing Match: {match_id}")
        print(f"================================")
        
        player_pool = pd.DataFrame(match_info["data"])
        
        # Apply Vegas Odds Multipliers
        player_pool = apply_vegas_multipliers_from_api(player_pool, match_info["odds"])
        
        # Run ILP Optimization
        processed_pool, lineup, total_proj = run_ilp_optimization(player_pool)
        
        # Save to SQLite Database
        lineup.insert(0, "Match_ID", match_id)
        lineup.to_sql("historical_lineups", conn, if_exists="append", index=False)
        
        # Generate Visualizations
        create_visualizations(processed_pool, f"{match_id}_Leverage_Chart.html")
        
        # Automated Captain & Vice-Captain assignment based on highest projection
        lineup_sorted = lineup.sort_values(by="Projection", ascending=False).reset_index(drop=True)
        captain = lineup_sorted.loc[0, "Player_Name"] if len(lineup_sorted) > 0 else ""
        vc = lineup_sorted.loc[1, "Player_Name"] if len(lineup_sorted) > 1 else ""

        player_lines = []
        for _, row in lineup_sorted.iterrows():
            name = row["Player_Name"]
            tag = ""
            if name == captain:
                tag = " (C 👑)"
            elif name == vc:
                tag = " (VC 🥈)"
            player_lines.append(f"• {name}{tag} [{row['Role']}]")

        players_formatted = "\n".join(player_lines)

        # Format and Dispatch Telegram Alert with C/VC tags
        tg_message = (
            f"🏏 *DFS Match Optimized*: `{match_id}`\n"
            f"📊 *Total Projected Points*: `{total_proj}`\n"
            f"👑 *Captain*: `{captain}`\n"
            f"🥈 *Vice-Captain*: `{vc}`\n\n"
            f"👥 *Selected 11 Players*:\n{players_formatted}"
        )
        send_telegram_message(tg_message)

    conn.close()
    print("\nAll matches processed successfully! Batched lineups logged, charts generated, and Telegram alerts dispatched with C/VC.")
