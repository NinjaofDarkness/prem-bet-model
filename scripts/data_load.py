import pandas as pd
import ast
import requests
from io import StringIO
from typing import List, Dict

def fetch_match_data(client, start_year: int, end_year: int, league: str) -> pd.DataFrame:
    """Fetches all match data for a league across seasons."""
    all_matches: List[Dict] = []

    for year in range(start_year, end_year + 1):
        try:
            matches = client.league(league).get_match_data(season=str(year))
            for m in matches:
                m["season"] = year
            all_matches.extend(matches)
        except Exception as e:
            print(f"[!] Error fetching match data for {year}: {e}")
            continue

    df = pd.json_normalize(all_matches, sep="_")
    return df

def fetch_team_data(df_matches, client, start_year: int, end_year: int, league: str) -> pd.DataFrame:
    """Fetches team-level summary data across seasons."""
    all_team_data: List[Dict] = []

    for year in range(start_year, end_year + 1):
        try:
            team_data = client.league(league).get_team_data(season=str(year))
            for team_name, stats in team_data.items():
                stats['season'] = year
                stats['team_name'] = team_name
                all_team_data.append(stats)
        except Exception as e:
            print(f"[!] Error fetching team stats for {year}: {e}")
            continue

    df = pd.json_normalize(all_team_data, sep="_")

    # Match team IDs to home match titles to ensure correct team names.
    id_title_map = (
        df_matches[['h_id', 'h_title']]
        .dropna()
        .drop_duplicates(subset='h_id')
        .set_index('h_id')['h_title']
        .to_dict()
    )

    df['title'] = df['id'].map(id_title_map)

    # Remove JSON Columns
    summary_rows = []

    for _, row in df.iterrows():
        team_id = row.get('id')
        title = row.get('title')
        season = row.get('season')
        history = row.get('history')

        if isinstance(history, str):
            history = ast.literal_eval(history)

        total_goals = sum(match.get('scored', 0) for match in history)
        total_conceded = sum(match.get('missed', 0) for match in history)
        total_xG = sum(match.get('xG', 0) for match in history)
        total_xGA = sum(match.get('xGA', 0) for match in history)
        total_pts = sum(match.get('pts', 0) for match in history)
        total_xpts = sum(match.get('xpts', 0) for match in history)

        summary_rows.append({
            'id': team_id,
            'title': title,
            'season': season,
            'goals': total_goals,
            'conceded': total_conceded,
            'xG': round(total_xG, 2),
            'xGA': round(total_xGA, 2),
            'pts': total_pts,
            'xpts': round(total_xpts, 2)
        })

    df_summary = pd.DataFrame(summary_rows)

    # Drop duplicates based on 'id', 'xG', and 'xGA'
    df_summary = df_summary.drop_duplicates(subset=['id', 'xG', 'xGA'])
    return df_summary

def fetch_elo_data(df_summary, start_year) -> pd.DataFrame:
    """Fetches ELO data from ClubElo API."""
    team_list = df_summary['title'].dropna().unique().tolist()
    print(team_list)

    final_team_name_map = {
    'Manchester City': 'ManCity',
    'Manchester United': 'ManUnited',
    'Wolverhampton Wanderers': 'Wolves',
    'West Bromwich Albion': 'WestBrom',
    'Newcastle United': 'Newcastle',
    'Sheffield United': 'SheffieldUnited',
    'Nottingham Forest': 'Forest',
    'Burnley': 'Burnley',
    'Brighton & Hove Albion': 'Brighton',
    }

    clubelo_data = {}
    for team_name in team_list:
        print(f"Fetching ELO data for {team_name}...")
        api_name = final_team_name_map.get(team_name, team_name.replace(" ", ""))
        print(f"Using API name: {api_name}")

        url = f"http://api.clubelo.com/{api_name}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            clubelo_data[team_name] = df
        except Exception as e:
            print(f"[!] Error fetching ELO data for {team_name}: {e}")
            continue
    
    if clubelo_data:
        all_data_df = pd.concat(
            clubelo_data.values(),
            keys=clubelo_data.keys(),
            names=['team', 'row']
        ).reset_index(level=0)
        all_data_df = all_data_df.rename(columns={'team': 'title'})
        # Ensure 'To' column is date 
        all_data_df['To'] = pd.to_datetime(all_data_df['To'], errors='coerce')
        all_data_df = all_data_df[all_data_df['To'].dt.year >= start_year]
    return all_data_df

def run_data_load(client, start_year, end_year, league, raw_match_data_path, raw_elo_data_path):
    df_matches = fetch_match_data(client,start_year, end_year, league)
    df_matches = df_matches[df_matches['isResult'] == True].reset_index(drop=True)
    df_matches.to_csv(raw_match_data_path, index=False)
    print("Saved match data")

    df_summary = fetch_team_data(df_matches, client, start_year, end_year, league)
    print("Team summary ready")

    df_elo = fetch_elo_data(df_summary, start_year)
    df_elo.to_csv(raw_elo_data_path, index=False)
    print("Saved ELO data")

    return df_matches, df_elo
