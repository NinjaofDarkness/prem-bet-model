import pandas as pd

def engineer_stat_diff(df: pd.DataFrame) -> pd.DataFrame:
    df['elo_diff'] = round(df['h_elo'] - df['a_elo'], 3)
    df['value_diff'] = round(df['total_market_value_h'] - df['total_market_value_a'], 3)
    df['age_diff'] = round(df['avg_age_h'] - df['avg_age_a'], 3)
    return df

def generate_h2h_features(matches_df, n=5):
    matches_df = matches_df.sort_values(by='datetime')
    h2h_data = []

    for idx, row in matches_df.iterrows():
        h_id, a_id, match_date = row['h_id'], row['a_id'], row['datetime']

        # Find past H2H matches before current match
        past_matches = matches_df[
            (((matches_df['h_id'] == h_id) & (matches_df['a_id'] == a_id)) |
             ((matches_df['h_id'] == a_id) & (matches_df['a_id'] == h_id))) &
            (matches_df['datetime'] < match_date)
        ].sort_values(by='datetime', ascending=False).head(n)

        # Initialize features
        h_wins, a_wins, draws, goal_diff_sum = 0, 0, 0, 0

        for _, past in past_matches.iterrows():
            home = past['h_id']
            away = past['a_id']
            gh = past['goals_h']
            ga = past['goals_a']

            if gh > ga:
                winner = 'H'
            elif gh < ga:
                winner = 'A'
            else:
                winner = 'D'

            if winner == 'H' and home == h_id:
                h_wins += 1
            elif winner == 'A' and away == h_id:
                h_wins += 1
            elif winner == 'H' and home == a_id:
                a_wins += 1
            elif winner == 'A' and away == a_id:
                a_wins += 1
            else:
                draws += 1

            goal_diff = (gh - ga) if home == h_id else (ga - gh)
            goal_diff_sum += goal_diff

        h2h_data.append({
            'h2h_home_wins': h_wins,
            'h2h_away_wins': a_wins,
            'h2h_draws': draws,
            'h2h_goal_diff_avg': round(goal_diff_sum / max(1, len(past_matches)), 3),
            'h2h_matches_played': len(past_matches)
        })

    h2h_df = pd.DataFrame(h2h_data)
    return pd.concat([matches_df.reset_index(drop=True), h2h_df], axis=1)

def compute_recent_form(df, team_col, side='h', n=5):
    df = df.sort_values(by='datetime')
    form_stats = []

    for idx, row in df.iterrows():
        team_id = row[team_col]
        match_date = row['datetime']

        # Filter past matches involving this team
        past_matches = df[
            (((df['h_id'] == team_id) | (df['a_id'] == team_id)) &
             (df['datetime'] < match_date))
        ].sort_values(by='datetime', ascending=False).head(n)

        points = goals_scored = goals_conceded = xg_for = xg_against = 0

        for _, match in past_matches.iterrows():
            if match['h_id'] == team_id:
                goals_scored += match['goals_h']
                goals_conceded += match['goals_a']
                xg_for += match['xG_h']
                xg_against += match['xG_a']
                result = match['goals_h'] - match['goals_a']
            else:
                goals_scored += match['goals_a']
                goals_conceded += match['goals_h']
                xg_for += match['xG_a']
                xg_against += match['xG_h']
                result = match['goals_a'] - match['goals_h']

            if result > 0:
                points += 3
            elif result == 0:
                points += 1

        form_stats.append({
            f'{side}_form_points': points,
            f'{side}_form_goals_scored': goals_scored,
            f'{side}_form_goals_conceded': goals_conceded,
            f'{side}_form_xg': round(xg_for / max(1, len(past_matches)), 3),
            f'{side}_form_xga': round(xg_against / max(1, len(past_matches)), 3)
        })

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(form_stats)], axis=1)

def clean_trainset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['h_id', 'a_id', 'goals_h', 'goals_a', 'xG_h', 
                          'xG_a', 'h_elo', 'a_elo', 'total_market_value_h', 
                          'total_market_value_a', 'avg_age_h', 'avg_age_a'])
    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.fillna(0)    
    return df

def engineer_features(trainset, final_trainset_path):
    trainset = engineer_stat_diff(trainset)
    trainset = generate_h2h_features(trainset)
    trainset = compute_recent_form(trainset, 'h_id', 'h')
    trainset = compute_recent_form(trainset, 'a_id', 'a')
    trainset = clean_trainset(trainset)
    trainset.to_csv(final_trainset_path, index=False)
    print(f"Final trainset saved")
    return trainset
