import pandas as pd
import numpy as np

def fractional_to_decimal(fraction_str):
    try:
        numerator, denominator = map(float, str(fraction_str).split('/'))
        return round((numerator / denominator) + 1, 3)
    except:
        return np.nan

def clean_fixtures(fixtures_df, gw, season):
    fixtures_df = fixtures_df[(fixtures_df['gw'] == gw) & (fixtures_df['season'] == season)]

    fixtures_df['datetime'] = pd.to_datetime(fixtures_df['datetime']).dt.strftime('%Y-%m-%d')

    fixtures_df['book_odds_h'] = fixtures_df['book_odds_h'].apply(fractional_to_decimal)
    fixtures_df['book_odds_d'] = fixtures_df['book_odds_d'].apply(fractional_to_decimal)
    fixtures_df['book_odds_a'] = fixtures_df['book_odds_a'].apply(fractional_to_decimal)
    
    fixtures_df = fixtures_df.dropna(subset=['book_odds_h', 'book_odds_d', 'book_odds_a'])
    return fixtures_df

def encode_outcome(row):
    if row['goals_h'] > row['goals_a']:
        return 'H'
    elif row['goals_h'] < row['goals_a']:
        return 'A'
    else:
        return 'D'

def add_gw(trainset):
    trainset = trainset.sort_values(by='datetime').reset_index(drop=True)
    # Create the 'gw' column: 1 to 38, each repeated 10 times
    trainset['gw'] = np.tile(np.repeat(np.arange(1, 39), 10), 10)
    trainset.insert(2, 'gw', trainset.pop('gw'))
    return trainset

def clean_and_convert_to_odds(matches_df):
    matches_df['datetime'] = pd.to_datetime(matches_df['datetime']).dt.strftime('%Y-%m-%d')
    matches_df['h_id'] = matches_df['h_id'].astype(int)
    matches_df['a_id'] = matches_df['a_id'].astype(int)
    matches_df['goals_h'] = matches_df['goals_h'].astype(int)
    matches_df['goals_a'] = matches_df['goals_a'].astype(int)
    matches_df['xG_h'] = matches_df['xG_h'].astype(float)
    matches_df['xG_a'] = matches_df['xG_a'].astype(float)
    
    # Ensure forecast columns are numeric
    matches_df['forecast_w'] = pd.to_numeric(matches_df['forecast_w'], errors='coerce')
    matches_df['forecast_d'] = pd.to_numeric(matches_df['forecast_d'], errors='coerce')
    matches_df['forecast_l'] = pd.to_numeric(matches_df['forecast_l'], errors='coerce')
    
    matches_df['book_odds_h'] = (1 / matches_df['forecast_w']).round(3)
    matches_df['book_odds_d'] = (1 / matches_df['forecast_d']).round(3)
    matches_df['book_odds_a'] = (1 / matches_df['forecast_l']).round(3)

    matches_df = matches_df.drop(
        columns=['id', 'isResult', 'h_short_title', 'a_short_title', 'forecast_w', 'forecast_d', 'forecast_l']
    )

    return matches_df

def merge_squad_values(matches_df, squad_data):
    trainset = matches_df.merge(squad_data, left_on=['h_id', 'season'], right_on=['id', 'season'], how='left')
    trainset = trainset.rename(columns={col: col + '_h' for col in squad_data.columns if col not in ['id', 'season']})

    trainset = trainset.merge(squad_data, left_on=['a_id', 'season'], right_on=['id', 'season'], how='left')
    trainset = trainset.rename(columns={col: col + '_a' for col in squad_data.columns if col not in ['id', 'season']})
    trainset = trainset.drop(columns=['id_x', 'id_y', 'title_h', 'title_a'])
    return trainset

def merge_elo_ratings(trainset, elo_data, date_col='datetime'):
    """
    Merge Elo ratings into fixture data based on date ranges and team names.

    Adds 'h_elo' and 'a_elo' columns to trainset using elo_data, matching:
    - h_title/a_title to elo_data['title']
    - datetime to elo_data[From:To] date interval
    """
    # Ensure all relevant columns are datetime
    trainset = trainset.copy()
    trainset[date_col] = pd.to_datetime(trainset[date_col])
    elo_data = elo_data.copy()
    elo_data['From'] = pd.to_datetime(elo_data['From'])
    elo_data['To'] = pd.to_datetime(elo_data['To']) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # make 'To' inclusive

    def merge_side(fixtures_df, elo_df, team_col, prefix):
        # Merge fixtures with Elo data on team name
        merged = fixtures_df.merge(
            elo_df,
            left_on=team_col,
            right_on='title',
            how='left'
        )

        # Filter for rows where fixture datetime falls within Elo date range
        in_range = (merged[date_col] >= merged['From']) & (merged[date_col] <= merged['To'])
        merged = merged[in_range]

        # Keep only necessary columns
        merged = merged[[*fixtures_df.columns, 'Elo']]
        merged = merged.rename(columns={'Elo': f'{prefix}_elo'})

        return merged.drop_duplicates(subset=fixtures_df.columns)

    print("Merging home team Elo...")
    fixtures_with_home = merge_side(trainset, elo_data, 'h_title', 'h')

    print("Merging away team Elo...")
    fixtures_with_both = merge_side(fixtures_with_home, elo_data, 'a_title', 'a')

    # Final check on missing values
    if 'h_elo' in fixtures_with_both.columns:
        h_na = fixtures_with_both['h_elo'].isna().sum()
        print(f"Missing h_elo: {h_na} / {len(fixtures_with_both)}")

    if 'a_elo' in fixtures_with_both.columns:
        a_na = fixtures_with_both['a_elo'].isna().sum()
        print(f"Missing a_elo: {a_na} / {len(fixtures_with_both)}")

    return fixtures_with_both

def prep_trainset(matches_df, elo_df, raw_squad_data_path, raw_fixtures_path, merged_trainset_path, gw_to_predict, season_to_predict):
    squad_data = pd.read_csv(raw_squad_data_path)
    fixt_list = pd.read_csv(raw_fixtures_path)

    fixt_list = clean_fixtures(fixt_list, gw_to_predict, season_to_predict)
    matches_df['outcome'] = matches_df.apply(encode_outcome, axis=1)
    matches_df = add_gw(matches_df)
    matches_df = clean_and_convert_to_odds(matches_df)

    matches_df = pd.concat([matches_df, fixt_list], ignore_index=True)
    print("New Fixtures added")

    trainset = merge_squad_values(matches_df, squad_data)
    trainset = merge_elo_ratings(trainset, elo_df)
    print("Merged squad and Elo ratings")

    trainset.to_csv(merged_trainset_path, index=False)
    print(f"Trainset saved")
    return trainset
