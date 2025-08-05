import pandas as pd

def predict_gw(gw_to_predict, season_to_predict, threshold, model, le, feature_cols, final_trainset_path, output_predictions_path):
    # Load upcoming fixtures
    df = pd.read_csv(final_trainset_path)
    df = df[((df['season'] == season_to_predict) & (df['gw'] == gw_to_predict))]

    # Preprocessing datetime if needed
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Dummy target column (required for LabelEncoder structure)
    if 'outcome' not in df.columns:
        df['outcome'] = 'Unknown'  # just placeholder

    # Predict
    X = df[feature_cols]
    y_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    df['predicted_outcome'] = le.inverse_transform(y_pred)

    # Add prediction probabilities
    label_order = le.classes_  # ['A', 'D', 'H'] most likely
    df['pred_A'] = y_proba[:, list(label_order).index('A')].round(3)
    df['pred_D'] = y_proba[:, list(label_order).index('D')].round(3)
    df['pred_H'] = y_proba[:, list(label_order).index('H')].round(3)

    # Calculate implied probabilities from bookmaker odds
    for col in ['book_odds_h', 'book_odds_d', 'book_odds_a']:
        df[f'implied_{col[-1]}'] = 1 / df[col]
    total_implied = df[['implied_h', 'implied_d', 'implied_a']].sum(axis=1)
    df[['implied_h', 'implied_d', 'implied_a']] = df[['implied_h', 'implied_d', 'implied_a']].div(total_implied, axis=0)

    # Calculate expected value
    for label, prob_col, odds_col in zip(['H', 'D', 'A'], ['pred_H', 'pred_D', 'pred_A'], ['book_odds_h', 'book_odds_d', 'book_odds_a']):
        df[f'ev_{label.lower()}'] = round(df[prob_col] * df[odds_col] - 1, 2)

    def decide_bet(row, threshold=threshold):
        evs = {
            'H': row['ev_h'],
            'D': row['ev_d'],
            'A': row['ev_a']
        }
        positive_evs = {k: v for k, v in evs.items() if v > threshold}
        if not positive_evs:
            return 'No Bet'
        return max(positive_evs, key=positive_evs.get)
    
    # Make betting decisions
    df['bet_decision'] = df.apply(decide_bet, axis=1)

    # Output
    output_cols = [
        'datetime', 'season', 'gw', 'h_title', 'a_title',
        'book_odds_h', 'book_odds_d', 'book_odds_a',
        'pred_H', 'pred_D', 'pred_A',
        'bet_decision', 'predicted_outcome'
    ]
    df[output_cols].to_csv(output_predictions_path, index=False)
    print("Predictions saved")
    return df[output_cols]
