import os
import pandas as pd

def simulate_bets(results_df):
    # Filter rows where the model made a bet
    bets = results_df[results_df['prediction'].notnull()].copy()

    # Map the odds for the decision placed
    bets['odds'] = bets.apply(
        lambda row: row['book_odds_h'] if row['prediction'] == 'H'
        else row['book_odds_d'] if row['prediction'] == 'D'
        else row['book_odds_a'],
        axis=1
    )

    # Determine win/loss
    bets['won'] = bets['prediction'] == bets['outcome']

    # Simulate profit/loss for each bet
    bets['profit'] = bets['won'].apply(lambda x: 1) * bets['odds'] - 1
    bets.loc[~bets['won'], 'profit'] = -1  # Lose the £1 stake

    # Calculate stats
    total_profit = bets['profit'].sum()
    roi = (total_profit / len(bets)) * 100  # % return per bet

    print(f"Total Bets Placed: {len(bets)}")
    print(f"Total Profit: £{total_profit:.2f}")
    print(f"ROI: {roi:.2f}%")

    # Save the detailed bets as a CSV
    output_dir = "data/output/train_eval"
    os.makedirs(output_dir, exist_ok=True)
    
    bets.to_csv(os.path.join(output_dir, "last_gw_sim_details.csv"), index=False)

    # Save summary stats
    summary_df = pd.DataFrame([{
        "total_bets": len(bets),
        "total_profit": round(total_profit, 2),
        "roi_percent": round(roi, 2)
    }])

    summary_df.to_csv(os.path.join(output_dir, "last_gw_sum_summary.csv"), index=False)


    return bets