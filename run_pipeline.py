import yaml
from understatapi import UnderstatClient
from scripts.data_load import run_data_load
from scripts.data_align import prep_trainset
from scripts.feature_engineering import engineer_features
from scripts.baseline_model import train_model
from scripts.simulate_returns import simulate_bets
from scripts.predict import predict_gw

### MANUAL CONFIGURATION BEFORE RUNNING SCRIPT ###
# - Check if last weeks data is available via API
# - Check if Squad data is up to date via TransferMarkt
# - Assign upcoming weeks fixture odds in 2025_fixture_list.csv
# - Set the season and gameweek to predict in config.yaml
# - Ensure all paths in config.yaml are correct

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    start_year = config['start_year']
    end_year = config['end_year']
    league = config['league']

    raw_match_data_path = config['raw_match_data_path']
    raw_elo_data_path = config['raw_elo_data_path']
    raw_fixtures_path = config['raw_fixtures_path']
    raw_squad_data_path = config['raw_squad_data_path']
    merged_trainset_path = config['merged_trainset_path']
    final_trainset_path = config['final_trainset_path']
    features_path = config['features_path']
    label_encoder_path = config['label_encoder_path']
    model_path = config['model_path']

    gw_to_predict = config['gw_to_predict']
    season_to_predict = config['season_to_predict']
    threshold = config['threshold_ev']

    output_path_template = config["output_predictions_path"]
    output_predictions_path = output_path_template.format(
        season_to_predict=season_to_predict,
        gw_to_predict=gw_to_predict
    )
    print(f"Output predictions path: {output_predictions_path}")

    client = UnderstatClient()

    matches_df, elo_df = run_data_load(
        client, start_year, end_year, league, raw_match_data_path, raw_elo_data_path
    )
    print(matches_df.tail())

    merged_trainset = prep_trainset(
        matches_df, elo_df, raw_squad_data_path, raw_fixtures_path, merged_trainset_path, gw_to_predict, season_to_predict
    )

    final_trainset = engineer_features(
        merged_trainset, final_trainset_path
    )

    model, le, feature_cols, results_df = train_model(
        final_trainset, gw_to_predict, season_to_predict, features_path, label_encoder_path, model_path
    )

    simulate_bets(results_df)

    predictions = predict_gw(
        gw_to_predict, season_to_predict, threshold, model, le, feature_cols, final_trainset_path, output_predictions_path
    )
    print(predictions)

if __name__ == "__main__":
    main()
