import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import json
import joblib
from sklearn.metrics import log_loss, accuracy_score, classification_report
import matplotlib.pyplot as plt

def train_model(final_trainset, gw_to_predict, season_to_predict, features_path, label_encoder_path, model_path):
    # Split from gw_to_predict
    final_trainset = final_trainset[~((final_trainset['season'] == season_to_predict) & (final_trainset['gw'] == gw_to_predict))]

    # Prepare datetime
    if 'datetime' in final_trainset.columns:
        final_trainset['datetime'] = pd.to_datetime(final_trainset['datetime'])

    # Split data: train vs latest gw in dataset
    latest_season = final_trainset['season'].max()
    latest_gw = final_trainset[final_trainset['season'] == latest_season]['gw'].max()

    test_df = final_trainset[
        (final_trainset['season'] == latest_season) &
        (final_trainset['gw'] == latest_gw)
    ].copy()

    train_df = final_trainset[
        ~(
            (final_trainset['season'] == latest_season) &
            (final_trainset['gw'] == latest_gw)
        )
    ].copy()

    # Drop columns not to be used as features
    target_col = 'outcome'
    drop_cols = ['datetime', 'season', 'gw', 'h_title', 'a_title', target_col]
    feature_cols = [col for col in final_trainset.columns if col not in drop_cols]

    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)

    # Encode target
    le = LabelEncoder()
    train_df['target'] = le.fit_transform(train_df[target_col])
    test_df['target'] = le.transform(test_df[target_col])

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    # Train XGBoost
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    joblib.dump(le, label_encoder_path)

    # Predict
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Simulate returns
    pred_proba_df = pd.DataFrame(y_proba, columns=le.classes_, index=test_df.index)

    # Add predictions and probabilities to test_df
    test_df['pred_label'] = le.inverse_transform(y_pred)
    test_df['pred_proba_h'] = pred_proba_df['H']
    test_df['pred_proba_d'] = pred_proba_df['D']
    test_df['pred_proba_a'] = pred_proba_df['A']

    # Decide bet: choose the class with highest predicted probability
    test_df['prediction'] = pred_proba_df.idxmax(axis=1)
    results_df = test_df.copy()
    results_df = results_df[['datetime', 'h_title', 'a_title', 'book_odds_h', 'book_odds_d', 'book_odds_a', 'prediction', 'outcome']]
    results_df.to_csv(os.path.join(os.path.dirname(model_path), 'results.csv'), index=False)

    # Evaluate
    output_dir = "data/output/train_eval"

    logloss = 0 #log_loss(y_test, y_proba) 
    acc = accuracy_score(y_test, y_pred)

    acc_csv_path = os.path.join(output_dir, "train_acc.csv")
    row = pd.DataFrame([{
        "season": season_to_predict,
        "gw": gw_to_predict,
        "log_loss": round(logloss, 4),
        "accuracy": round(acc, 4)
    }])

    if os.path.exists(acc_csv_path):
        row.to_csv(acc_csv_path, mode='a', header=False, index=False)
    else:
        row.to_csv(acc_csv_path, index=False)

    # Evaluate
    print("\nLog Loss:", logloss)
    print("\nAccuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=10, importance_type='gain', show_values=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()

    return model, le, feature_cols, results_df
