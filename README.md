# Premier League Betting Model

A machine learning pipeline that predicts Premier League match outcomes and simulates betting strategies to evaluate profitability.  
This project uses historical match data, ELO ratings, squad market values, xG statistics and home/away form to generate model-driven betting decisions.

---

## Features
- **Fully automated pipeline** — from raw data to predictions & ROI simulation.
- **Feature-rich models**:
  - ELO ratings 📊
  - Expected Goals (xG)
  - Squad market value & average player age
  - Head-to-head history
- **Weekly retraining** with up-to-date match data.
- **ROI simulation** — tests “£1 per match” betting strategy over historical seasons.
- Modular design for easy feature experimentation.

## Folder Structure
prem-bet-model/
│
├── data/
│ ├── input/ # Initial merge, prep for feature engineering
│ ├── raw/ # Raw downloaded data
│ ├── output/ # Predictions & evaluations
│
├── models/ # Trained models & encoders
├── scripts/ # Modules contributing to the pipeline
│ 
├── config.yaml # Pipeline configuration
├── run_pipeline.py # Orchestrates the full pipeline
├── requirements.txt # Dependencies
├── README.md
└── .gitignore

## Pipeline overview

data_load.py -> data_align.py -> feature_engineering.py -> baseline_model.py -> simulate_returns.py -> predict.py -> report.py

## Outputs
**Prediction Dashboard**
https://the-hearty-cash-machine.streamlit.app/

**Unpcoming GW predictions**
<img width="1388" height="290" alt="image" src="https://github.com/user-attachments/assets/4706668e-d593-4251-a378-cf026001ef3f" />

**Last GW Review**
<img width="382" height="56" alt="image" src="https://github.com/user-attachments/assets/f7ae5541-4656-4b98-a8a8-1e524f12c005" />

## Future Improvements
- Live match odds scraping for automated bet placement.
- Injured players valuations
- Incorporating weather & travel fatigue factors.
- Deeper player performance metrics (ratings trend, match fitness).

## Data Sources
- ClubElo —> ELO ratings
- Understat —> xG statistics
- Transfermarkt —> Squad values & player injuries

