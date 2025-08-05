import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="The Hearty Cash Machine", layout="wide")

st.markdown("<h1 style='text-align: center;'>💰💸 The Hearty Prem Cash Machine 💸💰</h1>", unsafe_allow_html=True)

# --- Load Predictions ---
@st.cache_data
def load_predictions(path):
    return pd.read_csv(path)

# --- Load Train Accuracy ---
@st.cache_data
def load_accuracy(path):
    return pd.read_csv(path)

# --- Display Predictions ---
st.header("📅 Match Predictions")
df_preds = load_predictions("data/output/predictions/2025_gw1.csv")
season = df_preds['season'].iloc[0]
gw = df_preds['gw'].iloc[0]

st.markdown(f"**Season:** {season} | **Gameweek:** {gw} | **Matches:** {len(df_preds)}")

cols = st.columns(3)  # 3 cards per row

for idx, row in df_preds.iterrows():
    col = cols[idx % 3]  # Cycle through the columns (0,1,2)

    with col:
        st.markdown(
            f"""
            **🗓️ {row['datetime']}**
            ### {row['h_title']} 🆚 {row['a_title']}  
            **Model Prediction:** `{row['predicted_outcome'].upper()}`  
            **Best Value Bet:** `{row['bet_decision'].upper()}`  
            ---
            | Model Predictions      | Bookie Odds        |
            |:---------------------:|:------------------:|
            | 🏠 {row['pred_H']*100:.1f}%       | 🏠 {row['book_odds_h']:.2f}    |
            | 🤝 {row['pred_D']*100:.1f}%       | 🤝 {row['book_odds_d']:.2f}    |
            | 🛫 {row['pred_A']*100:.1f}%       | 🛫 {row['book_odds_a']:.2f}    |
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")



st.dataframe(
    df_preds[['datetime', 'h_title', 'a_title', 'book_odds_h', 'book_odds_d', 'book_odds_a', 'pred_H', 'pred_D', 'pred_A', 'predicted_outcome', 'bet_decision']],
    use_container_width=True
)

st.header("🤓 Model Statistics")

# --- Feature Importance and Accuracy Side-by-Side ---
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "<h3 style='margin-bottom: 0.5rem;'>🧠 Feature Importance (Model Weights)</h3>",
        unsafe_allow_html=True
    )
    st.image("data/output/train_eval/feature_importance.png")

with col2:
    st.markdown(
        "<h3 style='margin-bottom: 0.5rem;'>📉 Training Performance Over Time</h3>",
        unsafe_allow_html=True
    )
    df_acc = load_accuracy("data/output/train_eval/train_acc.csv")

    fig, ax1 = plt.subplots(figsize=(6, 4))  # smaller to fit in column

    color = 'tab:red'
    ax1.set_xlabel('Gameweek')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(df_acc['gw'], df_acc['accuracy'], marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Log Loss', color=color)
    ax2.plot(df_acc['gw'], df_acc['log_loss'], marker='x', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    st.pyplot(fig)

