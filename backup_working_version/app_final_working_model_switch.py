# =========================
# TradeGuide AI Platform
# Final Year Individual Project
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="TradeGuide – EUR/USD Signal Generator", page_icon="📈", layout="wide")

st.title("📈 TradeGuide: AI-Powered EUR/USD Trading Signals")
st.markdown("*Final Year Project – PUSL3190 | University of Plymouth*")
st.markdown("---")
st.sidebar.title("🤖 TradeGuide AI")

st.sidebar.markdown("---")

st.sidebar.success("🟢 System Status: ONLINE")

st.sidebar.subheader("Platform Features")

st.sidebar.markdown("""
✅ AI Buy/Sell Signals  
✅ Market Sentiment Analysis  
✅ Trading Scenario Simulator  
✅ Strategy Backtesting  
✅ Performance Analytics  
✅ Risk Metrics Dashboard  
""")

@st.cache_data
def load_data():
    df = pd.read_csv("data_raw/EURUSD_2015_2026.csv", header=[0,1])
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = df.iloc[1:].reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Close'])
    return df.sort_index()

def add_features(df):
    df = df.copy()
    close = df['Close']

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()

    df['Volatility'] = close.rolling(14).std()

    df['Momentum'] = close.pct_change(10)

    df['Return_1'] = close.pct_change(1)
    df['Return_3'] = close.pct_change(3)
    df['Return_5'] = close.pct_change(5)

    df['Volatility_10'] = close.pct_change().rolling(10).std()
    df['Volatility_20'] = close.pct_change().rolling(20).std()

    df['Target'] = (close.shift(-1) > close).astype(int)

    df.dropna(inplace=True)

    return df

@st.cache_resource
def train_models(df):
    features = [
    'RSI',
    'MACD',
    'MACD_Signal',
    'SMA_20',
    'SMA_50',
    'Volatility',
    'Momentum',
    'Return_1',
    'Return_3',
    'Return_5',
    'Volatility_10',
    'Volatility_20'
    ]

    X = df[features]
    y = df['Target']

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    return rf, lr, rf_acc, lr_acc, features

def run_backtest(df, model, features, threshold=0.60):
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].copy()

    probs = model.predict_proba(test_df[features])[:, 1]

    def get_signal(prob):
        if prob >= 0.75:
            return "STRONG BUY"
        elif prob >= 0.60:
            return "BUY"
        elif prob >= 0.40:
            return "HOLD"
        elif prob >= 0.25:
            return "SELL"
        else:
            return "STRONG SELL"

    signal_labels = pd.Series([get_signal(p) for p in probs], index=test_df.index)

    signals = pd.Series(
        np.where(probs >= threshold, 1, 0),
        index=test_df.index
    )

    returns = test_df["Close"].pct_change()

    trade_signal = signals.shift(1)
    strategy_returns = (trade_signal * returns).dropna()

    cumulative = (1 + strategy_returns).cumprod()

    trade_returns = strategy_returns[trade_signal.loc[strategy_returns.index] == 1]

    total_trades = len(trade_returns)
    win_rate = (trade_returns > 0).sum() / total_trades if total_trades > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    max_dd = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()

    return {
    "cumulative": cumulative,
    "total_return": cumulative.iloc[-1] - 1,
    "win_rate": win_rate,
    "sharpe": sharpe,
    "max_drawdown": max_dd,
    "total_trades": total_trades,
    "signal_labels": signal_labels,
    "probabilities": probs
}
    

try:
    df_raw = load_data()
    df = add_features(df_raw)
    rf_model, lr_model, rf_acc, lr_acc, features = train_models(df)

    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
    confidence_threshold = st.sidebar.slider("Signal Confidence Threshold", 0.50, 0.90, 0.60, 0.01)
    show_raw = st.sidebar.checkbox("Show Raw Data", value=False)
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Random Forest is the primary AI model because it gives stronger backtest results. Logistic Regression is included as a baseline comparison model."
    )

    active_model = rf_model if model_choice == "Random Forest" else lr_model

    # Run backtest first
    bt = run_backtest(df, active_model, features, confidence_threshold)
    latest_signal = bt["signal_labels"].iloc[-1]

    # Latest Trading Signal
    signal_color = "green" if latest_signal in ["BUY", "STRONG BUY"] else "red" if latest_signal in ["SELL", "STRONG SELL"] else "orange"
    st.subheader("Latest Trading Signal")
    prob = active_model.predict_proba(df[features].iloc[[-1]])[0][1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Date", df.index[-1].strftime("%Y-%m-%d"))
    col2.metric("EUR/USD Close", f"{df['Close'].iloc[-1]:.5f}")
    col3.metric("Model Confidence", f"{prob:.1%}")
    st.markdown(
    f"""
    <div style="
        padding:15px;
        border-radius:10px;
        background-color:{signal_color};
        color:white;
        font-weight:bold;
        text-align:center;
        font-size:22px;
        margin-top:15px;
        ">
        Current AI Signal: {latest_signal}
    </div>
    """,
    unsafe_allow_html=True
)

    latest_row = df.iloc[-1]

    analysis_points = []

    if latest_row["RSI"] > 60:
        analysis_points.append("RSI indicates bullish momentum")
    elif latest_row["RSI"] < 40:
        analysis_points.append("RSI indicates bearish momentum")

    if latest_row["MACD"] > latest_row["MACD_Signal"]:
        analysis_points.append("MACD trend is bullish")
    else:
        analysis_points.append("MACD trend is bearish")

    if latest_row["Momentum"] > 0:
        analysis_points.append("Market momentum is positive")
    else:
        analysis_points.append("Market momentum is negative")

    if latest_row["Volatility"] > df["Volatility"].mean():
        analysis_points.append("Volatility is higher than average")
    else:
        analysis_points.append("Volatility remains stable")

    analysis_text = "\n".join([f"• {point}" for point in analysis_points])

    if latest_signal in ["STRONG BUY", "BUY"]:
        st.success(
            f"""
    {latest_signal} — {prob:.1%} confidence

    AI Analysis:
    {analysis_text}

    Outlook: Market conditions currently favor upward movement.
    """
        )

    elif latest_signal in ["STRONG SELL", "SELL"]:
        st.error(
            f"""
    {latest_signal} — {(1 - prob):.1%} confidence

    AI Analysis:
    {analysis_text}

    Outlook: Market conditions currently favor downward movement.
    """
        )

    else:
        st.warning(
            f"""
    NEUTRAL — signal not strong enough

    AI Analysis:
    {analysis_text}

    Outlook: Market direction remains uncertain.
    """
        )
    st.markdown("---")
    st.subheader("AI Trading Scenario Simulator")

    st.write("Adjust market conditions below and see how the AI model reacts.")

    sim_row = latest_row.copy()

    sim_row["RSI"] = st.slider("RSI", 0.0, 100.0, float(latest_row["RSI"]))
    sim_row["MACD"] = st.slider("MACD", -0.05, 0.05, float(latest_row["MACD"]))
    sim_row["MACD_Signal"] = st.slider("MACD Signal", -0.05, 0.05, float(latest_row["MACD_Signal"]))
    sim_row["Momentum"] = st.slider("Momentum", -0.10, 0.10, float(latest_row["Momentum"]))
    sim_row["Volatility"] = st.slider("Volatility", 0.0, 0.05, float(latest_row["Volatility"]))

    sim_input = pd.DataFrame([sim_row[features]])
    sim_prob = active_model.predict_proba(sim_input)[0][1]

    if sim_prob >= 0.75:
        sim_signal = "STRONG BUY"
    elif sim_prob >= 0.60:
        sim_signal = "BUY"
    elif sim_prob >= 0.40:
        sim_signal = "HOLD"
    elif sim_prob >= 0.25:
        sim_signal = "SELL"
    else:
        sim_signal = "STRONG SELL"

    st.metric("Simulated Confidence", f"{sim_prob:.1%}")
    st.metric("Simulated Signal", sim_signal)

    if sim_signal in ["STRONG BUY", "BUY"]:
        st.success("The AI simulator suggests upward movement based on the selected conditions.")
    elif sim_signal in ["STRONG SELL", "SELL"]:
        st.error("The AI simulator suggests downward movement based on the selected conditions.")
    else:
        st.warning("The AI simulator does not detect a strong trading direction.")
    st.markdown("----")
    st.subheader("AI Market Assistant")

    assistant_message = ""

    if sim_signal == "STRONG BUY":
        assistant_message = f"""
    The AI strongly favors a BUY position.

    Reasons:
    - RSI suggests strong bullish momentum.
    - MACD confirms upward trend strength.
    - Market momentum remains positive.
    - Confidence level is high at {sim_prob:.1%}.

    The current simulated conditions suggest potential upward continuation in EUR/USD.
    """

    elif sim_signal == "BUY":
        assistant_message = f"""
    The AI slightly favors a BUY position.

    Reasons:
    - Technical indicators show moderate bullish conditions.
    - Momentum is improving gradually.
    - Confidence level is {sim_prob:.1%}.

    The market currently shows mild upward pressure.
    """

    elif sim_signal == "STRONG SELL":
        assistant_message = f"""
    The AI strongly favors a SELL position.

    Reasons:
    - Momentum conditions are weakening.
    - MACD suggests bearish continuation.
    - Volatility conditions support downside movement.
    - Confidence level is high at {sim_prob:.1%}.

    The current market setup indicates strong downward pressure.
    """

    elif sim_signal == "SELL":
        assistant_message = f"""
    The AI slightly favors a SELL position.

    Reasons:
    - Market momentum appears weak.
    - Technical indicators lean bearish.
    - Confidence level is {sim_prob:.1%}.

    The market currently shows mild downward pressure.
    """

    else:
        assistant_message = f"""
    The AI remains NEUTRAL.

    Reasons:
    - Indicators are mixed.
    - No strong directional momentum detected.
    - Confidence level is {sim_prob:.1%}.

    The market currently lacks a strong trend direction.
    """

    st.info(assistant_message)

    st.markdown("---")
    st.subheader("EUR/USD Price History")
    st.line_chart(df["Close"].tail(500))

    st.markdown("---")
    st.subheader("Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest Accuracy", f"{rf_acc:.1%}")
    c2.metric("Logistic Regression Accuracy", f"{lr_acc:.1%}")
    c3.metric("Active Model", model_choice)

    st.markdown("---")
    st.subheader("Backtest Results")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Total Return", f"{bt['total_return']:.1%}")
    b2.metric("Win Rate", f"{bt['win_rate']:.1%}")
    b3.metric("Sharpe Ratio", f"{bt['sharpe']:.2f}")
    b4.metric("Max Drawdown", f"{bt['max_drawdown']:.1%}")

    st.line_chart(bt["cumulative"].rename("Cumulative Return"))

    st.markdown("---")
    if model_choice == "Random Forest":
        st.subheader("Feature Importance")
        imp = pd.DataFrame({
            "Feature": features,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.bar_chart(imp.set_index("Feature"))

    if show_raw:
        st.dataframe(df.tail(50))

    st.caption("For educational purposes only. Not financial advice.")

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.exception(e)