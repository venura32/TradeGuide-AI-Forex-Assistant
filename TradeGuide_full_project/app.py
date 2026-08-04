# =========================
# TradeGuide AI Platform
# Final Year Individual Project
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="TradeGuide – EUR/USD Signal Generator", page_icon="📈", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 22px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        white-space: normal !important;
        overflow: visible !important;
    }
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    hr {
        margin: 1rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 TradeGuide – AI Forex Decision Support Platform")

st.markdown("""
### Intelligent Forex Market Analysis & AI Trading Assistance

TradeGuide is an AI-powered Forex trading decision-support platform developed for financial market analysis, predictive modeling, strategy evaluation, and risk assessment using Machine Learning.
""")

st.markdown("---")

st.sidebar.title("🤖 TradeGuide Platform")
st.sidebar.success("🟢 System Status: ONLINE")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "AI Signal Center",
        "Market Analysis",
        "Risk Management",
        "Backtesting",
        "Trading Simulator",
        "Historical Replay",
        "AI Assistant",
        "Model Performance",
        "Trading Journal"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Platform Capabilities")
st.sidebar.markdown("""
✅ AI Buy/Sell Signal Generation  
✅ Machine Learning Forecasting  
✅ Market Momentum Analysis  
✅ Risk & Volatility Monitoring  
✅ Strategy Backtesting Engine  
✅ AI Trading Scenario Simulation  
✅ Historical AI Market Replay  
✅ Trading Performance Analytics  
✅ Interactive Decision Support  
""")

def get_confidence_color(prob):
    if prob >= 0.70 or prob <= 0.30:
        return "#2ecc71"
    elif prob >= 0.60 or prob <= 0.40:
        return "#f39c12"
    else:
        return "#e74c3c"

def get_confidence_label(prob):
    if prob >= 0.70 or prob <= 0.30:
        return "High Confidence"
    elif prob >= 0.60 or prob <= 0.40:
        return "Moderate Confidence"
    else:
        return "Weak Confidence"

def classify_signal(prob):
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
        'RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50',
        'Volatility', 'Momentum', 'Return_1', 'Return_3',
        'Return_5', 'Volatility_10', 'Volatility_20'
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

    signal_labels = pd.Series([classify_signal(p) for p in probs], index=test_df.index)
    signals = pd.Series(np.where(probs >= threshold, 1, 0), index=test_df.index)
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
    st.sidebar.info("Random Forest is the primary AI model because it gives stronger backtest results. Logistic Regression is included as a baseline comparison model.")

    active_model = rf_model if model_choice == "Random Forest" else lr_model
    bt = run_backtest(df, active_model, features, confidence_threshold)
    latest_signal = bt["signal_labels"].iloc[-1]
    latest_row = df.iloc[-1]
    prob = active_model.predict_proba(df[features].iloc[[-1]])[0][1]
    signal_color = "#2ecc71" if latest_signal in ["BUY", "STRONG BUY"] else "#e74c3c" if latest_signal in ["SELL", "STRONG SELL"] else "#f39c12"
    confidence_color = get_confidence_color(prob)
    confidence_label = get_confidence_label(prob)

    analysis_points = []
    if latest_row["RSI"] > 60:
        analysis_points.append("RSI indicates bullish momentum")
    elif latest_row["RSI"] < 40:
        analysis_points.append("RSI indicates bearish momentum")
    else:
        analysis_points.append("RSI remains in a neutral zone")
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

    if page == "Dashboard":
        st.header("📊 Dashboard Overview")
        st.write("A high-level summary of the latest EUR/USD AI signal, market price, model confidence, and backtest performance.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Date", df.index[-1].strftime("%Y-%m-%d"))
        col2.metric("Close Price", f"{df['Close'].iloc[-1]:.5f}")
        col3.metric("Confidence", f"{prob:.1%}")
        col4.metric("Signal", latest_signal)

        st.subheader("AI Confidence Level")
        st.progress(float(prob))

        st.markdown(
            f"<div style='padding:12px;border-radius:10px;background-color:{confidence_color};color:white;font-weight:bold;text-align:center;font-size:16px;margin-bottom:15px;'>{confidence_label} ({prob:.1%})</div>",
            unsafe_allow_html=True
        )

        st.markdown("### AI Decision Summary")
        if latest_signal in ["STRONG BUY", "BUY"]:
            st.success(f"AI Decision: {latest_signal} | Confidence: {prob:.1%} | Market bias is bullish.")
        elif latest_signal in ["STRONG SELL", "SELL"]:
            st.error(f"AI Decision: {latest_signal} | Confidence: {(1 - prob):.1%} | Market bias is bearish.")
        else:
            st.warning(f"AI Decision: HOLD | Confidence: {prob:.1%} | Market direction is uncertain.")

        st.markdown(
            f"<div style='padding:18px;border-radius:12px;background-color:{signal_color};color:white;font-weight:bold;text-align:center;font-size:22px;margin-top:15px;margin-bottom:15px;'>Current AI Trading Signal: {latest_signal}</div>",
            unsafe_allow_html=True
        )

        st.subheader("Backtest Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{bt['total_return']:.1%}")
        c2.metric("Win Rate", f"{bt['win_rate']:.1%}")
        c3.metric("Sharpe", f"{bt['sharpe']:.2f}")
        c4.metric("Max DD", f"{bt['max_drawdown']:.1%}")

        st.subheader("Recent EUR/USD Price Movement")
        st.line_chart(df["Close"].tail(300))

        st.markdown("---")
        st.subheader("Live Market Status")

        market_trend = "Bullish" if latest_signal in ["BUY", "STRONG BUY"] else "Bearish" if latest_signal in ["SELL", "STRONG SELL"] else "Neutral"
        volatility_status = "High Vol" if latest_row["Volatility"] > df["Volatility"].mean() else "Stable"
        momentum_status = "Positive" if latest_row["Momentum"] > 0 else "Negative"
        confidence_status = "High" if prob >= 0.70 or prob <= 0.30 else "Moderate"

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Trend", market_trend)
        a2.metric("Volatility", volatility_status)
        a3.metric("Momentum", momentum_status)
        a4.metric("Confidence", confidence_status)

        st.info(f"Current market conditions indicate a {market_trend.lower()} environment with {volatility_status.lower()} volatility. The AI model detects {momentum_status.lower()} momentum.")

        if latest_signal in ["BUY", "STRONG BUY"]:
            recommendation_text = "Market conditions currently favor bullish opportunities. Momentum and confidence suggest buyers remain active."
        elif latest_signal in ["SELL", "STRONG SELL"]:
            recommendation_text = "Current market structure favors bearish continuation. Selling pressure and model analysis suggest downside risk remains active."
        else:
            recommendation_text = "The market currently lacks strong directional confirmation. Mixed technical signals suggest uncertainty and consolidation."

        st.info(f"🤖 AI Trading Recommendation: {recommendation_text}")

    elif page == "AI Signal Center":
        st.header("🎯 AI Signal Center")
        st.write("This section displays the latest machine learning-based trading signal and the reasoning behind the decision.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Date", df.index[-1].strftime("%Y-%m-%d"))
        col2.metric("Close Price", f"{df['Close'].iloc[-1]:.5f}")
        col3.metric("Confidence", f"{prob:.1%}")

        st.markdown(
            f"<div style='padding:18px;border-radius:12px;background-color:{signal_color};color:white;font-weight:bold;text-align:center;font-size:22px;margin-top:15px;margin-bottom:15px;'>AI Signal: {latest_signal}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='padding:12px;border-radius:10px;background-color:{confidence_color};color:white;font-weight:bold;text-align:center;font-size:16px;margin-bottom:15px;'>{confidence_label} ({prob:.1%})</div>",
            unsafe_allow_html=True
        )

        if latest_signal in ["STRONG BUY", "BUY"]:
            st.success(f"{latest_signal} — {prob:.1%} confidence\n\nAI Reasoning:\n{analysis_text}\n\nOutlook: Market conditions currently favor upward movement.")
        elif latest_signal in ["STRONG SELL", "SELL"]:
            st.error(f"{latest_signal} — {(1 - prob):.1%} confidence\n\nAI Reasoning:\n{analysis_text}\n\nOutlook: Market conditions currently favor downward movement.")
        else:
            st.warning(f"HOLD / NEUTRAL — signal not strong enough\n\nAI Reasoning:\n{analysis_text}\n\nOutlook: Market direction remains uncertain.")

    elif page == "Market Analysis":
        st.header("📈 Market Analysis")
        st.write("This section shows EUR/USD price movement and key technical indicators used by the AI model.")

        st.subheader("EUR/USD Price History")
        st.line_chart(df["Close"].tail(500))

        st.subheader("Technical Indicator Snapshot")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RSI", f"{latest_row['RSI']:.2f}")
        m2.metric("MACD", f"{latest_row['MACD']:.5f}")
        m3.metric("Momentum", f"{latest_row['Momentum']:.3%}")
        m4.metric("Volatility", f"{latest_row['Volatility']:.5f}")

        st.subheader("Moving Averages")
        st.line_chart(df[["Close", "SMA_20", "SMA_50"]].tail(300))

        st.markdown("---")
        st.subheader("AI Market Condition Analyzer")

        if latest_signal in ["BUY", "STRONG BUY"]:
            market_direction = "Bullish"
        elif latest_signal in ["SELL", "STRONG SELL"]:
            market_direction = "Bearish"
        else:
            market_direction = "Sideways"

        if latest_row["Volatility"] > df["Volatility"].mean():
            volatility_condition = "High Vol"
        else:
            volatility_condition = "Stable"

        if latest_row["Momentum"] > 0:
            momentum_condition = "Positive"
        else:
            momentum_condition = "Negative"

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Direction", market_direction)
        mc2.metric("Volatility", volatility_condition)
        mc3.metric("Momentum", momentum_condition)

        if market_direction == "Bullish":
            ai_market_summary = f"The market currently shows bullish characteristics supported by {momentum_condition.lower()} momentum and {volatility_condition.lower()} volatility. Buyers appear to maintain short-term control."
        elif market_direction == "Bearish":
            ai_market_summary = f"The market currently reflects bearish pressure with {momentum_condition.lower()} momentum under {volatility_condition.lower()} volatility conditions. Selling momentum remains dominant."
        else:
            ai_market_summary = "The market currently appears to be moving sideways with mixed signals. Momentum and volatility conditions suggest market indecision."

        st.info(ai_market_summary)

    elif page == "Risk Management":
        st.header("🛡️ Advanced Risk Management")
        st.write("This section helps traders evaluate trade risk, position sizing, reward potential, and overall trade quality.")

        col1, col2 = st.columns(2)

        with col1:
            account_size = st.number_input("Account Balance ($)", min_value=100.0, value=1000.0, step=100.0)
            risk_percent = st.slider("Risk Per Trade (%)", 0.5, 10.0, 1.0, 0.5)
            stop_loss_pips = st.slider("Stop Loss Distance (Pips)", 5, 100, 30)

        with col2:
            take_profit_pips = st.slider("Take Profit Distance (Pips)", 10, 300, 60)
            leverage = st.selectbox("Leverage", [10, 25, 50, 100, 200])
            trading_style = st.selectbox("Trading Style", ["Scalping", "Day Trading", "Swing Trading"])

        risk_amount = account_size * (risk_percent / 100)
        pip_value = 10
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        reward_amount = (take_profit_pips / stop_loss_pips) * risk_amount
        rr_ratio = take_profit_pips / stop_loss_pips

        if rr_ratio >= 2:
            trade_quality = "Excellent Trade Setup"
            quality_color = "#2ecc71"
        elif rr_ratio >= 1.5:
            trade_quality = "Good Trade Setup"
            quality_color = "#f39c12"
        else:
            trade_quality = "Risky Trade Setup"
            quality_color = "#e74c3c"

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Money At Risk", f"${risk_amount:.2f}")
        r2.metric("Lot Size", f"{lot_size:.2f}")
        r3.metric("Reward", f"${reward_amount:.2f}")
        r4.metric("R/R Ratio", f"1:{rr_ratio:.2f}")

        st.markdown(
            f"<div style='padding:18px;border-radius:12px;background-color:{quality_color};color:white;font-weight:bold;text-align:center;font-size:22px;margin-top:15px;margin-bottom:15px;'>{trade_quality}</div>",
            unsafe_allow_html=True
        )

        st.subheader("AI Risk Evaluation")

        if rr_ratio >= 2:
            st.success(f"Trade setup appears favorable. Trading Style: {trading_style}. Risk Level: Moderate. Reward Potential: Strong.")
        elif rr_ratio >= 1.5:
            st.warning(f"Trade setup shows balanced risk and reward. Trading Style: {trading_style}. Risk Level: Medium. Reward Potential: Reasonable.")
        else:
            st.error(f"Trade setup presents weak reward potential. Trading Style: {trading_style}. Risk Level: High. Reward Potential: Limited.")

        x1, x2, x3 = st.columns(3)
        x1.metric("Drawdown", f"{bt['max_drawdown']:.1%}")
        x2.metric("Confidence", f"{prob:.1%}")
        x3.metric("Market Bias", latest_signal)

        st.info("This module is for educational and decision-support purposes only.")

    elif page == "Backtesting":
        st.header("🔁 Strategy Backtesting")
        st.write("This section evaluates how the AI signal strategy performed on historical EUR/USD test data.")

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Return", f"{bt['total_return']:.1%}")
        b2.metric("Win Rate", f"{bt['win_rate']:.1%}")
        b3.metric("Sharpe Ratio", f"{bt['sharpe']:.2f}")
        b4.metric("Max Drawdown", f"{bt['max_drawdown']:.1%}")

        st.subheader("Cumulative Strategy Return")
        st.line_chart(bt["cumulative"].rename("Cumulative Return"))
        st.caption("Backtesting is based on historical data and does not guarantee future performance.")

    elif page == "Trading Simulator":
        st.header("🧪 AI Trading Scenario Simulator")
        st.write("Adjust market conditions below and observe how the AI model reacts.")

        sim_row = latest_row.copy()
        sim_row["RSI"] = st.slider("RSI", 0.0, 100.0, float(latest_row["RSI"]))
        sim_row["MACD"] = st.slider("MACD", -0.05, 0.05, float(latest_row["MACD"]))
        sim_row["MACD_Signal"] = st.slider("MACD Signal", -0.05, 0.05, float(latest_row["MACD_Signal"]))
        sim_row["Momentum"] = st.slider("Momentum", -0.10, 0.10, float(latest_row["Momentum"]))
        sim_row["Volatility"] = st.slider("Volatility", 0.0, 0.05, float(latest_row["Volatility"]))

        sim_input = pd.DataFrame([sim_row[features]])
        sim_prob = active_model.predict_proba(sim_input)[0][1]
        sim_signal = classify_signal(sim_prob)

        s1, s2 = st.columns(2)
        s1.metric("Sim Confidence", f"{sim_prob:.1%}")
        s2.metric("Sim Signal", sim_signal)

        if sim_signal in ["STRONG BUY", "BUY"]:
            st.success("The AI simulator suggests upward movement based on the selected conditions.")
        elif sim_signal in ["STRONG SELL", "SELL"]:
            st.error("The AI simulator suggests downward movement based on the selected conditions.")
        else:
            st.warning("The AI simulator does not detect a strong trading direction.")

    elif page == "Historical Replay":
        st.header("🕰️ Historical AI Market Replay")
        st.write("Travel back in time and see how the AI would have interpreted historical EUR/USD market conditions. The same trained model is applied to past data, recreating the AI's decision-making process at any point in history.")

        min_date = df.index.min().date()
        max_date = df.index.max().date()

        st.subheader("Quick-Pick Historical Events")
        q1, q2, q3 = st.columns(3)

        if "replay_date" not in st.session_state:
            st.session_state.replay_date = df.index[len(df) // 2].date()

        with q1:
            if st.button("📉 COVID Crash (Mar 2020)"):
                target = date(2020, 3, 18)
                if target < min_date:
                    target = min_date
                if target > max_date:
                    target = max_date
                st.session_state.replay_date = target

        with q2:
            if st.button("⚡ High Volatility (Sep 2022)"):
                target = date(2022, 9, 28)
                if target < min_date:
                    target = min_date
                if target > max_date:
                    target = max_date
                st.session_state.replay_date = target

        with q3:
            if st.button("📊 Recent Market (2025)"):
                target = date(2025, 6, 15)
                if target < min_date:
                    target = min_date
                if target > max_date:
                    target = max_date
                st.session_state.replay_date = target

        st.markdown("---")
        st.subheader("Select a Historical Date")

        selected_date = st.date_input(
            "Pick any date from the dataset",
            value=st.session_state.replay_date,
            min_value=min_date,
            max_value=max_date
        )

        st.session_state.replay_date = selected_date

        target_ts = pd.Timestamp(selected_date)
        position = df.index.get_indexer([target_ts], method="nearest")[0]
        historical_row = df.iloc[position]
        historical_date = df.index[position]

        st.info(f"📅 Closest available trading date: **{historical_date.strftime('%Y-%m-%d')}**")

        hist_input = pd.DataFrame([historical_row[features]])
        hist_prob = active_model.predict_proba(hist_input)[0][1]
        hist_signal = classify_signal(hist_prob)

        hist_signal_color = "#2ecc71" if hist_signal in ["BUY", "STRONG BUY"] else "#e74c3c" if hist_signal in ["SELL", "STRONG SELL"] else "#f39c12"
        hist_conf_color = get_confidence_color(hist_prob)
        hist_conf_label = get_confidence_label(hist_prob)

        st.markdown("---")
        st.subheader("📊 Historical Market Snapshot")

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Date", historical_date.strftime("%Y-%m-%d"))
        h2.metric("Close Price", f"{historical_row['Close']:.5f}")
        h3.metric("RSI", f"{historical_row['RSI']:.2f}")
        h4.metric("MACD", f"{historical_row['MACD']:.5f}")

        h5, h6, h7, h8 = st.columns(4)
        h5.metric("Momentum", f"{historical_row['Momentum']:.3%}")
        h6.metric("Volatility", f"{historical_row['Volatility']:.5f}")
        h7.metric("SMA 20", f"{historical_row['SMA_20']:.5f}")
        h8.metric("SMA 50", f"{historical_row['SMA_50']:.5f}")

        st.markdown("---")
        st.subheader("🤖 AI Historical Interpretation")

        st.markdown(
            f"<div style='padding:18px;border-radius:12px;background-color:{hist_signal_color};color:white;font-weight:bold;text-align:center;font-size:22px;margin-top:15px;margin-bottom:15px;'>AI Signal: {hist_signal}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div style='padding:12px;border-radius:10px;background-color:{hist_conf_color};color:white;font-weight:bold;text-align:center;font-size:16px;margin-bottom:15px;'>{hist_conf_label} ({hist_prob:.1%})</div>",
            unsafe_allow_html=True
        )

        if hist_signal in ["BUY", "STRONG BUY"]:
            hist_direction = "Bullish"
        elif hist_signal in ["SELL", "STRONG SELL"]:
            hist_direction = "Bearish"
        else:
            hist_direction = "Sideways"

        hist_volatility = "High Vol" if historical_row["Volatility"] > df["Volatility"].mean() else "Stable"
        hist_momentum = "Positive" if historical_row["Momentum"] > 0 else "Negative"

        c1, c2, c3 = st.columns(3)
        c1.metric("Market Condition", hist_direction)
        c2.metric("Volatility State", hist_volatility)
        c3.metric("Momentum State", hist_momentum)

        hist_points = []
        if historical_row["RSI"] > 60:
            hist_points.append("RSI indicated bullish momentum")
        elif historical_row["RSI"] < 40:
            hist_points.append("RSI indicated bearish momentum")
        else:
            hist_points.append("RSI remained in a neutral zone")
        if historical_row["MACD"] > historical_row["MACD_Signal"]:
            hist_points.append("MACD trend was bullish")
        else:
            hist_points.append("MACD trend was bearish")
        if historical_row["Momentum"] > 0:
            hist_points.append("Market momentum was positive")
        else:
            hist_points.append("Market momentum was negative")
        if historical_row["Volatility"] > df["Volatility"].mean():
            hist_points.append("Volatility was higher than average")
        else:
            hist_points.append("Volatility was stable")
        hist_analysis = "\n".join([f"• {p}" for p in hist_points])

        st.markdown(f"### 📜 At this historical point, the AI would have interpreted the market as {hist_direction.upper()}")

        if hist_signal in ["STRONG BUY", "BUY"]:
            st.success(f"Historical AI Decision: {hist_signal} | Confidence: {hist_prob:.1%}\n\nMarket Reasoning:\n{hist_analysis}\n\nOutlook at the time: Conditions favored upward movement based on the model's evaluation of technical indicators.")
        elif hist_signal in ["STRONG SELL", "SELL"]:
            st.error(f"Historical AI Decision: {hist_signal} | Confidence: {(1 - hist_prob):.1%}\n\nMarket Reasoning:\n{hist_analysis}\n\nOutlook at the time: Conditions favored downward movement based on the model's evaluation of technical indicators.")
        else:
            st.warning(f"Historical AI Decision: HOLD | Confidence: {hist_prob:.1%}\n\nMarket Reasoning:\n{hist_analysis}\n\nOutlook at the time: Market direction was uncertain, with mixed technical signals.")

        st.markdown("---")
        st.subheader("📈 Price Movement Around This Date")
        st.caption("Showing 100 trading days before and 30 trading days after the selected date.")

        start_pos = max(0, position - 100)
        end_pos = min(len(df), position + 31)
        chart_df = df.iloc[start_pos:end_pos][["Close", "SMA_20", "SMA_50"]]

        st.line_chart(chart_df)

        st.info("This Historical Replay feature uses the same trained AI model to re-evaluate past market conditions. It demonstrates how the model would have responded across different market environments, supporting model explainability and historical analysis.")

    elif page == "AI Assistant":
        st.header("🤖 AI Market Assistant")
        st.write("This section explains the current AI decision in a more human-readable way.")

        st.info(f"Current Signal: {latest_signal}\n\nConfidence: {prob:.1%}\n\nAI Reasoning:\n{analysis_text}\n\nThe assistant interprets technical indicators such as RSI, MACD, momentum, volatility, and moving averages to explain why the model leans toward the current market direction.")

        st.markdown("---")
        st.subheader("🧠 AI Insights Panel")

        trend_strength = "Strong" if prob >= 0.75 or prob <= 0.25 else "Moderate" if prob >= 0.60 or prob <= 0.40 else "Weak"
        risk_level = "High" if bt["max_drawdown"] < -0.10 else "Medium" if bt["max_drawdown"] < -0.05 else "Low"
        market_sentiment = "Bullish" if latest_signal in ["BUY", "STRONG BUY"] else "Bearish" if latest_signal in ["SELL", "STRONG SELL"] else "Neutral"
        recommendation = "Potential buying opportunity detected." if latest_signal in ["BUY", "STRONG BUY"] else "Potential selling opportunity detected." if latest_signal in ["SELL", "STRONG SELL"] else "Wait for stronger confirmation before entering trades."
        confidence_category = "Very High" if prob >= 0.80 or prob <= 0.20 else "High" if prob >= 0.70 or prob <= 0.30 else "Moderate" if prob >= 0.60 or prob <= 0.40 else "Low"

        c1, c2, c3 = st.columns(3)
        c1.metric("Sentiment", market_sentiment)
        c2.metric("Trend", trend_strength)
        c3.metric("Risk", risk_level)

        c4, c5 = st.columns(2)
        c4.metric("Confidence Tier", confidence_category)
        c5.metric("Trade Bias", latest_signal)

        st.success(f"AI Recommendation: {recommendation}")

    elif page == "Model Performance":
        st.header("📊 Model Performance")
        st.write("This section compares the machine learning models used in the system.")

        c1, c2, c3 = st.columns(3)
        c1.metric("RF Accuracy", f"{rf_acc:.1%}")
        c2.metric("LR Accuracy", f"{lr_acc:.1%}")
        c3.metric("Active Model", model_choice)

        st.caption("Random Forest is the main AI model. Logistic Regression is included as a baseline comparison model.")

        if model_choice == "Random Forest":
            st.subheader("Feature Importance")
            imp = pd.DataFrame({
                "Feature": features,
                "Importance": rf_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            st.bar_chart(imp.set_index("Feature"))

        if show_raw:
            st.subheader("Recent Processed Data")
            st.dataframe(df.tail(50))

    elif page == "Trading Journal":
        st.header("📝 Trading Journal")
        st.write("This section allows users to record trade ideas, observations, and reflections during analysis.")

        journal_date = st.date_input("Trade Date")
        journal_pair = st.text_input("Currency Pair", value="EUR/USD")
        journal_signal = st.selectbox("Trade Bias", ["BUY", "SELL", "HOLD", "WAIT"])
        journal_notes = st.text_area("Trade Notes / Reasoning")

        st.subheader("Journal Preview")
        st.write(f"Date: {journal_date}")
        st.write(f"Pair: {journal_pair}")
        st.write(f"Bias: {journal_signal}")
        st.write(f"Notes: {journal_notes}")

        st.info("This journal is currently session-based for demo purposes. Persistent saving can be added later.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:13px;padding:10px;'>"
        "<b>TradeGuide AI Platform</b> | Powered by Machine Learning, Financial Analytics, and AI Decision Support<br>"
        "Final Year Project – PUSL3190 | University of Plymouth"
        "</div>",
        unsafe_allow_html=True
    )

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.exception(e)