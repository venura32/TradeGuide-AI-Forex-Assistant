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

def run_backtest(df, model, features, threshold=0.55):
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].copy()

    probs = model.predict_proba(test_df[features])[:, 1]

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
        "total_trades": total_trades
    }
    

try:
    df_raw = load_data()
    df = add_features(df_raw)
    rf_model, lr_model, rf_acc, lr_acc, features = train_models(df)

    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
    confidence_threshold = st.sidebar.slider("Signal Confidence Threshold", 0.50, 0.90, 0.60, 0.01)
    show_raw = st.sidebar.checkbox("Show Raw Data", value=False)

    active_model = rf_model if model_choice == "Random Forest" else lr_model

    st.subheader("🔔 Latest Trading Signal")
    prob = active_model.predict_proba(df[features].iloc[[-1]])[0][1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Date", df.index[-1].strftime('%Y-%m-%d'))
    col2.metric("EUR/USD Close", f"{df['Close'].iloc[-1]:.5f}")
    col3.metric("Model Confidence", f"{prob:.1%}")

    if prob >= confidence_threshold:
        st.success(f"🟢 BUY Signal — {prob:.1%} confidence price will rise.")
    elif prob <= 1 - confidence_threshold:
        st.error(f"🔴 SELL Signal — {1-prob:.1%} confidence price will fall.")
    else:
        st.warning(f"⚪ HOLD — Confidence too low ({prob:.1%}).")

    st.markdown("---")
    st.subheader("📊 EUR/USD Price History")
    st.line_chart(df['Close'].tail(500))

    st.markdown("---")
    st.subheader("🎯 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest Accuracy", f"{rf_acc:.1%}")
    c2.metric("Logistic Regression Accuracy", f"{lr_acc:.1%}")
    c3.metric("Active Model", model_choice)

    st.markdown("---")
    st.subheader("📉 Backtest Results")
    bt = run_backtest(df, active_model, features, confidence_threshold)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Total Return", f"{bt['total_return']:.1%}")
    b2.metric("Win Rate", f"{bt['win_rate']:.1%}")
    b3.metric("Sharpe Ratio", f"{bt['sharpe']:.2f}")
    b4.metric("Max Drawdown", f"{bt['max_drawdown']:.1%}")
    st.line_chart(bt['cumulative'].rename("Cumulative Return"))

    st.markdown("---")
    if model_choice == "Random Forest":
        st.subheader("🔍 Feature Importance")
        imp = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_}).sort_values('Importance', ascending=False)
        st.bar_chart(imp.set_index('Feature'))

    if show_raw:
        st.dataframe(df.tail(50))

    st.caption("⚠️ For educational purposes only. Not financial advice.")

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.exception(e)