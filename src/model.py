from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def add_features(df):
    df = df.copy()

    df['Returns'] = df['Close'].pct_change()
    df['Momentum'] = df['Close'].pct_change(10)
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() / (-df['Close'].diff().clip(upper=0)).rolling(14).mean()))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df

def train_model(X_train, y_train):
    # Model 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Model 2: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    return rf, lr