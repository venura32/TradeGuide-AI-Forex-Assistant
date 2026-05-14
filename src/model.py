from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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