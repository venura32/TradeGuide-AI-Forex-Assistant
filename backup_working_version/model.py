import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def predict_probabilities(model, X):
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities