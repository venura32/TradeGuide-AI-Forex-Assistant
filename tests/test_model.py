import pandas as pd
from src.model import add_features, train_model

def test_add_features_creates_expected_columns():
    data = {"Close": [100, 101, 100.5, 102, 103, 102.5, 104, 105, 104.5, 106, 107, 106.5, 108, 109, 108.5, 110, 111, 110.5, 112, 113, 112.5, 114, 115, 114.5, 116, 117, 116.5, 118, 119, 120]}
    sample_df = pd.DataFrame(data)
    result = add_features(sample_df)
    assert "RSI" in result.columns

def test_add_features_creates_all_expected_columns():
    data = {"Close": [100, 101, 100.5, 102, 103, 102.5, 104, 105, 104.5, 106, 107, 106.5, 108, 109, 108.5, 110, 111, 110.5, 112, 113, 112.5, 114, 115, 114.5, 116, 117, 116.5, 118, 119, 120]}
    sample_df = pd.DataFrame(data)
    result = add_features(sample_df)
    expected_columns = ["Returns", "Momentum", "Volatility", "RSI", "MACD", "MACD_Signal", "Target"]
    for col in expected_columns:
        assert col in result.columns

def test_add_features_has_no_nulls():
    data = {"Close": [100, 101, 100.5, 102, 103, 102.5, 104, 105, 104.5, 106, 107, 106.5, 108, 109, 108.5, 110, 111, 110.5, 112, 113, 112.5, 114, 115, 114.5, 116, 117, 116.5, 118, 119, 120]}
    sample_df = pd.DataFrame(data)
    result = add_features(sample_df)
    assert result.isnull().sum().sum() == 0

def test_train_model_returns_two_fitted_models():
    data = {"Close": [100, 101, 100.5, 102, 103, 102.5, 104, 105, 104.5, 106, 107, 106.5, 108, 109, 108.5, 110, 111, 110.5, 112, 113, 112.5, 114, 115, 114.5, 116, 117, 116.5, 118, 119, 120]}
    sample_df = pd.DataFrame(data)
    result = add_features(sample_df)
    feature_cols = ["Returns", "Momentum", "Volatility", "RSI", "MACD", "MACD_Signal"]
    X = result[feature_cols]
    y = result["Target"]
    rf, lr = train_model(X, y)
    assert hasattr(rf, "predict")
    assert hasattr(lr, "predict")