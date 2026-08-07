import pandas as pd
import pytest
from src.strategy import apply_strategy

def test_apply_strategy_signal_above_threshold():
    results = pd.DataFrame({"Return": [0.01, 0.02, -0.01, 0.03]})
    probabilities = [0.9, 0.3, 0.6, 0.2]
    output = apply_strategy(results, probabilities, threshold=0.55)
    assert list(output["ML_Signal"]) == [1, 0, 1, 0]

def test_trade_signal_is_shifted_ml_signal():
    results = pd.DataFrame({"Return": [0.01, 0.02, -0.01, 0.03]})
    probabilities = [0.9, 0.3, 0.6, 0.2]
    output = apply_strategy(results, probabilities, threshold=0.55)
    assert output["Trade_Signal"].equals(output["ML_Signal"].shift(1))

def test_cumulative_strategy_compounds_correctly():
    results = pd.DataFrame({"Return": [0.10, -0.05, 0.02]})
    probabilities = [0.9, 0.9, 0.9]  # all above threshold -> ML_Signal = [1, 1, 1]
    output = apply_strategy(results, probabilities, threshold=0.55)
    expected = [1.0, 0.95, 0.969]
    assert output["Cumulative_Strategy"].tolist() == pytest.approx(expected)