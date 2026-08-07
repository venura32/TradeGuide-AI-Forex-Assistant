import pandas as pd
from src.strategy import apply_strategy

def test_apply_strategy_signal_above_threshold():
    results = pd.DataFrame({"Return": [0.01, 0.02, -0.01, 0.03]})
    probabilities = [0.9, 0.3, 0.6, 0.2]
    output = apply_strategy(results, probabilities, threshold=0.55)
    assert list(output["ML_Signal"]) == [1, 0, 1, 0]