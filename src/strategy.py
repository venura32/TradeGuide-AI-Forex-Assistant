import pandas as pd

def apply_strategy(results, probabilities, threshold=0.55):
    final_results = results.copy()

    final_results["Up_Probability"] = probabilities

    final_results["ML_Signal"] = (
        final_results["Up_Probability"] > threshold
    ).astype(int)

    # Use yesterday's signal on today's return to avoid look-ahead bias
    final_results["Trade_Signal"] = final_results["ML_Signal"].shift(1)

    final_results["ML_Strategy"] = (
        final_results["Trade_Signal"] * final_results["Return"]
    )

    final_results["Cumulative_Strategy"] = (
        1 + final_results["ML_Strategy"].fillna(0)
    ).cumprod()

    return final_results