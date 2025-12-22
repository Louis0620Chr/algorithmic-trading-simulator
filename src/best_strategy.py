import pandas as pd


def select_best(results_dataframe: pd.DataFrame, metric: str = "sharpe_ratio") -> pd.Series:
    if results_dataframe.empty:
        raise ValueError("results_dataframe is empty or missing. Run the grid search first.")
    best_row = results_dataframe.loc[results_dataframe[metric].idxmax()]
    return best_row