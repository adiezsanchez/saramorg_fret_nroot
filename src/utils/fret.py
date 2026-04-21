import numpy as np
import pandas as pd

def compute_fret_ratios(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Compute FRET ratios (raw and normalized) from per-nucleus intensity features.

    This function calculates FRET (Förster Resonance Energy Transfer) ratios using donor and acceptor channels,
    both as the sum and mean intensity measurements, and appends these ratios as new columns to the input dataframe.

    Normalization scope is per image.

    The following columns are expected in the input DataFrame:
        - "edCerulean_CTRL_sum_int" : Donor signal (sum intensity)
        - "edCitrine_FRET_sum_int"  : Acceptor FRET signal (sum intensity)
        - "edCerulean_CTRL_mean_int": Donor signal (mean intensity)
        - "edCitrine_FRET_mean_int" : Acceptor FRET signal (mean intensity)

    Added columns:
        - "FRET_ratio_sum"         : Raw FRET ratio from sums
        - "FRET_ratio_mean"        : Raw FRET ratio from means
        - "FRET_ratio_sum_norm"    : FRET ratio (sum-based) normalized to [0, 1]
        - "FRET_ratio_mean_norm"   : FRET ratio (mean-based) normalized to [0, 1]

    Args:
        df (pd.DataFrame): Per-nucleus feature table containing intensity columns for donor and acceptor markers.

    Returns:
        pd.DataFrame: The same DataFrame with new FRET ratio columns appended.
    """
    # Extract donor and acceptor signals from the feature table
    dd_sum = df["edCerulean_CTRL_sum_int"]
    da_sum = df["edCitrine_FRET_sum_int"]

    dd_mean = df["edCerulean_CTRL_mean_int"]
    da_mean = df["edCitrine_FRET_mean_int"]

    # Compute mask for valid signals (avoid division by zero)
    valid_sum = (dd_sum > 0) & (da_sum > 0)
    valid_mean = (dd_mean > 0) & (da_mean > 0)

    # Compute raw FRET ratios (set to NaN if invalid)
    df["FRET_ratio_sum"] = np.where(valid_sum, da_sum / dd_sum, np.nan)
    df["FRET_ratio_mean"] = np.where(valid_mean, da_mean / dd_mean, np.nan)

    # Helper function to normalize series between 0 and 1
    def normalize(series: "pd.Series") -> "pd.Series":
        min_val = series.min(skipna=True)
        max_val = series.max(skipna=True)
        if max_val == min_val:
            # Avoid division by zero if constant or all NaN
            return pd.Series(0, index=series.index)
        return (series - min_val) / (max_val - min_val)

    # Compute normalized FRET ratios (range: 0 to 1)
    df["FRET_ratio_sum_norm"] = normalize(df["FRET_ratio_sum"])
    df["FRET_ratio_mean_norm"] = normalize(df["FRET_ratio_mean"])

    return df