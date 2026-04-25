"""A-share factor expressions using the Qlib Expression Engine.

Each entry is a Qlib expression string that can be used directly with
``D.features()`` or in handler configurations.

Categories
----------
- Volume-price divergence
- Momentum
- Volatility
- Mean reversion
"""

CUSTOM_FACTORS: list[str] = [
    # -- Volume-price divergence --
    "Corr($close, $volume, 20)",
    "$volume / Mean($volume, 20)",

    # -- Momentum --
    "Mean($close, 20) / Mean($close, 60) - 1",
    "$close / Delay($close, 20) - 1",
    "$close / Delay($close, 60) - 1",

    # -- Volatility --
    "Std($close / Delay($close, 1) - 1, 20)",
    "Std($close / Delay($close, 1) - 1, 60)",
    "Std($close / Delay($close, 1) - 1, 20) / Std($close / Delay($close, 1) - 1, 60)",

    # -- Mean reversion --
    "$close / Mean($close, 20) - 1",
]

# Compact aliases mapping (friendly name → expression)
FACTOR_ALIASES: dict[str, str] = {
    "pv_corr_20": CUSTOM_FACTORS[0],
    "volume_ratio_20": CUSTOM_FACTORS[1],
    "ma_ratio_20_60": CUSTOM_FACTORS[2],
    "ret_20d": CUSTOM_FACTORS[3],
    "ret_60d": CUSTOM_FACTORS[4],
    "realized_vol_20": CUSTOM_FACTORS[5],
    "realized_vol_60": CUSTOM_FACTORS[6],
    "vol_ratio_20_60": CUSTOM_FACTORS[7],
    "deviation_ma20": CUSTOM_FACTORS[8],
}
