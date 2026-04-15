"""
Tests for dquant.data.factors_utils

Covers:
- calculate_common_factors: default windows, custom windows, missing close, empty DataFrame
- calculate_rsi: value range [0, 100]
- calculate_macd: returns 3 Series of same length as input
- calculate_bollinger: upper > middle > lower invariant
"""

import numpy as np
import pandas as pd
import pytest

from dquant.data.factors_utils import (
    calculate_bollinger,
    calculate_common_factors,
    calculate_macd,
    calculate_rsi,
)


class TestCalculateCommonFactors:
    """Tests for calculate_common_factors."""

    def test_default_windows_produce_expected_columns(self, test_data):
        """Default windows produce momentum_5/10/20, volatility_5/10/20, ma_5/10/20, bias_5/10/20."""
        df = test_data(days=50)
        result = calculate_common_factors(df)

        for w in [5, 10, 20]:
            assert f"momentum_{w}" in result.columns, f"Missing momentum_{w}"
            assert f"volatility_{w}" in result.columns, f"Missing volatility_{w}"
            assert f"ma_{w}" in result.columns, f"Missing ma_{w}"
            assert f"bias_{w}" in result.columns, f"Missing bias_{w}"

    def test_multi_symbol_data(self, test_data):
        """Multi-symbol data is computed per symbol group."""
        symbols = ["000001.SZ", "600000.SH", "300001.SZ"]
        df = test_data(days=50, symbols=symbols)
        result = calculate_common_factors(df)

        # Each symbol should have computed factor values
        for sym in symbols:
            sym_rows = result[result["symbol"] == sym]
            # After enough rows for rolling windows, values should exist
            non_na_momentum = sym_rows["momentum_5"].dropna()
            assert len(non_na_momentum) > 0, f"No momentum_5 values for {sym}"

    def test_custom_window_lists(self, test_data):
        """Custom window lists override defaults."""
        df = test_data(days=60)
        result = calculate_common_factors(
            df,
            momentum_windows=[3, 7],
            volatility_windows=[3],
            ma_windows=[3, 15],
        )

        # Custom windows should be present
        assert "momentum_3" in result.columns
        assert "momentum_7" in result.columns
        assert "volatility_3" in result.columns
        assert "ma_3" in result.columns
        assert "ma_15" in result.columns
        assert "bias_3" in result.columns
        assert "bias_15" in result.columns

        # Default windows should NOT be present
        assert "momentum_5" not in result.columns
        assert "momentum_10" not in result.columns
        assert "momentum_20" not in result.columns

    def test_missing_close_column_returns_unchanged(self, test_data):
        """Early return when 'close' column is absent."""
        df = test_data(days=10)
        df_no_close = df.drop(columns=["close"])

        result = calculate_common_factors(df_no_close)

        # Should return the same DataFrame (no factor columns added)
        assert "momentum_5" not in result.columns
        assert "volatility_5" not in result.columns
        assert len(result) == len(df_no_close)

    def test_empty_dataframe(self):
        """Empty DataFrame should not raise and returns empty result."""
        df = pd.DataFrame(columns=["symbol", "close", "volume"])
        result = calculate_common_factors(df)
        assert isinstance(result, pd.DataFrame)

    def test_volume_factors_present(self, test_data):
        """When volume column exists, volume_ma_5 and volume_ratio are added."""
        df = test_data(days=30)
        result = calculate_common_factors(df)

        assert "volume_ma_5" in result.columns
        assert "volume_ratio" in result.columns


class TestCalculateRSI:
    """Tests for calculate_rsi."""

    def test_rsi_value_range(self):
        """RSI values should be in [0, 100] range (excluding NaN)."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 2))
        rsi = calculate_rsi(prices, period=14)

        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), (
            f"RSI out of range: min={valid.min()}, max={valid.max()}"
        )

    def test_rsi_length_matches_input(self):
        """RSI output has same length as input."""
        prices = pd.Series(np.random.randn(50) + 100)
        rsi = calculate_rsi(prices)
        assert len(rsi) == len(prices)


class TestCalculateMACD:
    """Tests for calculate_macd."""

    def test_returns_three_series_same_length(self):
        """MACD returns (macd, signal, histogram), all same length as input."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)))

        macd, signal, hist = calculate_macd(prices)

        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(hist) == len(prices)

    def test_histogram_equals_macd_minus_signal(self):
        """Histogram = MACD - Signal."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)))

        macd, signal, hist = calculate_macd(prices)

        # Allow small floating-point tolerance
        np.testing.assert_allclose(
            hist.dropna().values,
            (macd - signal).dropna().values,
            atol=1e-10,
        )


class TestCalculateBollinger:
    """Tests for calculate_bollinger."""

    def test_upper_greater_than_middle_greater_than_lower(self):
        """Upper > Middle > Lower invariant (for non-zero std)."""
        np.random.seed(42)
        # Use volatile data so std is non-trivial
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 3))

        upper, middle, lower = calculate_bollinger(prices, window=20, num_std=2.0)

        # Only check rows where all three are non-NaN
        valid_idx = (
            upper.dropna()
            .index.intersection(middle.dropna().index)
            .intersection(lower.dropna().index)
        )

        assert (upper[valid_idx] > middle[valid_idx]).all(), "Upper not > Middle"
        assert (middle[valid_idx] > lower[valid_idx]).all(), "Middle not > Lower"

    def test_output_length_matches_input(self):
        """Bollinger bands have same length as input."""
        prices = pd.Series(np.random.randn(50) + 100)
        upper, middle, lower = calculate_bollinger(prices)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)
