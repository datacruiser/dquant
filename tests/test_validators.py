"""
数据验证器测试
"""

import numpy as np
import pandas as pd
import pytest

from dquant.data.validators import DataCleaner, DataQualityReport, DataValidator


def _make_valid_data(rows=100):
    """创建有效测试数据"""
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * rows,
            "open": np.linspace(10, 15, rows),
            "high": np.linspace(10.5, 15.5, rows),
            "low": np.linspace(9.5, 14.5, rows),
            "close": np.linspace(10.1, 15.1, rows),
            "volume": np.random.randint(100000, 200000, rows),
        },
        index=dates,
    )


# ============================================================
# DataValidator 测试
# ============================================================


class TestCheckRequiredColumns:
    def test_all_present(self):
        df = _make_valid_data()
        passed, missing = DataValidator.check_required_columns(df, ["symbol", "close"])
        assert passed is True
        assert missing == []

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        passed, missing = DataValidator.check_required_columns(df, ["a", "c", "d"])
        assert passed is False
        assert set(missing) == {"c", "d"}


class TestCheckMissingValues:
    def test_no_missing(self):
        df = _make_valid_data()
        result = DataValidator.check_missing_values(df)
        assert all(v == 0 for v in result.values())

    def test_with_missing(self):
        df = _make_valid_data()
        df.loc[0, "close"] = np.nan
        df.loc[1, "close"] = np.nan
        result = DataValidator.check_missing_values(df, ["close", "volume"])
        assert result["close"] == 2

    def test_specific_columns(self):
        df = _make_valid_data()
        df.loc[0, "close"] = np.nan
        result = DataValidator.check_missing_values(df, ["close"])
        assert result["close"] == 1
        assert "open" not in result


class TestCheckDuplicates:
    def test_no_duplicates(self):
        df = _make_valid_data()
        assert DataValidator.check_duplicates(df) == 0

    def test_with_duplicates(self):
        df = _make_valid_data(rows=5)
        df = pd.concat([df, df.iloc[[0]]])
        assert DataValidator.check_duplicates(df, subset=["symbol"]) >= 1


class TestCheckPriceValidity:
    def test_valid_prices(self):
        df = _make_valid_data()
        count, issues = DataValidator.check_price_validity(df)
        assert count == 0
        assert issues == {}

    def test_negative_price(self):
        df = _make_valid_data(rows=5)
        df.loc[0, "close"] = -1.0
        count, issues = DataValidator.check_price_validity(df)
        assert count >= 1
        assert "close_negative" in issues

    def test_high_less_than_low(self):
        df = _make_valid_data(rows=5)
        df.loc[0, "high"] = 5.0
        df.loc[0, "low"] = 20.0
        count, issues = DataValidator.check_price_validity(df)
        assert count >= 1
        assert "high_lt_low" in issues

    def test_close_out_of_range(self):
        df = _make_valid_data(rows=5)
        # 设置 close 远高于 high（需要确保 low 也合理）
        df.loc[0, "high"] = 10.5
        df.loc[0, "low"] = 9.5
        df.loc[0, "close"] = 999.0  # 远高于 high
        count, issues = DataValidator.check_price_validity(df)
        assert count >= 1
        assert "close_out_of_range" in issues


class TestCheckVolumeValidity:
    def test_valid_volume(self):
        df = _make_valid_data()
        assert DataValidator.check_volume_validity(df) == 0

    def test_negative_volume(self):
        df = _make_valid_data(rows=5)
        df.loc[0, "volume"] = -100
        assert DataValidator.check_volume_validity(df) >= 1

    def test_no_volume_column(self):
        df = pd.DataFrame({"a": [1]})
        assert DataValidator.check_volume_validity(df) == 0


class TestValidate:
    def test_valid_data_passes(self):
        df = _make_valid_data()
        # 只检查 columns 和 price（跳过 duplicates，因为单 symbol 数据行数>1 会被标记）
        result = DataValidator().validate(df, checks=["columns", "price", "volume"])
        assert result["passed"] is True
        assert result["issues"] == {}

    def test_missing_columns_fails(self):
        df = pd.DataFrame({"a": [1]})
        result = DataValidator().validate(df, checks=["columns"])
        assert result["passed"] is False
        assert "missing_columns" in result["issues"]

    def test_selective_checks(self):
        df = pd.DataFrame({"a": [1]})  # 缺少必需列
        result = DataValidator().validate(df, checks=["missing"])
        # 只检查 missing，不检查 columns
        assert "missing_columns" not in result["issues"]


# ============================================================
# DataCleaner 测试
# ============================================================


class TestRemoveOutliers:
    def test_iqr_method(self):
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})
        cleaned = DataCleaner.remove_outliers(df, ["value"], method="iqr", threshold=1.5)
        assert len(cleaned) < len(df)
        assert 100 not in cleaned["value"].values

    def test_zscore_method(self):
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})
        cleaned = DataCleaner.remove_outliers(df, ["value"], method="zscore", threshold=2.0)
        assert len(cleaned) < len(df)

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        cleaned = DataCleaner.remove_outliers(df, ["nonexistent"])
        assert len(cleaned) == 3


class TestFillMissing:
    def test_ffill(self):
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        result = DataCleaner.fill_missing(df, method="ffill")
        assert result["a"].iloc[1] == 1

    def test_zero_fill(self):
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        result = DataCleaner.fill_missing(df, method="zero")
        assert result["a"].iloc[1] == 0

    def test_value_fill(self):
        df = pd.DataFrame({"a": [1, np.nan, 3]})
        result = DataCleaner.fill_missing(df, method="value", value=-1)
        assert result["a"].iloc[1] == -1


class TestFixPriceAnomalies:
    def test_swap_high_low(self):
        df = pd.DataFrame({"high": [5.0], "low": [20.0]})
        result = DataCleaner.fix_price_anomalies(df)
        assert result["high"].iloc[0] == 20.0
        assert result["low"].iloc[0] == 5.0

    def test_clamp_to_range(self):
        df = pd.DataFrame(
            {
                "open": [999.0],
                "close": [0.0],
                "high": [10.0],
                "low": [5.0],
            }
        )
        result = DataCleaner.fix_price_anomalies(df)
        assert result["open"].iloc[0] == 10.0
        assert result["close"].iloc[0] == 5.0


class TestNormalizeVolume:
    def test_log_method(self):
        df = pd.DataFrame({"volume": [100, 1000, 10000]})
        result = DataCleaner.normalize_volume(df, method="log")
        assert all(result["volume"] > 0)

    def test_rank_method(self):
        df = pd.DataFrame({"volume": [100, 1000, 10000]})
        result = DataCleaner.normalize_volume(df, method="rank")
        assert result["volume"].max() == 1.0

    def test_no_volume_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = DataCleaner.normalize_volume(df)
        assert "volume" not in result.columns


class TestClean:
    def test_full_pipeline(self):
        df = pd.DataFrame(
            {
                "symbol": ["A", "A", "B"],
                "close": [10.0, 10.0, 20.0],
                "volume": [100, 100, 200],
            }
        )
        cleaned = DataCleaner().clean(df, fix_prices=False)
        assert len(cleaned) == 2  # 去重


# ============================================================
# DataQualityReport 测试
# ============================================================


class TestDataQualityReport:
    def test_generate_report(self):
        df = _make_valid_data()
        report = DataQualityReport(df).generate()
        assert "数据质量报告" in report
        assert "通过" in report

    def test_report_with_issues(self):
        df = pd.DataFrame({"a": [1]})  # 缺少必需列
        report = DataQualityReport(df).generate()
        assert "失败" in report  # columns 检查失败
