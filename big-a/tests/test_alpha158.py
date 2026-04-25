"""Tests for Alpha158 feature handler."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.mark.skip_if_no_data
class TestAlpha158Handler:
    @pytest.fixture(autouse=True)
    def setup(self, qlib_initialized):
        from big_a.factors.handler import create_alpha158_dataset
        self.dataset = create_alpha158_dataset()

    def test_dataset_creates_successfully(self):
        from qlib.data.dataset import DatasetH

        assert isinstance(self.dataset, DatasetH)

    def test_feature_count_is_158(self):
        from big_a.factors.handler import get_train_data

        df = get_train_data(self.dataset)
        assert df.shape[1] == 158

    def test_no_date_overlap(self):
        from big_a.factors.handler import get_train_data, get_test_data

        train_df = get_train_data(self.dataset)
        test_df = get_test_data(self.dataset)
        train_max = train_df.index.get_level_values("datetime").max()
        test_min = test_df.index.get_level_values("datetime").min()
        assert train_max < test_min
