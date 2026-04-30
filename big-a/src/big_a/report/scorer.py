"""Multi-model scoring engine for watchlist stocks.

Provides WatchlistScorer class that integrates Kronos and LightGBM models
to generate comprehensive scoring reports for watchlist stocks.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from typing import Any

from big_a.config import load_config, PROJECT_ROOT
from big_a.data.screener import load_watchlist
from big_a.qlib_config import init_qlib
from big_a.models.lightgbm_model import load_model, create_dataset, predict_to_dataframe


def _convert_market_units(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["close_raw"] = result["close"] / result["factor"].replace(0, np.nan)
    result["open_raw"] = result["open"] / result["factor"].replace(0, np.nan)
    result["high_raw"] = result["high"] / result["factor"].replace(0, np.nan)
    result["low_raw"] = result["low"] / result["factor"].replace(0, np.nan)

    result["volume_shares"] = result["volume"] * result["factor"] * 100
    result["amount_yuan"] = result["amount"] * 1000

    return result


class WatchlistScorer:
    """Multi-model scoring engine for watchlist stocks.

    Integrates Kronos (time series prediction) and LightGBM (factor model)
    to generate comprehensive scoring reports with portfolio simulation.

    Example::

        scorer = WatchlistScorer()
        results = scorer.run()
        # results contains watchlist, scores, trends, market_data, portfolio
    """

    def __init__(
        self,
        watchlist_path: str = "configs/watchlist.yaml",
        kronos_config_path: str = "configs/model/kronos.yaml",
        lightgbm_model_path: str = "output/lightgbm_model.pkl",
        lightgbm_data_config: str = "configs/data/handler_alpha158.yaml",
        lightgbm_model_config: str = "configs/model/lightgbm.yaml",
        lookback_days: int = 10,
        account: float = 1_000_000,
        skip_trend: bool = False,
        skip_lightgbm: bool = False,
        skip_kronos: bool = False,
        qualitative: bool = False,
        hf_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the scorer.

        Parameters
        ----------
        watchlist_path : str
            Path to watchlist YAML config.
        kronos_config_path : str
            Path to Kronos model config.
        lightgbm_model_path : str
            Path to saved LightGBM model.
        lightgbm_data_config : str
            Path to LightGBM data handler config.
        lightgbm_model_config : str
            Path to LightGBM model config.
        lookback_days : int
            Number of historical days to analyze.
        account : float
            Simulated portfolio capital.
        skip_trend : bool
            If True, skip Kronos rolling trend computation (much faster).
        skip_lightgbm : bool
            If True, skip LightGBM scoring (much faster).
        skip_kronos : bool
            If True, skip Kronos scoring (much faster).
        qualitative : bool
            If True, run hedge fund qualitative analysis.
        hf_config : dict or None
            Configuration dict for HedgeFundSignalGenerator.
        """
        self.watchlist_path = watchlist_path
        self.kronos_config_path = kronos_config_path
        self.lightgbm_model_path = lightgbm_model_path
        self.lightgbm_data_config = lightgbm_data_config
        self.lightgbm_model_config = lightgbm_model_config
        self.lookback_days = lookback_days
        self.account = account
        self.skip_trend = skip_trend
        self.skip_lightgbm = skip_lightgbm
        self.skip_kronos = skip_kronos
        self.qualitative = qualitative
        self.hf_config = hf_config or {}
        self._kronos_config: dict[str, Any] | None = None

    def load_watchlist(self) -> dict[str, str]:
        """Load watchlist from YAML config.

        Returns
        -------
        dict[str, str]
            Mapping of stock codes to names.
        """
        import yaml

        config = load_config(self.watchlist_path)
        watchlist = config.get("watchlist", {})

        if not isinstance(watchlist, dict):
            logger.warning("watchlist is not a dict in config: {}", self.watchlist_path)
            return {}

        logger.info("Loaded {} stocks from watchlist", len(watchlist))
        return watchlist

    def _load_kronos_config(self) -> dict[str, Any]:
        """Load Kronos config from YAML."""
        if self._kronos_config is None:
            config = load_config(self.kronos_config_path)
            self._kronos_config = config.get("kronos", {})
        return self._kronos_config

    def score_kronos(self, instruments: list[str]) -> pd.DataFrame:
        """Score stocks with Kronos model (current scores only).

        Parameters
        ----------
        instruments : list[str]
            List of Qlib instrument codes.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: instrument, score, score_pct, signal_date.
        """
        cfg = self._load_kronos_config()

        from big_a.models.kronos import KronosSignalGenerator
        gen = KronosSignalGenerator(
            tokenizer_id=cfg.get("tokenizer_id", "NeoQuasar/Kronos-Tokenizer-base"),
            model_id=cfg.get("model_id", "NeoQuasar/Kronos-base"),
            device=cfg.get("device", "cpu"),
            lookback=cfg.get("lookback", 90),
            pred_len=cfg.get("pred_len", 10),
            max_context=cfg.get("max_context", 512),
            signal_mode=cfg.get("signal_mode", "mean"),
        )

        gen.load_model(local_files_only=False)

        from qlib.data import D
        calendar = D.calendar(freq="day")
        end_date = calendar[-1]
        start_date = calendar[-(cfg.get("lookback", 90) + 10)]

        signals = gen.generate_signals(instruments, start_date, end_date)

        if signals.empty:
            logger.warning("No Kronos signals generated")
            return pd.DataFrame(columns=["instrument", "score", "score_pct", "signal_date"])

        # Convert to expected format
        result = signals.reset_index()
        result.columns = ["signal_date", "instrument", "score"]
        result["signal_date"] = pd.to_datetime(result["signal_date"])

        market_data = self.fetch_market_data(instruments, n_days=20)
        if not market_data.empty:
            last_date = market_data.index.get_level_values(0)[-1]
            last_prices = market_data.xs(last_date, level=0)["close"]
            last_close_map = last_prices.to_dict()
            result["last_close"] = result["instrument"].map(last_close_map)
            result["score_pct"] = (result["score"] / result["last_close"]) * 100
            result = result.drop(columns=["last_close"])
        else:
            result["score_pct"] = np.nan

        return result[["instrument", "score", "score_pct", "signal_date"]]

    def score_kronos_rolling(self, instruments: list[str], n_days: int = 10) -> pd.DataFrame:
        """Generate rolling Kronos scores for the last n_days.

        Parameters
        ----------
        instruments : list[str]
            List of Qlib instrument codes.
        n_days : int
            Number of days to generate rolling scores for.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: date, instrument, score, score_pct.
        """
        cfg = self._load_kronos_config()
        lookback = cfg.get("lookback", 90)

        from big_a.models.kronos import KronosSignalGenerator
        gen = KronosSignalGenerator(
            tokenizer_id=cfg.get("tokenizer_id", "NeoQuasar/Kronos-Tokenizer-base"),
            model_id=cfg.get("model_id", "NeoQuasar/Kronos-base"),
            device=cfg.get("device", "cpu"),
            lookback=lookback,
            pred_len=cfg.get("pred_len", 10),
            max_context=cfg.get("max_context", 512),
            signal_mode=cfg.get("signal_mode", "mean"),
        )

        gen.load_model(local_files_only=False)

        # Get trading calendar
        from qlib.data import D
        calendar = D.calendar(freq="day")

        # Get the last n_days trading dates
        target_dates = calendar[-n_days:] if len(calendar) >= n_days else calendar

        # Load data for all needed dates (lookback + n_days buffer)
        data_start_date = calendar[-(lookback + n_days + 5)] if len(calendar) >= (lookback + n_days + 5) else calendar[0]
        data_end_date = calendar[-1]

        data = gen.load_data(instruments, str(data_start_date.date()), str(data_end_date.date()))

        records = []

        for i, target_date in enumerate(target_dates):
            logger.debug("Kronos rolling: date {}/{} ({})", i + 1, len(target_dates), target_date.strftime("%Y-%m-%d"))
            # For each target date, we need lookback days ending at that date
            # The index in the data DataFrame depends on where this date falls

            for instrument in instruments:
                try:
                    # Get stock data
                    stock_df = data.xs(instrument, level=0)

                    # Find the index position of target_date
                    if target_date not in stock_df.index:
                        continue

                    target_idx = stock_df.index.get_loc(target_date)

                    # Get lookback window ending at target_date
                    start_idx = max(0, target_idx - lookback)
                    window_df = stock_df.iloc[start_idx:target_idx + 1]

                    if len(window_df) < lookback:
                        continue

                    # Run prediction
                    pred_df = gen.predict(window_df)
                    if pred_df is None or pred_df.empty:
                        continue

                    last_close = window_df["close"].iloc[-1]
                    pred_closes = pred_df["close"].values

                    if gen.signal_mode == "last":
                        score = float(pred_closes[-1] - last_close)
                    else:
                        score = float(np.mean(pred_closes) - last_close)

                    score_pct = (score / last_close) * 100

                    records.append({
                        "date": pd.Timestamp(target_date),
                        "instrument": instrument,
                        "score": score,
                        "score_pct": score_pct,
                    })
                except Exception as e:
                    logger.debug("Kronos rolling failed for {} on {}: {}", instrument, target_date, e)
                    continue

        if not records:
            logger.warning("No rolling Kronos scores generated")
            return pd.DataFrame(columns=["date", "instrument", "score", "score_pct"])

        result = pd.DataFrame(records)
        return result

    def score_lightgbm(self, instruments: list[str]) -> pd.DataFrame:
        """Score stocks with LightGBM model.

        Parameters
        ----------
        instruments : list[str]
            List of Qlib instrument codes.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: date, instrument, score.
        """
        # Load model
        model_path = PROJECT_ROOT / self.lightgbm_model_path
        if not model_path.exists():
            logger.error("LightGBM model not found at {}", model_path)
            return pd.DataFrame(columns=["date", "instrument", "score"])

        model_path = PROJECT_ROOT / self.lightgbm_model_path
        if not model_path.exists():
            logger.error("LightGBM model not found at {}", model_path)
            return pd.DataFrame(columns=["date", "instrument", "score"])

        model = load_model(model_path)

        config = load_config(self.lightgbm_model_config, self.lightgbm_data_config)

        handler_kwargs = config["dataset"]["kwargs"]["handler"]["kwargs"]
        handler_kwargs["instruments"] = instruments

        from qlib.data import D
        calendar = D.calendar(freq="day")
        latest_date = str(calendar[-1].date()) if len(calendar) > 0 else "2024-12-31"
        handler_kwargs["end_time"] = latest_date
        config["dataset"]["kwargs"]["segments"]["test"][1] = latest_date

        dataset = create_dataset(config)
        preds = predict_to_dataframe(model, dataset, segment="test")

        if preds.empty:
            logger.warning("No LightGBM predictions generated")
            return pd.DataFrame(columns=["date", "instrument", "score"])

        # Reset index to get date and instrument as columns
        result = preds.reset_index()
        result.columns = ["date", "instrument", "score"]

        # Filter to last n_days (reuse D and calendar from above)
        if len(calendar) >= self.lookback_days:
            cutoff_date = calendar[-self.lookback_days]
            result = result[result["date"] >= pd.Timestamp(cutoff_date)]

        return result

    def fetch_market_data(self, instruments: list[str], n_days: int = 10) -> pd.DataFrame:
        """Fetch recent OHLCV data for the report.

        Parameters
        ----------
        instruments : list[str]
            List of Qlib instrument codes.
        n_days : int
            Number of days to fetch.

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex (date, instrument) and columns:
            open, high, low, close, volume, amount, change_pct.
        """
        from qlib.data import D

        calendar = D.calendar(freq="day")

        if len(calendar) < n_days:
            n_days = len(calendar)

        start_date = calendar[-n_days]
        end_date = calendar[-1]

        fields = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$factor"]
        raw = D.features(
            instruments,
            fields=fields,
            start_time=str(start_date.date()),
            end_time=str(end_date.date()),
        )

        if raw.empty:
            logger.warning("No market data fetched")
            return pd.DataFrame()

        raw.columns = [c.lstrip("$") for c in raw.columns]

        raw_sorted = raw.swaplevel(0, 1).sort_index()
        raw_sorted["change_pct"] = raw_sorted.groupby(level=1)["close"].pct_change() * 100

        return raw_sorted

    def compute_portfolio(
        self,
        scores: pd.DataFrame,
        watchlist: dict[str, str],
        topk: int = 5,
        max_weight: float = 0.25,
    ) -> pd.DataFrame:
        """Simulate portfolio positions based on scores.

        Parameters
        ----------
        scores : pd.DataFrame
            DataFrame with columns: instrument, combined_score.
        watchlist : dict[str, str]
            Mapping of stock codes to names.
        topk : int
            Number of top stocks to select.
        max_weight : float
            Maximum weight per stock.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: instrument, name, weight, allocation, signal, entry_price.
        """
        if scores.empty:
            logger.warning("No scores provided for portfolio computation")
            return pd.DataFrame(columns=["instrument", "name", "weight", "allocation", "signal", "entry_price"])

        # Rank stocks by score (descending)
        ranked = scores.sort_values("combined_score", ascending=False).copy()

        # Select topk stocks
        n_select = min(topk, len(ranked))
        selected = ranked.head(n_select).copy()

        # Assign equal weight, capped at max_weight
        target_weight = 1.0 / n_select
        selected["weight"] = min(target_weight, max_weight)

        # Calculate allocation
        selected["allocation"] = selected["weight"] * self.account

        # Add stock names
        selected["name"] = selected["instrument"].map(watchlist)

        # Add signal description
        def get_signal(score: float) -> str:
            if score > 0.5:
                return "Strong Buy"
            elif score > 0:
                return "Buy"
            elif score > -0.5:
                return "Sell"
            else:
                return "Strong Sell"

        selected["signal"] = selected["combined_score"].apply(get_signal)

        # Get entry prices from latest market data
        market_data = self.fetch_market_data(selected["instrument"].tolist(), n_days=1)
        if not market_data.empty:
            last_date = market_data.index.get_level_values(0)[-1]
            last_prices = market_data.xs(last_date, level=0)["close"]
            last_factors = market_data.xs(last_date, level=0)["factor"]
            selected["entry_price"] = selected["instrument"].map(last_prices) / selected["instrument"].map(last_factors)
        else:
            selected["entry_price"] = np.nan

        # Add cash allocation
        cash_weight = max(0, 1 - selected["weight"].sum())
        if cash_weight > 0:
            cash_row = pd.DataFrame([{
                "instrument": "CASH",
                "name": "现金",
                "weight": cash_weight,
                "allocation": cash_weight * self.account,
                "signal": "Hold",
                "entry_price": 1.0,
                "combined_score": 0,
            }])
            selected = pd.concat([selected, cash_row], ignore_index=True)

        return selected[["instrument", "name", "weight", "allocation", "signal", "entry_price"]]

    def run(self) -> dict:
        """Run full scoring pipeline.

        Returns
        -------
        dict
            Dictionary with keys:
            - watchlist: dict[str, str]
            - kronos_scores: DataFrame
            - kronos_trend: DataFrame
            - lightgbm_scores: DataFrame
            - lightgbm_trend: DataFrame
            - market_data: DataFrame
            - portfolio: DataFrame
            - summary: dict
        """
        logger.info("Starting watchlist scoring pipeline")

        # Initialize Qlib
        init_qlib()

        # Load watchlist
        watchlist = self.load_watchlist()
        instruments = list(watchlist.keys())

        if not instruments:
            logger.error("No instruments in watchlist")
            return {
                "watchlist": {},
                "kronos_scores": pd.DataFrame(),
                "kronos_trend": pd.DataFrame(),
                "lightgbm_scores": pd.DataFrame(),
                "lightgbm_trend": pd.DataFrame(),
                "market_data": pd.DataFrame(),
                "portfolio": pd.DataFrame(),
                "summary": {},
            }

        logger.info("Scoring {} instruments with Kronos and LightGBM", len(instruments))

        # Limit OpenMP threads to 1 to avoid deadlock between LightGBM (C++ OpenMP)
        # and PyTorch (also OpenMP) thread pools. Both libraries use OpenMP for
        # multithreading, and without this setting they can conflict on macOS causing
        # a deadlock when LightGBM runs before Kronos (which imports torch at module level).
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # LightGBM must run before Kronos to avoid torch + pickle.load deadlock
        logger.info("正在运行 LightGBM 模型打分...")
        if self.skip_lightgbm:
            lightgbm_scores = pd.DataFrame()
            logger.info("Skipping LightGBM scoring (skip_lightgbm=True)")
        else:
            lightgbm_scores = self.score_lightgbm(instruments)

        logger.info("正在运行 Kronos 模型打分 (首次加载模型可能需要 30 秒以上)...")
        if self.skip_kronos:
            kronos_scores = pd.DataFrame()
            logger.info("Skipping Kronos scoring (skip_kronos=True)")
        else:
            kronos_scores = self.score_kronos(instruments)
        if self.skip_trend:
            kronos_trend = pd.DataFrame()
            logger.info("Skipping Kronos rolling trend (skip_trend=True)")
        else:
            logger.info("正在计算 Kronos 滚动趋势 (这可能需要几分钟)...")
            kronos_trend = self.score_kronos_rolling(instruments, n_days=self.lookback_days)

        if not lightgbm_scores.empty:
            from qlib.data import D
            calendar = D.calendar(freq="day")
            if len(calendar) >= self.lookback_days:
                cutoff_date = calendar[-self.lookback_days]
                lightgbm_trend = lightgbm_scores[lightgbm_scores["date"] >= pd.Timestamp(cutoff_date)].copy()
            else:
                lightgbm_trend = lightgbm_scores.copy()
        else:
            lightgbm_trend = pd.DataFrame()

        # Fetch market data
        market_data = self.fetch_market_data(instruments, n_days=self.lookback_days)

        # Compute combined scores
        # Get latest scores from each model
        if not kronos_scores.empty:
            latest_kronos = kronos_scores.groupby("instrument").last().reset_index()
        else:
            latest_kronos = pd.DataFrame(columns=["instrument", "score", "score_pct", "signal_date"])

        if not lightgbm_scores.empty:
            latest_lightgbm = lightgbm_scores.groupby("instrument").last().reset_index()
        else:
            latest_lightgbm = pd.DataFrame(columns=["date", "instrument", "score"])

        # Merge and normalize
        if not latest_kronos.empty and not latest_lightgbm.empty:
            combined = latest_kronos[["instrument", "score_pct"]].merge(
                latest_lightgbm[["instrument", "score"]],
                on="instrument",
                how="inner",
                suffixes=("_kronos", "_lgb"),
            )

            # Min-max normalize both to 0-1 range
            if not combined.empty:
                combined["kronos_norm"] = (combined["score_pct"] - combined["score_pct"].min()) / (
                    combined["score_pct"].max() - combined["score_pct"].min() + 1e-5
                )
                combined["lgb_norm"] = (combined["score"] - combined["score"].min()) / (
                    combined["score"].max() - combined["score"].min() + 1e-5
                )

                # Combined score = simple average
                combined["combined_score"] = (combined["kronos_norm"] + combined["lgb_norm"]) / 2

                # Map back to -1 to 1 range for interpretation
                combined["combined_score"] = combined["combined_score"] * 2 - 1
            else:
                combined["combined_score"] = 0
        elif not latest_kronos.empty:
            combined = latest_kronos[["instrument", "score_pct"]].copy()
            combined["score_lgb"] = 0
            combined["combined_score"] = np.nan
        elif not latest_lightgbm.empty:
            combined = latest_lightgbm[["instrument", "score"]].copy()
            combined["score_pct"] = 0
            combined["combined_score"] = np.nan
        else:
            combined = pd.DataFrame(columns=["instrument", "score_pct", "score_lgb", "combined_score"])

        # Compute portfolio
        if not combined.empty and "combined_score" in combined.columns:
            valid_scores = combined[combined["combined_score"].notna()].copy()
            portfolio = self.compute_portfolio(valid_scores, watchlist)
        else:
            portfolio = pd.DataFrame()

        # Compute summary
        if not combined.empty and "combined_score" in combined.columns:
            valid_scores = combined[combined["combined_score"].notna()]["combined_score"]
            summary = {
                "total_stocks": len(watchlist),
                "bullish_count": int((valid_scores > 0).sum()),
                "bearish_count": int((valid_scores <= 0).sum()),
                "avg_score": float(valid_scores.mean()) if len(valid_scores) > 0 else 0.0,
                "best_stock": combined.loc[valid_scores.idxmax(), "instrument"] if len(valid_scores) > 0 else None,
                "worst_stock": combined.loc[valid_scores.idxmin(), "instrument"] if len(valid_scores) > 0 else None,
            }
        else:
            summary = {
                "total_stocks": len(watchlist),
                "bullish_count": 0,
                "bearish_count": 0,
                "avg_score": 0.0,
                "best_stock": None,
                "worst_stock": None,
            }

        logger.info("Scoring pipeline complete")

        # Qualitative analysis (hedge fund)
        hedge_fund_analysis: dict = {}
        if self.qualitative:
            logger.info("正在进行 AI 定性分析 (多Agent工作流，请耐心等待)...")
            from big_a.models.hedge_fund import HedgeFundSignalGenerator

            from qlib.data import D
            calendar = D.calendar(freq="day")
            end_date = str(calendar[-1].date())
            start_date = str(calendar[-90].date()) if len(calendar) >= 90 else str(calendar[0].date())

            hf_gen = HedgeFundSignalGenerator(config=self.hf_config)
            hf_result = hf_gen.generate_signals(
                instruments=instruments,
                start_date=start_date,
                end_date=end_date,
                return_details=True,
            )
            hedge_fund_analysis = {
                "details": hf_result.get("details", {}),
                "signals": hf_result.get("signals", pd.DataFrame()),
            }

        return {
            "watchlist": watchlist,
            "kronos_scores": kronos_scores,
            "kronos_trend": kronos_trend,
            "lightgbm_scores": lightgbm_scores,
            "lightgbm_trend": lightgbm_trend,
            "market_data": market_data,
            "portfolio": portfolio,
            "summary": summary,
            "hedge_fund_analysis": hedge_fund_analysis,
        }
