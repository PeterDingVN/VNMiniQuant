from __future__ import annotations

import sys
import warnings

from typing import Any, Optional, Dict
import numpy as np
import pandas as pd

from .finance_backtest import FinanceMetrics, ZeroPosError

warnings.filterwarnings('ignore')

class TaStatTest:

    def __init__(self, alpha_type: str = "ta", config: Optional[dict] = None):
        self.alpha_type = alpha_type
        self.config = config or {}
        self.alpha = None
        self.bt_fin = None
        self.result = {}

    def set_context(self, alpha=None, bt_fin=None, config: Optional[dict] = None):
        self.alpha = alpha
        self.bt_fin = bt_fin
        if config is not None:
            self.config = config
        return self

    @staticmethod
    def _safe_corr(a: pd.Series, b: pd.Series) -> float:
        x = pd.concat([a, b], axis=1).dropna()
        if len(x) < 3:
            return np.nan
        if x.iloc[:, 0].nunique() <= 1 or x.iloc[:, 1].nunique() <= 1:
            return np.nan
        return float(x.iloc[:, 0].corr(x.iloc[:, 1]))

    @staticmethod
    def _to_series(x, index, name="position") -> pd.Series:
        if isinstance(x, pd.Series):
            s = x.copy().reindex(index)
            s.name = name
            return s
        return pd.Series(np.asarray(x), index=index, name=name)

    def _warmup_start(self) -> int:     
        params = self.config['alpha_cfg']['params']
        vals = [abs(v) for v in params.values() if isinstance(v, (int, float))]
        if not vals:
            return 1
        return int(max(vals)) + 2

    def _shift_input(self, df: pd.DataFrame, shift_bars: int = 1) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col != "datetime":
                df[col] = df[col].shift(shift_bars)
        return df

    def _metric_eval(self, df: pd.DataFrame) -> Dict[str, Any]:
        bt_cfg = self.config['bt_cfg']
        metrics = FinanceMetrics(df=df, **bt_cfg)

        sharpe, sortino = metrics.Sharpe_after_fee()
        mdd_val = metrics.MDD()[1]
        tot_ret, ret_per_year, cagr = metrics.Return()
        total_profit, profit_after_fee_per_year = metrics.Profit()

        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "mdd": float(mdd_val),
            "total_return": float(tot_ret),
            "ret_per_year": float(ret_per_year),
            "cagr": float(cagr),
            "total_profit": float(total_profit),
            "profit_after_fee_per_year": float(profit_after_fee_per_year),
        }

    def future_leak(self, data: pd.DataFrame, min_perf_drop: float = 0.2) -> Dict[str, Any]:
        sys.stdout.write("\033[35mChecking future leak ...\033[0m")
        sys.stdout.flush()

        n_iter = 60
        drop_count = 0
        diff_count = 0

        
        full_df = data.copy()
        full_pos = self.alpha.run(full_df)
        full_df["position"] = full_pos.values


        n_total = len(data)
        start_warmup = self._warmup_start()
        min_chunk_len = int(max(n_total * 0.2, start_warmup))

        iterations_log = []
        iter_i = 1

        while iter_i <= n_iter + 1:
            chunk_len = np.random.randint(min_chunk_len, n_total + 1)
            chunk = data.iloc[0 : chunk_len]

            # Baseline run on chunk
            chunk_df = chunk.reindex()
            pos_chunk = self.alpha.run(chunk_df)
            chunk_df["position"] = pos_chunk.values


            try:
                base_eval = self._metric_eval(chunk_df)
            except ZeroPosError:
                continue

            # Shifted run on chunk (shifted by 1 bar)
            chunk_shifted = self._shift_input(chunk_df, shift_bars=1)
            pos_shifted = self.alpha.run(chunk_shifted)
            chunk_shifted["position"] = pos_shifted.values

            try:
                shifted_eval = self._metric_eval(chunk_shifted)
            except ZeroPosError:
                continue

            # Measure performance degradation
            base_sharpe = base_eval.get("sharpe", np.nan)
            shifted_sharpe = shifted_eval.get("sharpe", np.nan)
            base_ret = base_eval.get("total_return", np.nan)
            shifted_ret = shifted_eval.get("total_return", np.nan)

            sharpe_drop = (base_sharpe - shifted_sharpe) / max(abs(base_sharpe), 1e-12) if np.isfinite(base_sharpe) and np.isfinite(shifted_sharpe) else 0.0
            ret_drop = (base_ret - shifted_ret) / max(abs(base_ret), 1e-12) if np.isfinite(base_ret) and np.isfinite(shifted_ret) else 0.0

            is_drop = (sharpe_drop > min_perf_drop) or (ret_drop > min_perf_drop)
            if is_drop:
                drop_count += 1

            # Compare chunk positions against full data positions using exact datetime reference

            orig_pos_chunk = full_df.set_index("datetime")["position"].reindex(chunk["datetime"])
            curr_pos_chunk = pos_chunk

            has_diff = not np.allclose(
                curr_pos_chunk,
                orig_pos_chunk,
                equal_nan=True,
                atol=1e-6
            )

            if has_diff:
                diff_count += 1

            iterations_log.append({
                "iter": iter_i,
                "is_drop": is_drop,
                "has_diff": has_diff,
                "sharpe_drop": float(sharpe_drop),
                "ret_drop": float(ret_drop),
            })

            iter_i += 1

        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

        # Step 3: Print final colored status line
        if drop_count > 9 or diff_count >= 1:
            sys.stdout.write("\r\033[K\033[31mFail future leak test.\033[0m")
            sys.stdout.flush()
        else:
            sys.stdout.write("\r\033[K\033[32mPass future leak test.\033[0m")
            sys.stdout.flush()
            

    def overfit(self, *args, **kwargs):
        pass


    # ----- MAIN FUNC ------
    def stat_check(self, data: pd.DataFrame):
        self.future_leak(data=data)
        self.overfit()
        