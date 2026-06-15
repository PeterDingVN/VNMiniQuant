import numpy as np
import pandas as pd
from config import RISK_FREE_RATE, TRADE_PERIOD, AssetType

class TransactionCost:
    """
        Add transaction cost into return to better reflect reality
        The cost is used row-wise and have yet to be cumsum together
        
        Use:
        log_ret + fees = real_ret => real_ret * signal = strat_ret => cumsum = total ret
    """

    def __init__(self, cost_rate: float):
        self.cost_rate = cost_rate

    def transaction_cost_arr(self, signal: pd.Series) -> pd.Series:

        sig = signal.fillna(0)
        _delta = sig.diff().abs().fillna(0)
        delta = np.where(
            _delta==0, 0,
            np.where(abs(_delta)==1, 1, 2
            )
        )

        # cost as a simple-return fraction of equity
        cost_simple = self.cost_rate * delta

        # convert to log format -> directly subtract from log_return
        # cost_log = np.log1p(-cost_simple)
        
        return cost_simple

class FinanceTest:

    # ---------
    #   Validate input
    # ----------
    @staticmethod
    def is_array(input_):
        if isinstance(input_, pd.Series):
            return input_.to_numpy()  # already 1D
        input_ = np.asarray(input_)       # handles list, tuple, ndarray, etc.

        return input_.ravel()   
    

    # ----------------------------------
    # This is metric used for MCPT TEST =>> NEED FIX
    # -----------------------------------
    @ staticmethod
    def profit_factor(df_: pd.DataFrame, pos_col: str) -> float:

        if pos_col not in df_:
            raise KeyError("Return to your strategy and add position column into data")
        
        log_close = np.log(df_['close']/df_['close'].shift(1))
        returns = log_close*df_[pos_col]
        
        # Profit factor cal
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()

        if losses == 0:
            return np.inf
        return gains / losses
    
    
    # -----------------------
    # Reports
    # ------------------------
    @staticmethod
    def fixed_capital_fp(
            df_: pd.DataFrame,
            asset_type: str = "future",
            risk_free_annual: float = RISK_FREE_RATE,
            trade_period: int = TRADE_PERIOD,
            ):
        """
            Required columns:
            ['time', 'close', 'position']

        """
        
        # Validate data and Set index
        df = df_.copy()
        required_cols = ["time", "close", "position"]
        if not all(c in df.columns for c in required_cols):
            raise KeyError(f"Missing columns. Required: {required_cols}")

        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = (
            df.dropna(subset=["time"])
            .sort_values("time")
            .set_index("time")
        )
        df = df.dropna(subset=["close"])


        # Position and Daily Absolute return
        pos_held = df["position"].shift(1).fillna(0.0)
        price_diff = df["close"].diff().fillna(0.0)


        # Cost calculation
        if asset_type == "future":
            cost_rate = AssetType.cost_type["future"]
            cost_arr = TransactionCost(cost_rate=cost_rate).transaction_cost_arr(df["position"])
            gains_after_fee = price_diff * pos_held - cost_arr

        elif asset_type == "stock":
            cost_rate = AssetType.cost_type["stock"]
            cost_arr = TransactionCost(cost_rate=cost_rate).transaction_cost_arr(df["position"])
            proportional_cost = cost_arr * df["close"]
            gains_after_fee = price_diff * pos_held - proportional_cost

        else:
            raise ValueError("asset_type must be 'future' or 'stock'")

        
        # Daily PnL
        daily_gains = (
            gains_after_fee
            .resample("D")
            .sum(min_count=1)
            .dropna()
        )

        yearly_max_close = df["close"].groupby(df.index.year).transform("max")

        yearly_max_close_daily = (
            yearly_max_close
            .resample("D")
            .last()
            .reindex(daily_gains.index)
            .replace(0, np.nan)
            .ffill()
        )

        daily_ret = (
            daily_gains / yearly_max_close_daily
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        equity_curve = 1.0 + daily_ret.cumsum()


        # 1. TOTAL RETURN
        n_years = max(len(daily_ret) / trade_period, 1e-9)

        total_return = (
            equity_curve.iloc[-1] - 1.0
        ) / n_years

        
        # 2. MDD
        rolling_peak = equity_curve.cummax()
        drawdown = (
            equity_curve / rolling_peak
        ) - 1.0
        max_drawdown = drawdown.min()


        # 3. SHARPE
        rf_daily = (
            (1.0 + risk_free_annual)
            ** (1.0 / trade_period)
            - 1.0
        )

        excess_ret = daily_ret - rf_daily
        vol = excess_ret.std(ddof=1)

        sharpe = (
            np.nan
            if (vol == 0 or np.isnan(vol))
            else (
                excess_ret.mean()/ vol * np.sqrt(trade_period)
            )
        )


        return {
            "return_per_year": float(total_return * 100),
            "max_drawdown": float(max_drawdown * 100),
            "sharpe": float(sharpe)}