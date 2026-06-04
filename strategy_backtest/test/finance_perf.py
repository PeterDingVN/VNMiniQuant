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
        delta = sig.diff().abs().fillna(0)
        cost_simple = self.cost_rate * delta


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
        df_ = df_.copy()
        df_.columns = [c.lower() for c in df_.columns]

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
    
    
    # # -----------------------
    # # Reports
    # # ------------------------
    @staticmethod
    # def fixed_capital_fp(
    #         df_: pd.DataFrame,
    #         asset_type: str = "future",
    #         risk_free_annual: float = RISK_FREE_RATE,
    #         trade_period: int = TRADE_PERIOD
    #         ):
       

    #     # -------------------------
    #     # Validate data and set index
    #     # -------------------------
    #     df = df_.copy()

    #     required_cols = ["time", "close", "position"]
    #     if not all(c in df.columns for c in required_cols):
    #         raise KeyError(f"Missing columns. Required: {required_cols}")

    #     df["time"] = pd.to_datetime(df["time"], errors="coerce")

    #     df = (
    #         df.dropna(subset=["time"])
    #         .sort_values("time")
    #         .set_index("time")
    #     )

    #     df = df.dropna(subset=["close"])

    #     # -------------------------
    #     # Position and price movement
    #     # -------------------------
    #     pos_held = df["position"].shift(1).fillna(0.0)
    #     price_diff = df["close"].diff().fillna(0.0)

    #     # -------------------------
    #     # Transaction cost
    #     # -------------------------
    #     if asset_type == "future":

    #         cost_rate = AssetType.cost_type["future"]

    #         cost_arr = TransactionCost(
    #             cost_rate=cost_rate
    #         ).transaction_cost_arr(df["position"])

    #         gains_after_fee = (
    #             price_diff * pos_held
    #             - cost_arr
    #         )

    #     elif asset_type == "stock":

    #         cost_rate = AssetType.cost_type["stock"]

    #         cost_arr = TransactionCost(
    #             cost_rate=cost_rate
    #         ).transaction_cost_arr(df["position"])

    #         proportional_cost = (
    #             cost_arr * df["close"]
    #         )

    #         gains_after_fee = (
    #             price_diff * pos_held
    #             - proportional_cost
    #         )

    #     else:
    #         raise ValueError(
    #             "asset_type must be 'future' or 'stock'"
    #         )

    #     # -------------------------
    #     # DAILY PNL
    #     # -------------------------
    #     daily_pnl = (
    #         gains_after_fee
    #         .resample("D")
    #         .sum(min_count=1)
    #         .dropna()
    #     )

    #     # cumulative pnl curve
    #     pnl_curve = daily_pnl.cumsum()



    #     yearly_max_close = df["close"].cummax()

    #     yearly_max_close_daily = (
    #         yearly_max_close
    #         .resample("D")
    #         .last()
    #         .reindex(pnl_curve.index)
    #         .replace(0, np.nan)
    #         .ffill()
    #     )

    #     # normalize pnl curve by yearly max close

    #     equity_curve = (
    #         pnl_curve / yearly_max_close_daily
    #     ).replace([np.inf, -np.inf], np.nan).ffill()

    #     # -------------------------
    #     # 1. TOTAL RETURN
    #     # -------------------------
    #     n_years = max(len(daily_pnl) / trade_period, 1e-9)
    #     total_return = equity_curve.iloc[-1] / n_years

    #     # -------------------------
    #     # MAX DRAWDOWN
    #     # -------------------------
    #     rolling_peak = equity_curve.cummax()
    #     drawdown = equity_curve - rolling_peak
    #     max_drawdown = drawdown.min()


    #     # -------------------------
    #     # SHARPE
    #     # -------------------------
    #     rf_daily = (
    #         (1.0 + risk_free_annual)
    #         ** (1.0 / trade_period)
    #         - 1.0
    #     )

    #     excess_ret = (
    #         daily_pnl - rf_daily
    #     )

    #     vol = excess_ret.std(ddof=1)

    #     sharpe = (
    #         np.nan
    #         if (vol == 0 or np.isnan(vol))
    #         else (
    #             excess_ret.mean()
    #             / vol
    #             * np.sqrt(trade_period)
    #         )
    #     )

    #     return {
    #         "return_per_year": float(total_return * 100),
    #         "max_drawdown": float(max_drawdown * 100),
    #         "sharpe": float(sharpe),
    #     }



    # @staticmethod
    # def fixed_capital_fp(
    #             df_: pd.DataFrame,
    #             asset_type: str = "future",
    #             risk_free_annual: float = 0.02, # Example
    #             trade_period: int = 252
    #             ):
            
    #         # -------------------------
    #         # Validate data and set index
    #         # -------------------------
    #         df = df_.copy()
    #         required_cols = ["time", "close", "position"]
    #         if not all(c in df.columns for c in required_cols):
    #             raise KeyError(f"Missing columns. Required: {required_cols}")

    #         df["time"] = pd.to_datetime(df["time"], errors="coerce")
    #         df = (
    #             df.dropna(subset=["time"])
    #             .sort_values("time")
    #             .set_index("time")
    #         )
    #         df = df.dropna(subset=["close"])

    #         # -------------------------
    #         # 1. ESTABLISH STATIC CAPITAL
    #         # -------------------------
    #         # To "force" returns down and MDD to be more realistic, 
    #         # we use the starting price as the constant capital base.
    #         initial_capital = df["close"].iloc[0]

    #         # -------------------------
    #         # Position and price movement
    #         # -------------------------
    #         pos_held = df["position"].shift(1).fillna(0.0)
    #         price_diff = df["close"].diff().fillna(0.0)

    #         # -------------------------
    #         # Transaction cost
    #         # -------------------------
    #         # (Assuming AssetType and TransactionCost classes are defined elsewhere)
    #         if asset_type == "future":
    #             cost_rate = AssetType.cost_type["future"]
    #             cost_arr = TransactionCost(cost_rate=cost_rate).transaction_cost_arr(df["position"])
    #             gains_after_fee = (price_diff * pos_held) - cost_arr
    #         elif asset_type == "stock":
    #             cost_rate = AssetType.cost_type["stock"]
    #             cost_arr = TransactionCost(cost_rate=cost_rate).transaction_cost_arr(df["position"])
    #             proportional_cost = cost_arr * df["close"]
    #             gains_after_fee = (price_diff * pos_held) - proportional_cost
    #         else:
    #             raise ValueError("asset_type must be 'future' or 'stock'")

    #         # -------------------------
    #         # DAILY PNL (Percentage)
    #         # -------------------------
    #         # We convert dollar gains to percentage returns relative to STATIC capital
    #         # This is the crucial step to fix Sharpe and Return scale.
    #         daily_pnl_dollars = (
    #             gains_after_fee
    #             .resample("D")
    #             .sum(min_count=1)
    #             .dropna()
    #         )
            
    #         # Convert to percentage returns
    #         daily_ret_pct = daily_pnl_dollars / initial_capital

    #         # Cumulative equity curve (Arithmetic for fixed capital)
    #         equity_curve = daily_pnl_dollars.cumsum() / initial_capital

    #         # -------------------------
    #         # 1. TOTAL RETURN (Annualized)
    #         # -------------------------
    #         n_years = max(len(daily_pnl_dollars) / trade_period, 1e-9)
    #         total_return = equity_curve.iloc[-1] / n_years

    #         # -------------------------
    #         # 2. MAX DRAWDOWN
    #         # -------------------------
    #         # MDD must be calculated on the cumulative percentage curve
    #         rolling_peak = equity_curve.cummax()
    #         drawdown = equity_curve - rolling_peak
    #         max_drawdown = drawdown.min()

    #         # -------------------------
    #         # 3. SHARPE (Unit Corrected)
    #         # -------------------------
    #         # Original code subtracted a % rate from a $ gain. 
    #         # Now both are in percentage decimals.
    #         rf_daily = (1.0 + risk_free_annual) ** (1.0 / trade_period) - 1.0
            
    #         excess_ret = daily_ret_pct - rf_daily
    #         vol = excess_ret.std(ddof=1)

    #         sharpe = (
    #             np.nan
    #             if (vol == 0 or np.isnan(vol))
    #             else (excess_ret.mean() / vol * np.sqrt(trade_period))
    #         )

    #         return {
    #             "return_per_year": float(total_return * 100),
    #             "max_drawdown": float(max_drawdown * 100),
    #             "sharpe": float(sharpe),
    #         }
    


    @staticmethod
    def fixed_capital_fp(
            df_: pd.DataFrame,
            asset_type: str = "future",
            risk_free_annual: float = RISK_FREE_RATE,
            trade_period: int = 252
            ):
        
        # -------------------------
        # 1. Prepare Data
        # -------------------------
        df = df_.copy()
        df.columns = [c.lower() for c in df.columns]
        if 'date' not in df.columns:
            df = df.rename(columns={'datetime': 'date'})

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date").set_index("date")

        # -------------------------
        # 2. Define Static Capital (Harmonizer)
        # -------------------------
        # We use the maximum close price as the 'cash_max' for the entire backtest.
        # This is the most conservative 'Fixed Capital' approach for 1 unit.
        cash_max = df["close"].max() 

        # -------------------------
        # 3. Calculate PnL (Dollar Gains)
        # -------------------------
        pos_held = df["position"].shift(1).fillna(0.0)
        price_diff = df["close"].diff().fillna(0.0)

        if asset_type == "future":
            cost_rate = AssetType.cost_type["future"]
            cost_arr = TransactionCost(cost_rate=cost_rate).transaction_cost_arr(df["position"])
            gains_after_fee = (price_diff * pos_held) - cost_arr
        else:
            cost_rate = AssetType.cost_type["stock"]
            cost_arr = TransactionCost(cost_rate=cost_rate).transaction_cost_arr(df["position"])
            gains_after_fee = (price_diff * pos_held) - (cost_arr * df["close"])

        # Resample to Daily (D) to get the daily PnL stream
        daily_pnl_dollars = gains_after_fee.resample("D").sum()
        pnl_curve_dollars = daily_pnl_dollars.cumsum()

        # -------------------------
        # 4. Correct Time Scaling (Forces Return Down)
        # -------------------------
        # Using calendar days for n_years is more accurate for "D" resampled data
        total_days = (df.index[-1] - df.index[0]).days
        n_years = max(total_days / 365, 1e-9)

        # -------------------------
        # 5. Harmonized Metrics
        # -------------------------
        # Return per year: (Total Dollar PnL / Capital) / Years
        total_return_pct = (pnl_curve_dollars.iloc[-1] / cash_max) / n_years

        # Max Drawdown: (Max Dollar DD / Capital)
        # Using your formula: max(pnl.cummax - pnl) / cash_max
        dollar_drawdown = pnl_curve_dollars.cummax() - pnl_curve_dollars
        max_drawdown_pct = dollar_drawdown.max() / cash_max

        # -------------------------
        # 6. Sharpe (Percentage based)
        # -------------------------
        daily_ret_pct = daily_pnl_dollars / cash_max
        rf_daily = (1.0 + risk_free_annual) ** (1.0 / trade_period) - 1.0
        
        excess_ret = daily_ret_pct - rf_daily
        vol = daily_ret_pct.std(ddof=1)

        sharpe = (
            np.nan if (vol == 0 or np.isnan(vol))
            else (excess_ret.mean() / vol * np.sqrt(trade_period))
        )

        return {
            "return_per_year": float(total_return_pct * 100),
            "max_drawdown": float(-max_drawdown_pct * 100), # Return as negative
            "sharpe": float(sharpe),
        }
    
# cmd: python -m strategy_backtest.test.finance_perf.py
if __name__ == '__main__':
    df = pd.read_csv(
        r'C:\Users\HP\.0_PycharmProjects\VNMiniQuant_Futures\data\cached_data\stock_price_cache\DCL_pos_mcty.csv')
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.columns = [c.lower() for c in df.columns]
    res = FinanceTest.fixed_capital_fp(df_=df)
    print(res)    

    
    
