import numpy as np
import pandas as pd
from config import RISK_FREE_RATE, TRADE_PERIOD, AssetType
import warnings
import functools

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
        delta = np.where(_delta==0, 0, 1)

        # cost as a simple-return fraction of equity
        cost_simple = self.cost_rate * delta

        # convert to log format -> directly subtract from log_return
        # cost_log = np.log1p(-cost_simple)
        
        return cost_simple

class FinanceTest:

    # ---------
    #   Validate and warning
    # ----------

    @staticmethod
    def is_array(input_):
        if isinstance(input_, pd.Series):
            return input_.to_numpy()  # already 1D
        input_ = np.asarray(input_)       # handles list, tuple, ndarray, etc.

        return input_.ravel()   
    
    @staticmethod
    def input_warning(func):

        # Customize yellow warning text
        def yellow_warning(message, category, filename, lineno, file=None, line=None):
            print(
                f"\033[38;2;255;255;0m{category.__name__}: {message}\033[0m"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name_ = func.__name__

            warnings.showwarning = yellow_warning
            warnings.warn(
                f"The calculation of {name_} assumes input as 'Log_return', which means"
                f" you must translate your current input by doing np.log(close_t / close_t-1)"
            )
            
            return func(*args, **kwargs)
        return wrapper

    

    # ----------------------------------
    # This is metric used for MCPT TEST
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
    # Additional metrics
    # ------------------------

    # NORMAL RETURN
    @staticmethod
    def fixed_capital_fp(df_: pd.DataFrame,
                            asset_type: float = "future",
                            risk_free_annual=RISK_FREE_RATE,
                            trade_ped=TRADE_PERIOD):
        
        df = df_.reset_index().copy()

        if not all(c in df.columns for c in ['position', 'time', 'close']):
            raise KeyError("Provide your data with cols exactly named: position, time, close")
        
        df = df.set_index('time')

        # normalized pnl/equity curve

        if asset_type == "future":
            cost_rate = AssetType.cost_type['future']
            cost_arr = TransactionCost(cost_rate = cost_rate).transaction_cost_arr(signal=df['position']) 
            cost_point = cost_arr * df['close']
            gains_after_fee = df['close'].diff().fillna(0) * df['position'] - cost_point

        elif asset_type == 'stock':
            cost_rate = AssetType.cost_type['stock']
            cost_arr = TransactionCost(cost_rate = cost_rate).transaction_cost_arr(signal=df['position']) 
            cost_pct = cost_arr * df['close']
            gains_after_fee = df['close'].diff().fillna(0) * df['position'] - cost_pct*df['close']


        daily_gains = gains_after_fee.fillna(0).resample('D').sum(min_count=1).dropna(how='all')
        equity = daily_gains.cumsum()

        year_no = len(df.index.year.unique())

        max_annual_close = df.groupby(df.index.year)['close'].transform('max')
        max_annual_close = (
            max_annual_close
            .groupby(pd.Grouper(freq='D'))
            .last(min_count=1)
            .dropna()
        )

        pnl_curve = equity / year_no / max_annual_close
   

        # TOTAL RET - daily return (according to data TimeFrame)
        total_return = pnl_curve.iloc[-1] 

        # MAX DRAWDOWN
        peak = pnl_curve.cummax()
        dd = pnl_curve - peak
        mdd = dd.replace([np.inf, -np.inf], np.nan).min()

        # SHARPE
        avg_ret = daily_gains.mean()
        volatility = daily_gains.std()
        sharpe = (avg_ret - risk_free_annual) / volatility * np.sqrt(trade_ped)
        

        return {"total_return": total_return, "max_drawdown": mdd, "sharpe": sharpe}