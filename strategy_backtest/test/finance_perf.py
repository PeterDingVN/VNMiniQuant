import numpy as np
import pandas as pd
from config import RISK_FREE_RATE, TRADE_PERIOD, COST_RATE, BAR_PER_DAY
import warnings
import functools

class TransactionCost:
    """
        Add transaction cost into return to better reflect reality
        The cost is used row-wise and have yet to be cumsum together
        
        Use:
        log_ret + fees = real_ret => real_ret * signal = strat_ret => cumsum = total ret
    """

    def __init__(self, cost_rate: float = COST_RATE):
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
    def profit_factor(ret: pd.Series|np.ndarray) -> float:
        
        # ensure shape of input
        returns = FinanceTest.is_array(ret)

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
    @input_warning
    def fixed_capital_fp(df_: pd.DataFrame,
                            point_ret_col="point_ret",
                            risk_free_annual=RISK_FREE_RATE,
                            daily_bars=BAR_PER_DAY):
        
        df = df_.copy()
        if 'point_ret' not in df.columns:
            raise KeyError("Need to provide point_ret, find its calculation in StrategyLaunch.py")

        gains = df[point_ret_col].fillna(0.0).astype(float)

        equity = gains.cumsum()
        prev_eq = equity.shift(1)

        # portfolio returns (this fixes the "reset denominator" issue)
        ret = (gains / prev_eq).fillna(0)
        ret = np.where(np.isinf(ret), 0, ret)

        
        # TOTAL RET - according to TimeFrame
        year_no = len(df['time'].dt.year.unique())
        df['year'] = df['time'].dt.year
        max_annual_close = df.groupby('year')['close'].transform('max')
        total_return = (equity / year_no / max_annual_close).iloc[-1] 

        # MAX DRAWDOWN
        peak = equity.cummax()
        dd = equity / peak - 1.0
        mdd = dd.min()

        # SHARPE
        rf_per_bar = risk_free_annual / (daily_bars*252)
        excess = ret - rf_per_bar
        std = excess.std(ddof=1)
        sharpe = np.nan if (std is None or std == 0 or np.isnan(std)) \
                        else (np.sqrt(daily_bars*252) * excess.mean() / std)

        # In case we need PLOT 
        # out = df.copy()
        # out["equity"] = equity
        # out["ret"] = ret
        # out["drawdown"] = dd

        return {"total_return": total_return, "max_drawdown": mdd, "sharpe": sharpe}

        



        

