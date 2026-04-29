import numpy as np
import pandas as pd
from config import RISK_FREE_RATE, TRADE_PERIOD, COST_RATE
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
        cost_log = np.log1p(-cost_simple)

        return cost_log

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

        
    # ---------------------------------
    #        HELPER
    #----------------------------------
    def _equity_curve_raw(ret: np.ndarray) -> np.ndarray:

        # Calculate raw return at each date since the first day
        returns = FinanceTest.is_array(ret)
        return np.exp(np.cumsum(returns))
    

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
    def total_return(ret) -> float:
        returns = FinanceTest.is_array(ret)

        returns = returns[~np.isnan(returns)]
        return float(np.exp(np.sum(returns)) - 1)
    
    @staticmethod
    @input_warning
    def annualized_return(ret, freq: int = TRADE_PERIOD) -> float:
        returns = FinanceTest.is_array(ret)

        returns = returns[~np.isnan(returns)]
        return float(np.exp(freq * np.mean(returns)) - 1)


    # VOLATILITY
    @staticmethod
    @input_warning
    def annualized_volatility(ret, freq: int = TRADE_PERIOD) -> float:
        returns = FinanceTest.is_array(ret)

        returns = returns[~np.isnan(returns)]
        return float(np.std(returns) * np.sqrt(freq))

    @staticmethod
    @input_warning
    def max_drawdown(ret) -> float:

        returns = FinanceTest.is_array(ret)
        returns = returns[~np.isnan(returns)]

        equity   = FinanceTest._equity_curve_raw(returns)
        peak     = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return float(drawdown.min())

    
    # RISK ADJUSTED RETURN
    @staticmethod
    @input_warning
    def sharpe_ratio(ret, risk_free_rate: float = RISK_FREE_RATE,
                     freq: int = TRADE_PERIOD) -> float:
        
        returns        = FinanceTest.is_array(ret)
        excess_returns = returns[~np.isnan(returns)] - risk_free_rate / freq

        std = np.std(excess_returns)
        if std == 0:
            return np.inf
        return float(np.sqrt(freq) * np.mean(excess_returns) / std)

    @staticmethod
    @input_warning
    def sortino_ratio(ret, risk_free_rate: float = RISK_FREE_RATE,
                      freq: int = TRADE_PERIOD) -> float:

        returns = FinanceTest.is_array(ret)
        returns = returns[~np.isnan(returns)]

        excess  = returns - risk_free_rate / freq
        downside = excess[excess < 0]
        if len(downside) == 0:
            return np.inf
        
        downside_std = np.sqrt(np.mean(downside ** 2))   # semi-deviation
        if downside_std == 0:
            return np.inf
        
        return float(np.sqrt(freq) * np.mean(excess) / downside_std)

    @staticmethod
    @input_warning
    def calmar_ratio(ret, freq: int = TRADE_PERIOD) -> float:

        returns = FinanceTest.is_array(ret)
        returns = returns[~np.isnan(returns)]

        ann_ret  = float(np.exp(freq * np.mean(returns)) - 1)

        equity   = FinanceTest._equity_curve_raw(returns)
        peak     = np.maximum.accumulate(equity)
        mdd      = abs(float(((equity - peak) / peak).min()))

        if mdd == 0:
            return np.inf
        return ann_ret / mdd


    # TRADE LEVEL
    @staticmethod
    @input_warning
    def winrate(ret) -> float:
        returns = FinanceTest.is_array(ret)
        return float(np.mean(returns > 0))

        



        

