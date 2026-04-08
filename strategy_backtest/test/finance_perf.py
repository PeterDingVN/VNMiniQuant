import numpy as np
import pandas as pd
from config import RISK_FREE_RATE, TRADE_PERIOD
import warnings
import functools

class FinanceTest:

    # ---------
    #   Validate and warning
    # ----------

    @staticmethod
    def is_array(input_):
        if isinstance(input_, pd.Series):
            return input.to_numpy()  # already 1D
        input_ = np.asarray(input_)       # handles list, tuple, ndarray, etc.

        return input_.ravel()   
    
    @staticmethod
    def input_warning(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name_ = func.__name__
            warnings.warn(
                f"The calculation of {name_} assumes input as 'Log_return', which means"
                f"you must translate your current input by doing np.log(close_t / close_t-1)"
            )
            return func(*args, **kwargs)
        return wrapper

        
    # ----------------------------------
    # This is metric used for MCPT TEST
    # -----------------------------------
    @ staticmethod
    def profit_factor(ret: pd.Series|np.ndarray):
        
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
    @staticmethod
    def sharpe_ratio(ret, risk_free_rate: float = RISK_FREE_RATE, freq: int = TRADE_PERIOD):
         # ensure shape of input
        returns = FinanceTest.is_array(ret)
        
        # Sharpe calculation
        excess_returns = returns - risk_free_rate / freq
        std = np.std(excess_returns)
        if std == 0:
            return np.inf
        return np.sqrt(freq) * np.mean(excess_returns) / std
    


    @ staticmethod
    @input_warning
    def _compute_equity_curve(ret):
        pass

    @staticmethod
    def total_return():
        pass

    @staticmethod
    def max_drawdown():
        pass

    @staticmethod
    def winrate(ret):

        # ensure shape of input
        returns = FinanceTest.is_array(ret)

        return np.mean(returns>0) # how much win over the period

        



        

