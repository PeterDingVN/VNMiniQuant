from pathlib import Path
from dataclasses import dataclass

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dt src and Portfolio (example)
DATA_SOURCE = ["KBS", "VCI", "TCBS"]
PORTFOLIO = ['CTD', 'AGR', 'PDR', 'NHA', 'DIG', 'CTS']

# Data -> change bar_per_day according to Interval
START_DATE = '2015-01-01'
INTERVAL = '1d'
BAR_PER_DAY = 1

# Number of data needed to generate ONE VALID PREDICTION
N_LAGS = 11

# Finance Performance Test config
RISK_FREE_RATE = 0.0 # ---> US Treasury Bond Rate
TRADE_PERIOD = 252 # ----> total trade periods per year, subtracting holidays, weekends

@dataclass
class AssetType:
    cost_type = {
    'stock':0.0015, # -> splippage and trans cost: 0.15% plus 0.02% from slippage, no tax accounted
    'future': 0.044
    }


# Strategy 1: EMA crossover + MACD
@dataclass
class DonchianCfg:
    config = {
        "rl_period": 11,
        "cl_period": 43,
        "rl_smooth": 2,
        "cl_smooth": 35,
        "method": "linreg",
        "lookback": 64,
        "don_lb": 28,
        "basis_lb": 31,
        "long_basis": 1.47,
        "short_basis": -0.765,
        "trend_lb": 53,
        "atr_lb": 44,
        "bull_threshold": 0.005,
        "bear_threshold": 0.015,
        "smooth_lb": 17

    }
    

# Core System config
@dataclass
class SysConfig:
    oos_testsize: float = 0.15
    k_fold: int = 3
    w4w_testsize: float = 0.2
    gap: int = 0
    n_perm: int = 1   # Reduce to 1 perm for faster testing
    perm_start_index=28
    perm_end_index=1
    init_capital = 100