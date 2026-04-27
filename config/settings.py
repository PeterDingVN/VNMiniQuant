from pathlib import Path
from dataclasses import dataclass

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Portfolio (example)
PORTFOLIO = ['CTD', 'AGR', 'PDR', 'NHA', 'DIG', 'CTS']
# Data Source Params
DATA_SOURCE = ["KBS", "VCI", "TCBS"]

# Data

START_DATE = '2015-01-01'
INTERVAL = '1d'



# Number of lookback periods
N_LAGS = 11

# Finance Performance Test config
RISK_FREE_RATE = 0.64 # ---> US Treasury Bond Rate
TRADE_PERIOD = 252 # ----> total trade periods per year, subtracting holidays, weekends


# TA indicator 
# Strategy 1: EMA x MACD
LONG_EMA = 233
SHORT_EMA = 55
SIGNAL = 50
EMA_START = 0  # Min start for short (fastlen) and long (slowlen) ema should, if set, be 233
SIGNAL_START= 0 # signal start should, if set, be 50

# Slipage and Fees
COST_RATE = 0.0017 # -> 0.15% plus 0.02% from slippage, no tax accounted

# Core System config
@dataclass
class SysConfig:
    oos_testsize: float = 0.15
    k_fold: int = 3
    w4w_testsize: float = 0.2
    gap: int = 0
    n_perm: int = 1000
    perm_start_index=0
    perm_end_index=1
