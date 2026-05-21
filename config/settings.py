from pathlib import Path
from dataclasses import dataclass

# ================================  STRATEGY SETTINGS  ==================================
# Change this according to your strategy
"""
Syntax is:

config = {
    "para 1": value,
    ...
        }

"""

# Strategy 1: EMA crossover + MACD
@dataclass
class DonchianCfg:
    config = {
        "don_lookback": 12,
        "ema_lookback": 20,
        "atr_lookback": 10,
        "long_atr_mult": 2,
        "short_atr_mult": 0.6
        }




# ================================  BASE SETTINGS  ======================================
# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Dt src and Portfolio (example)
DATA_SOURCE = ["KBS", "VCI", "TCBS"]


# Data Timeframe and settings
START_DATE = '2015-01-01'
INTERVAL = '1d'
BAR_PER_DAY = 1
RISK_FREE_RATE = 0.0 # ---> US Treasury Bond Rate
TRADE_PERIOD = 252 # ----> total trade periods per year, subtracting holidays, weekends


# Choose stock or future depends on Asset Type
@dataclass
class AssetType:
    cost_type = {
    'stock':0.0015, # -> splippage and trans cost: 0.15% plus 0.02% from slippage, no tax accounted
    'future': 0.04
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
    


