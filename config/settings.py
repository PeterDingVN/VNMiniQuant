from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
## Data Source Params
DATA_SOURCE = ["KBS", "VCI", "FMP", "TCBS"]
START_DATE = '2015-01-01'
INTERVAL = '1d'

# Portfolio (example)
PORTFOLIO = ['CTD', 'AGR', 'PDR', 'NHA', 'DIG']

# Number of lookback periods
N_LAGS = 11


# Finance Performance Test config
RISK_FREE_RATE = 0.64 # ---> US Treasury Bond Rate
TRADE_PERIOD = 252 # ----> total trade periods per year, subtracting holidays, weekends

# Number of permutation for Monte Carlos Simulation
NUM_OF_PERM = 1000