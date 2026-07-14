from AlphaBase import AlphaBase
from TrainingEngine import TrainTA, TrainTestSplit
import numpy as np

import pandas as pd

BLUE = "\033[38;5;45m"
RESET = "\033[0m"

class AlphaCore(AlphaBase):

# CONFIG
    def __init__(self):
        super().__init__()

# Gen Data
    def generate_data(self, dt_name: str = None):
        if dt_name:
            return self.dm_list[dt_name] # # Double check how Tickers are called in Helper.py
        return self.dm_list
    
# TA
    def train_ta(self, data: pd.DataFrame, param_range: dict,
                 oos_ratio: float = 0.15,
                 w4w_val_ratio: float = 0.15,
                 w4w_gap: int = 0,
                 n_fold: int = 5,
                 n_trials: int = 110,
                 opt_dir: str = 'maximize',
                 opt_metric: str = 'sharpe'):
        train = TrainTA(oos_ratio = oos_ratio,
                 w4w_val_ratio = w4w_val_ratio,
                 w4w_gap = w4w_gap,
                 n_fold = n_fold,
                 n_trials = n_trials,
                 opt_dir = opt_dir,
                 opt_metric = opt_metric)
        train.start_training(data, param_range)

    def backtest_ta(self, data: pd.DataFrame, plot_pnl: bool = True):
        oos_size = TrainTA().__dict__['oos_ratio']
        train_df, test_df = TrainTestSplit(test_size=oos_size).split(data)

        for label, df in (("TRAINING RESULT", train_df), ("TEST RESULT", test_df)):
            print(' ')
            print(f"{BLUE}*****************  {label} *****************{RESET}")
            pos = self.alpha.run(df)
            df = df.copy()
            df.loc[:, 'position'] = np.asarray(pos)

            self.bt_fin.pnl_report(df, plot=plot_pnl)
            
        # them stat tets, future leak test, overfit test
        

# ML -> coming soon ...
    def backtest_ml(self):
        pass
    
    


# python -m ExeCore.exe
# check capital scale
if __name__ == '__main__':
    alpha = AlphaCore()
    data_list = alpha.generate_data()
    data = data_list['VCI_5m']

    # param_range = {
    # "don_lookback": (10, 100),
    # "ema_lookback": (10, 200),
    # "atr_lookback": (5, 100),
    # "long_atr_mult": (0.5, 7.0, 0.05),
    # "short_atr_mult": (0.5, 7.0, 0.05),
    # }
    # alpha.train_ta(data, param_range=param_range)

    alpha.backtest_ta(data, plot_pnl=True)
    
