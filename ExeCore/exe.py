from AlphaBase import AlphaBase
from TrainingEngine import TrainTA, TrainTestSplit

import pandas as pd

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
    def train_ta(self, data: pd.DataFrame, param_range: dict):
        train = TrainTA()
        train.start_training(data, param_range)

    def backtest_ta(self, data: pd.DataFrame, plot_pnl: bool = True):
        oos_size = TrainTA().__dict__['oos_ratio']
        dt_list = TrainTestSplit(test_size=oos_size).split(data)

        for idx, df in enumerate(dt_list):
            if idx == 0:
                print("******  TRAINING RESULT ******")
            else:
                print("******  TEST RESULT ******")
            pos = self.alpha.run(df)
            df['position'] = pos
            
            if plot_pnl:
                self.bt_fin.pnl_report(df, plot=True)
            self.bt_fin.pnl_report(df, plot=False)
        # them stat tets, future leak test, overfit test
        

# ML -> coming soon ...
    def backtest_ml(self):
        pass
    
    


# python -m ExeCore.exe
# check capital scale
# check ticker call
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
    
