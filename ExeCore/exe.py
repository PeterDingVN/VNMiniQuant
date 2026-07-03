# Call AlphaBase:
# - Gen data framework
# - Call Alpha auto
# Option for train test




from AlphaBase import AlphaBase
import pandas as pd

class MyAlpha(AlphaBase):

# CONFIG
    def __init__(self):
        super().__init__()

    def generate_data(self, dt_name: str = None):
        if dt_name:
            return self.dm_list[dt_name] # # Double check how Tickers are called in Helper.py
        return self.dm_list
    
    def generate_pos(self, data):
        pos = self.alpha.run(data)
        return pos
    
    def backtest(self, data: pd.DataFrame, plot_pnl: bool = True):
        if plot_pnl:
            self.bt_fin.pnl_report(data, plot=True)
        self.bt_fin.pnl_report(data, plot=False)
        

        # them stat tets, future leak test, overfit test
    
    # Them ham train? live_trade? 


# python -m ExeCore.exe
if __name__ == '__main__':
    obj = MyAlpha()
    data_list = obj.generate_data()
    data = data_list['BSI_1d']

    pos = obj.generate_pos(data)
    data['position'] = pos

    obj.backtest(data, plot_pnl=True)
    
