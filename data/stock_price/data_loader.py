from vnstock import Vnstock
from config.settings import *

class LoadData:

    def __init__(self,
        symbol: str = 'VCI',  # Stock name
        start: str = START_DATE,  # Earliest date for data
        interval: str = INTERVAL,       # Interval for data (1d as default)
        count_back: int = N_LAGS+1
                 ):

        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.candle_nums = count_back

    # Load data for training
    def fetch_stock_price_data(self, for_train: bool = True):

        # Loop thru sources
        for src in DATA_SOURCE:
            try:
                stock_ = Vnstock().stock(symbol=self.symbol, source=src)
                if stock_:

                    # Training data -> take everyhthing back till 2015
                    if for_train:
                        print(f"Successfully loaded data for training: {self.symbol} from {src}")
                        data_ = stock_.quote.history(start=self.start, interval=self.interval)

                    # For pred, take up to N_LAGS + 1 latest data
                    else:
                        print(f"Successfully loaded data for prediction: {self.symbol} from {src}")
                        data_ = stock_.quote.history(count_back=self.candle_nums, interval=self.interval)
                        if len(data_) != self.candle_nums:
                            dif = self.candle_nums - len(data_)
                            data_ = stock_.quote.history(count_back=self.candle_nums+dif, interval=self.interval)
                            data_ = data_[-self.candle_nums:]
                    return data_

            except ValueError or ModuleNotFoundError:
                continue
        return "No data source available. Retry later."



# Test Case
if __name__ == "__main__":
    data_loader = LoadData(symbol='DIG')
    df = data_loader.fetch_stock_price_data(for_train=False)
    print(df)
