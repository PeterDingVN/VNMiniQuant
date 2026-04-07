from datetime import datetime
from vnstock import Vnstock
from config.settings import DATA_SOURCE, START_DATE, INTERVAL, N_LAGS
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[0]


# =======================================================
# ============== Load Data From API source Class ========
# =======================================================
class LoadData:

    """
        This class is responsible for loading stock price data from API sources. 
        It relies on lib: vnstock
        It fetches data based on symbol, start date, interval, and count back (number of candles) parameters.

        Output: Dataframe with stock price data

        Params:
        - symbol: Stock name (default: 'VCI')
        - start: Earliest date for data (default: START_DATE from config)
        - interval: Interval for stock data (min, day, hr) (default: INTERVAL from config)
        - count_back: Number of candles to fetch (default: N_LAGS + 1 from config)
    """

    def __init__(self,
        symbol: str = 'VCI',
        start: str = START_DATE,
        interval: str = INTERVAL,
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
    

# =======================================================
# ============== Access Loaded Data Class ===============
# =======================================================
class AccessData(LoadData):

    """
        This class is responsible for accessing the fetched stock price data, either for training or prediction purposes.
        It is able to access already-crawled data if the data is up-to-date or used for training purpose.

        Output: Dataframe with stock price data for the specified purpose (train, retrain, pred)

        Params:
            - all params from LoadData class
            - purpose: The purpose for accessing data (train, retrain, pred)

    """ 


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_path = BASE_DIR / "stock_price_cache" / f'{self.symbol}.parquet'

    # ===== SubFunc: Check if the dataset already exist =====
    def check_path_existence(self) -> Path|None:

        if self.file_path.exists():
            return self.file_path
        else:
            return None

    # ===== Subfunc: If the data path is there, is it up to date? =====
    def data_is_updated(self, df: pd.DataFrame, date_col: str = 'time') -> bool:
        latest = pd.to_datetime(df[date_col]).max().date()
        today = datetime.now().date()
        delta = (today - latest).days
        return delta == 0 or (delta <= 2 and today.weekday() in (5, 6))

    # ===== MAIN FUNC: Access the data accordingly =====
    def access_data(self, purpose:
                    str = "train",
                    replace_old_data: bool = False) -> pd.DataFrame:

        # All purposes possible
        if purpose not in ("train", "retrain", "pred"):
            raise ValueError(f"purpose must be 'train', 'retrain', or 'pred', got '{purpose}'")

        # Check for parquet file -> if not -> fetch data and create the data file
        is_path = self.check_path_existence()
        if not is_path:
            data_ = self.fetch_stock_price_data()
            data_.to_parquet(self.file_path, index=False)
        data_ori = pd.read_parquet(self.file_path)


        # If file exist then see for purposes:
        # -> train: return ori_data
        # -> retrain: check if ori_data up-to-date and return new dataset if not -> overwrite ori train file is optional
        # -> pred: same as retrain (no overwrite) -> take only latest N_LAGS + 1 rows
        if replace_old_data and purpose!="retrain":
            print(
                """
                    replace_old_data is defaul as FALSE when purpose != "retrain"
                """
            )
        
        # ==== Retrain step ==== 
        if purpose == 'retrain':

            # Check for most updated data - retrain purpose
            if self.data_is_updated(data_ori)==False:
                data_new = self.fetch_stock_price_data()

                # Optional overwrite old original data from train purpose
                if replace_old_data:
                    latest = pd.to_datetime(data_ori['time']).max().date()
                    today = datetime.now().date()
                    print(f"Current {self.symbol} data is updated from {latest} to {today}")
                    data_new.to_parquet(self.file_path, index=False)
                return data_new
            
            # If data is already updated -> return ori_data
            else:
                return data_ori

        # ==== Pred ==== 
        elif purpose == 'pred':

            # Check for most updated data - prediction purpose
            if not self.data_is_updated(data_ori):
                data_pred_new = self.fetch_stock_price_data(for_train=False)
                return data_pred_new
            else:
                data_ori = data_ori[-self.candle_nums:]
                return data_ori

        # ==== Train ====
        return data_ori # return ori data in case purpose = "train"

# =======================================================
# ================== TEST CASE =========================
# =======================================================

# CMD: python -m data.stock_price.stock_price_access

if __name__ == '__main__':
    CTD = AccessData(symbol='AGR')
    data = CTD.access_data(purpose="train")
    print(data)
    print(len(data))









