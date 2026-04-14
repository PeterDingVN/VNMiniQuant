from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from vnstock import Vnstock

from config import DATA_SOURCE, START_DATE, INTERVAL, N_LAGS, PROJECT_ROOT, PORTFOLIO



# =======================================================
# ============== Load Data From API source Class ========
# =======================================================

class _FetchData:

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
                 source: list[str] = DATA_SOURCE):
        self.source = source


    # Load data for training
    def fetch_stock_price_data(self, symbol: str, 
                                      interval: str, 
                                      start: str, 
                                      source: list[str],
                                      countback: int=None, 
                                      for_train: bool = True):

        # Loop thru sources
        for src in source:
            try:
                stock_ = Vnstock().stock(symbol=symbol, source=src)
                if stock_:

                    # Training data -> take everyhthing back till 2015
                    if for_train:
                        print(f"Successfully loaded data for training: {symbol} from {src}")
                        data_ = stock_.quote.history(start=start, interval=interval)

                    # For pred, take up to N_LAGS + 1 latest data
                    else:
                        print(f"Successfully loaded data for prediction: {symbol} from {src}")
                        data_ = stock_.quote.history(count_back=countback, interval=interval)

                        # countback looks for no of days (including days off) 
                        # BUT WE NEED exact no of candles not days
                        # the if func helps bridge the difference in definitions/
                        if len(data_) != countback:
                            dif = countback - len(data_)
                            data_ = stock_.quote.history(count_back=countback+dif, interval=interval)
                            data_ = data_[-countback:]
                            
                    return data_

            except ValueError or ModuleNotFoundError:
                continue
        return "No data source available. Retry later."
    

# =======================================================
# ============== Access Loaded Data Class ===============
# =======================================================
class AccessSingleData(_FetchData):

    """
        This class is responsible for accessing the fetched stock price data, either for training or prediction purposes.
        It is able to access already-crawled data if the data is up-to-date or used for training purpose.

        Output: Dataframe with stock price data for the specified purpose (train, retrain, pred)

        Params:
            - all params from LoadData class
            - purpose: The purpose for accessing data (train, retrain, pred)

    """ 


    def __init__(self,
                 symbol: str,
                 cache_root: str = PROJECT_ROOT,  # Not recommend changing this
                 start: str = START_DATE,
                 interval: str = INTERVAL,
                 countback: int = N_LAGS+1,  
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.symbol = symbol
        self.start = start
        self.interval = interval
        self.candle_nums = countback
        self.folder_path = cache_root/ "cached_data" / "stock_price_cache"

        if not self.folder_path.exists():
            self.folder_path.mkdir(parents=True, exist_ok=True) 

        self.file_path = self.folder_path / f'{self.symbol}.parquet'

    # ===== SubFunc: Check if the dataset already exist =====
    def _check_path_existence(self) -> Path|None:

        if self.file_path.exists():
            return self.file_path
        else:
            return None

    # ===== Subfunc: If the data path is there, is it up to date? =====
    def _data_is_updated(self, df: pd.DataFrame, date_col: str = 'time') -> bool:
        latest = pd.to_datetime(df[date_col]).max().date()
        today = datetime.now().date()
        delta = (today - latest).days
        return delta == 0 or (delta <= 2 and today.weekday() in (5, 6))

    # ===== MAIN FUNC: Access the data accordingly =====
    def access_one_data(self, purpose: str = "train", replace_old_data: bool = False) -> pd.DataFrame:

        # All purposes possible
        if purpose not in ("train", "retrain", "pred"):
            raise ValueError(f"purpose must be 'train', 'retrain', or 'pred', got '{purpose}'")

        # Check for parquet file -> if not -> fetch data and create the data file
        is_path = self._check_path_existence()
        if not is_path:
            data_ = self.fetch_stock_price_data(source=self.source, symbol=self.symbol, 
                                                 interval=self.interval, start=self.start)
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
            if self._data_is_updated(data_ori)==False:
                data_new = self.fetch_stock_price_data(source=self.source, symbol=self.symbol, 
                                                        interval=self.interval, start=self.start)

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
            if not self._data_is_updated(data_ori):
                data_pred_new = self.fetch_stock_price_data(source=self.source, symbol=self.symbol, 
                                                             interval=self.interval, start=self.start,
                                                             countback=self.candle_nums, for_train=False)
                return data_pred_new
            else:
                data_ori = data_ori[-self.candle_nums:]
                return data_ori

        # ==== Train ====
        return data_ori # return ori data in case purpose = "train"


class AccessData:

    def __init__(self, symbol: list[str],
                       cache_root: str = PROJECT_ROOT,  # Not recommend changing this
                       start: str = START_DATE,
                       interval: str = INTERVAL,
                       countback: int = N_LAGS+1):
        self.symbols = symbol
        self.cache_root = cache_root
        self.start = start
        self.interval = interval
        self.countback = countback
        
    def _make_single_accessor(self, symbol: str) -> AccessSingleData:
        return AccessSingleData(
            symbol = symbol,
            cache_root=self.cache_root,
            start=self.start,
            interval=self.interval,
            countback=self.countback
        )
        
    def access_data(self, 
                    maxthread: int=4,
                    purpose: str = "train", 
                    replace_old_data: bool = False) -> list[dict]:
        
        if (not isinstance(maxthread, int)) or (not (1 <= maxthread <= 5)):
            raise ValueError('maxthread must be integer, min 1 and max 5')

        results, errors = [], []

        with ThreadPoolExecutor(max_workers=maxthread) as executor:
            futures = {
                executor.submit(
                    self._make_single_accessor(sym).access_one_data,
                    purpose,
                    replace_old_data,
                ): sym
                for sym in self.symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    results.append({"symbol": sym, "data": future.result()})
                except Exception as e:
                    errors.append({"symbol": sym, "error": str(e)})
                    print(f"[ERROR] {sym}: {e}")

        if errors:
            print(f"\n{len(errors)}/{len(self.symbols)} symbol(s) failed.")
        return results
        

# =======================================================
# ================== TEST CASE =========================
# =======================================================

# CMD: python -m data_api.stock_price.stock_price_access

if __name__ == '__main__':
    import time
    time_s = time.perf_counter()

    all_symbols = PORTFOLIO
    portfo = AccessData(all_symbols).access_data(purpose='pred')
    print(len(portfo[0]['data']))
    
    print(time.perf_counter()-time_s)











