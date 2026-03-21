from datetime import datetime
from data.stock_price.data_loader import LoadData
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[0]


class AccessData(LoadData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_path = BASE_DIR / "data_cache" / f'{self.symbol}.parquet'

    # SubFunc: Check if the dataset already exist
    def check_path_existence(self) -> Path|None:

        if self.file_path.exists():
            return self.file_path
        else:
            return None

    # Subfunc: If the data path is there, is it up to date?
    def data_is_updated(self, df: pd.DataFrame, date_col: str = 'time') -> bool:
        latest = pd.to_datetime(df[date_col]).max().date()
        today = datetime.now().date()
        delta = (today - latest).days
        return delta == 0 or (delta <= 2 and today.weekday() in (5, 6))

    # MAIN FUNC: Access the data accordingly
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

        # If file exist then:
        # -> return if purpose = train
        # -> check up-to-date and return retrain -> overwrite ori train file is optional
        # -> pred -> same as retrain (no overwrite) -> take only latest N_LAGS + 1 rows

        if purpose == 'retrain':

            # Check for most updated data - retrain purpose
            if not self.data_is_updated(data_ori):
                data_new = self.fetch_stock_price_data()

                # Optional overwrite old original data from train purpose
                if replace_old_data:
                    latest = pd.to_datetime(data_ori['time']).max().date()
                    today = datetime.now().date()
                    print(f"Current {self.symbol} data is updated from {latest} to {today}")
                    data_new.to_parquet(self.file_path, index=False)
                return data_new
            else:
                return data_ori

        elif purpose == 'pred':

            # Check for most updated data - prediction purpose
            if not self.data_is_updated(data_ori):
                data_pred_new = self.fetch_stock_price_data(for_train=False)
                return data_pred_new
            else:
                data_ori = data_ori[-self.candle_nums:]
                return data_ori

        return data_ori # return ori data in case purpose = "train"

if __name__ == '__main__':
    CTD = AccessData(symbol='MWG')
    data = CTD.access_data(purpose="pred")
    print(data)
    print(len(data))









