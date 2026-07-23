import json
import importlib.util
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np

from DataApi import OhlcvGenerator
from Backtest import FinanceBacktest, TaStatTest
from .Helper import StandardizedDataDict
from TrainingEngine import TrainTA, TrainTestSplit


ALPHA_DIR = Path(__file__).resolve().parent.parent/ "Alpha" # --> Change this to alpha_sample


BLUE = "\033[38;5;45m"
RESET = "\033[0m"

class DataManager:
    @staticmethod
    def generate(config: List[Dict]) -> pd.DataFrame:
        data_list = config['data']
        data_cfg = {key: [item[key] for item in data_list] for key in data_list[0]}
        tv_username = config['username']
        tv_password = config['password']
        update_data = config['update_data'].strip().lower() == 'true'

        database = OhlcvGenerator(**data_cfg, update_data=update_data, username=tv_username, password=tv_password)

        dict_dta = database.generate()
        symbol_configs = database.symbol_configs

        return StandardizedDataDict(dict_dta, symbol_configs)



class Backtest:
    @staticmethod
    def bt_finance(bt_cfg: dict):
        bt = FinanceBacktest(**bt_cfg)
        return bt

    @staticmethod
    def stat_test(alpha_type: str):
        if alpha_type == "ta":
            return TaStatTest()
        else:
            raise NotImplementedError("ML is udner dev, please use ta for now!")



class ConfigManager:
    @staticmethod
    # ------------- Load All Config ---------------
    def _load_config() -> dict:
        cfg_files = list(ALPHA_DIR.glob("*Cfg.json"))

        if not cfg_files:
            raise FileNotFoundError(f"No alpha config (*Cfg.json) found in {ALPHA_DIR}")
        if len(cfg_files) > 1:
            raise RuntimeError("Multiple config files found. Exactly one is allowed.")

        with open(cfg_files[0], "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    # ------------- Load All Config ---------------
    def _dump_config(new_cfg: dict) -> dict:
        cfg_files = list(ALPHA_DIR.glob("*Cfg.json"))

        if not cfg_files:
            raise FileNotFoundError(f"No alpha config (*Cfg.json) found in {ALPHA_DIR}")
        if len(cfg_files) > 1:
            raise RuntimeError("Multiple config files found. Exactly one is allowed.")

        with open(cfg_files[0], "w", encoding="utf-8") as f:
            return json.dump(new_cfg, f, indent=6)



class AlphaManager:

    @staticmethod
    # --------------- Load Alpha into Class ----------------
    def _load_alpha(config: dict):
        if config['alpha_cfg']['alpha_type'] == 'ta':
            alpha_cfg = config["alpha_cfg"]
            alpha_file = ALPHA_DIR / f'{alpha_cfg["filename"]}.py'

            if not alpha_file.exists():
                raise FileNotFoundError(f"Cannot find {alpha_file}. Alpha file must be inside {ALPHA_DIR}")

            spec = importlib.util.spec_from_file_location("alpha_module", alpha_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec for {alpha_file}")
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            try:
                return getattr(module, alpha_cfg["classname"])
            except AttributeError as e:
                raise ImportError(
                    f"Class '{alpha_cfg['classname']}' not found in {alpha_cfg['filename']}")

        else:
            raise NotImplementedError("ML is under developement, use ta for now!")

    @staticmethod
    def _train_alpha(alpha_type: str, alpha_class: None, config: dict, **kwargs):
        if alpha_type == 'ta':
            if "data" not in kwargs or "param_range" not in kwargs:
                raise TypeError("_train_alpha for ta requires 2 arguments: data, param_range")

            data: pd.DataFrame = kwargs.pop("data")
            param_range: dict = kwargs.pop("param_range")

            # Extract optional arguments with default values
            oos_ratio: float = kwargs.pop("oos_ratio", 0.15)
            w4w_val_ratio: float = kwargs.pop("w4w_val_ratio", 0.15)
            w4w_gap: int = kwargs.pop("w4w_gap", 0)
            n_fold: int = kwargs.pop("n_fold", 5)
            n_trials: int = kwargs.pop("n_trials", 500)
            opt_dir: str = kwargs.pop("opt_dir", "maximize")
            opt_metric: str = kwargs.pop("opt_metric", "sharpe")

            if kwargs:
                raise TypeError(f"Got unexpected keyword arguments for 'ta' alpha: {list(kwargs.keys())}")

            # --- Proceed with TA Training Logic ---
            ta_train = TrainTA(alpha=alpha_class,
                               config=config,
                               oos_ratio=oos_ratio, w4w_val_ratio=w4w_val_ratio,
                               w4w_gap=w4w_gap,
                               n_fold=n_fold, n_trials=n_trials,
                               opt_dir=opt_dir, opt_metric=opt_metric)
            best_params = ta_train.start_training(data=data, param_range=param_range)
            config['alpha_cfg']['params'] = best_params
            ConfigManager._dump_config(new_cfg=config)

        else:  
            raise NotImplementedError("ML alpha is under development, please use 'ta' instead") # not implement yet

    

class AlphaBase: 
    def __init__(self):

        # Config 
        self.config = ConfigManager._load_config()
        
        # Data
        self.dm_list = DataManager.generate(self.config)

        # Alpha
        self.class_alpha = AlphaManager._load_alpha(self.config)

        # Finance bt
        self.bt_fin = Backtest.bt_finance(self.config['bt_cfg'])

        # Stat test
        self.bt_stat = Backtest.stat_test(self.config["alpha_cfg"]["alpha_type"])


    def generate_data(self, dt_name: str = None):
        if dt_name:
            return self.dm_list[dt_name]
        return self.dm_list


    def train(self, data: pd.DataFrame, param_range: dict, **kwargs):

        if self.config['alpha_cfg']['alpha_type'] == 'ta':
            self.oos_ratio = kwargs.get("oos_ratio", 0.2)
            w4w_val_ratio = kwargs.get("w4w_val_ratio", 0.15)
            w4w_gap = kwargs.get("w4w_gap", 0)
            n_fold = kwargs.get("n_fold", 5)
            n_trials = kwargs.get("n_trials", 500)
            opt_dir = kwargs.get("opt_dir", "maximize")
            opt_metric = kwargs.get("opt_metric", "sharpe")

            AlphaManager._train_alpha(
                alpha_type='ta',
                alpha_class=self.class_alpha,
                config=self.config,
                data=data, param_range=param_range,
                oos_ratio=self.oos_ratio, w4w_val_ratio=w4w_val_ratio, w4w_gap=w4w_gap,
                n_fold=n_fold, n_trials=n_trials,
                opt_dir=opt_dir, opt_metric=opt_metric)
            
        else:
            raise NotImplementedError("ML is under development, please use 'ta' as alpha type instead")


    def backtest(self, data: pd.DataFrame, oos_ratio: float, plot_pnl: bool = True):
        if self.config['alpha_cfg']['alpha_type'] == 'ta':
            alpha = self.class_alpha(self.config['alpha_cfg']['params'])

            # fin bt
            fin = Backtest.bt_finance(self.config['bt_cfg'])

            try:
                os_ratio = self.oos_ratio
            except AttributeError:
                if not (0 < oos_ratio < 1):
                    raise ValueError("OOS size must be larger than 0 and smaller than 1")
                os_ratio = oos_ratio

            train_df, test_df = TrainTestSplit(test_size=os_ratio).split(data)

            print(f"""{BLUE}##########################################  
        Financial Backtest {self.config['bt_cfg']['fee_type']} 
##########################################{RESET}""")

            for label, df in (("\033[34mIN SAMPLE PERFORMANCE\033[0m", train_df), 
                            ("\033[34mOUT SAMPLE PERFORMANCE\033[0m", test_df)):
                print(' ')
                print(label)
                pos = alpha.run(df)
                df = df.copy()
                df.loc[:, 'position'] = np.asarray(pos)
                fin.pnl_report(df, plot=plot_pnl)

            print(' ')
            print("\033[34mALL DATA PERFORMANCE\033[0m")
            pos = alpha.run(data)
            data = data.copy()
            data.loc[:, 'position'] = np.asarray(pos)
            fin.pnl_report(data, plot=plot_pnl)
                    
            # stat test
            # ta
            print(f"""{BLUE}##########################################  
        TA Stat Backtest {self.config['bt_cfg']['fee_type']} 
##########################################{RESET}""")
            self.bt_stat.set_context(alpha=alpha, bt_fin=self.bt_fin, config=self.config)
            self.bt_stat.stat_check(data=data)

        else:
            raise NotImplementedError("ML is under developement, use alphatype = 'ta' instead.")
    