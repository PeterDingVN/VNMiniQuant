import json
import importlib.util
from pathlib import Path
from typing import List, Dict

import pandas as pd

from DataApi import OhlcvGenerator
from Backtest import FinanceBacktest

from .Helper import StandardizedDataDict

ALPHA_DIR = Path(__file__).resolve().parent.parent/ "alpha_sample"
# ALPHA_DIR_LOCAL = Path(__file__).resolve().parent.parent/ "Alpha"


# Module: Class Data, Class Test (fin, stat), Class Train (ta -> optuna, ml -> tts build/ custom func) -> voi ML can build cho user
# env de ho train de hon
# Config ta/ml used for Stat test later

class Data:
    @staticmethod
    def generate(config: List[Dict]) -> pd.DataFrame:
        data_list = config['data']
        data_cfg = {key: [item[key] for item in data_list] for key in data_list[0]}
        tv_username = config['username']
        tv_password = config['password']

        database = OhlcvGenerator(**data_cfg, username=tv_username, password=tv_password)

        dict_dta = database.generate()
        symbol_configs = database.symbol_configs

        return StandardizedDataDict(dict_dta, symbol_configs)

class Backtest:
    @staticmethod
    def bt_finance(bt_cfg: dict):
        bt = FinanceBacktest(**bt_cfg)
        return bt


class AlphaBase: 
    
    def __init__(self):
        self.config = self._load_config()
        self.class_alpha = self._load_alpha()
        
        # Data
        self.dm_list = Data.generate(self.config)

        # Alpha
        params = self.config['alpha_cfg']['params']
        self.alpha = self.class_alpha(params)

        # Finance bt
        bt_config = self.config['bt_cfg']
        self.bt_fin = Backtest.bt_finance(bt_config)


    # ------------- Load All Config ---------------
    def _load_config(self) -> dict:
        cfg_files = list(ALPHA_DIR.glob("*.json"))

        if not cfg_files:
            raise FileNotFoundError(f"No alpha config (*Cfg.json) found in {ALPHA_DIR}")
        if len(cfg_files) > 1:
            raise RuntimeError("Multiple config files found. Exactly one is allowed.")

        with open(cfg_files[0], "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------- Load Alpha into Class ----------------
    def _load_alpha(self):
        alpha_cfg = self.config["alpha_cfg"]
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
                f"Class '{alpha_cfg['classname']}' not found in {alpha_cfg['filename']}"
            )

