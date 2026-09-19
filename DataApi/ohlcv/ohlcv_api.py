import os
import re

import urllib3
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from .tradingview_socket import TvSocket
import math

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
YELLOW = "\033[93m"
RED = "\033[91m"
DARK_RED = "\033[38;5;166m"
PINK = "\033[35m"
GREEN = "\033[92m"
PURPLE = "\033[95m"
RESET = "\033[0m"
DATE_COLS = ['datetime', 'date', 'time', 'timestamp']


# =========== Error class =============
class InputError(Exception):
    pass


# =========== Info class ===============
@dataclass
class ResolutionMap:
    available_timeframe = {
        "binance": {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
                    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
                    "1d": "1d"
        },
        "trading_view": { "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "45m": "45",
                        "1h": "1H", "2h": "2H", "3h": "3H","4h": "4H",
                        "1d": "1D"
                        },
        "vietstock":   { "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "45m": "45",
                        "1h": "60", "2h": "120", "3h": "180","4h": "240",
                        "1d": "1D"
                        }
        }
    
    transformed_timeframe = {}
    for platform, mapping in available_timeframe.items():

        candidates = []
        for tf, raw in mapping.items():

            if tf.lower().endswith("m"):
                base = int(tf[:-1])

            elif tf.lower().endswith("h"):
                base = int(tf[:-1]) * 60

            elif tf.lower().endswith("d"):
                base = 1440

            candidates.append((base, tf))

        transformed_timeframe[platform] = sorted(candidates, key=lambda x: x[0])

@dataclass
class ExchangePlatform:
    platform = {
         "tv_vnstock": ["HNX","HOSE","UpCoM"],
         "tv_vnfuture": ["HNX"],
         "tv_usstock": ["NASDAQ", "NYSE"],
         "tv_usfuture": ["CBOE", "TVC", "COMEX"],
         "tv_commodity": ["DARWINEX", "TVC", "FRED", "ECONOMICS"]
        }
    


# ========== Session API class =========
@dataclass
class Headers:
    headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
            'Connection': 'keep-alive',
            'Origin': 'https://stockchart.vietstock.vn',
            'Referer': 'https://stockchart.vietstock.vn/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"'
            }

    headers_vps = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
            'Connection': 'keep-alive',
            'Origin': 'https://chart.vps.com.vn',
            'Referer': 'https://chart.vps.com.vn/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36',
            'sec-ch-ua': '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"'
            }

class RobustSession:
    """
        Auto retry when error 429, 500, 502, 503, 504 is thrown
        Return: new session, max is 4 before raising error
    """
    @staticmethod
    def _create_robust_session(retries: int = 3, backoff_factor: float = 0.25) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=5, pool_maxsize=16)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        return session



# ========== Helper classes ============
class CachingData:

    @staticmethod
    def _get_cache_path(symbol: str, timeframe: str, cache_dir: str):
        base_symbol = symbol
        tf = timeframe

        # Std symbol name
        suf = base_symbol.split(":", 1)[0]
        if suf in ['VN', 'CP', 'C&M', 'VNF']:
            base_symbol = base_symbol.split(":", 1)[1]

        # Std symbol tf
        if isinstance(tf, str):
            if base_symbol.endswith(f"_{tf}"):
                base_symbol = base_symbol[:-len(f"_{tf}")]
            elif tf.startswith(f"{base_symbol}_"):
                tf = tf[len(base_symbol) + 1:]
        else:
            return None

        safe_symbol = base_symbol.replace("/", "_").replace(":", "_")

        return os.path.join(cache_dir, f"{safe_symbol}_{tf}.csv")

    @staticmethod
    def _load_from_cache(symbol: str, timeframe: str, cache_dir: str) -> Optional[pd.DataFrame]:
        cache_path = CachingData._get_cache_path(symbol, timeframe, cache_dir)
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                if df.empty:
                    raise ValueError(f"Dataframe of {symbol}_{timeframe} is empty")
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                return df
            except Exception:
                return None
        
    @staticmethod
    def _save_to_cache(symbol: str, df: pd.DataFrame, timeframe: str, cache_dir: str) -> None:
        cache_path = CachingData._get_cache_path(symbol, timeframe, cache_dir)
        df.to_csv(cache_path, index=False)


class CleanData:

    @staticmethod
    def _standardize_dataframe(df: pd.DataFrame, provider: str, target_tf: str) -> pd.DataFrame:
        required = {"datetime", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns after fetch: {missing}")
        
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        platform = provider.lower()
        if platform in ["crypto", "tv_vnstock", "tv_vnfuture", "tv_commodity"]:
            df["datetime"] = (pd.to_datetime(df["datetime"])
                                .dt.tz_localize(None)
                                + pd.Timedelta(hours=7))

        elif platform in ['tv_usstock', 'tv_usfuture']:     
            df["datetime"] = (pd.to_datetime(df["datetime"], utc=True)
                            .dt.tz_convert("America/New_York")
                            .dt.tz_localize(None))
            
        if str(target_tf)[-1]=='d':
            df['datetime'] = df['datetime'].dt.normalize()

        return df

    @staticmethod
    def _resample_dataframe(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        target = target_interval
        value, unit = int(target[:-1]), target[-1].lower()
        if unit == 'm':
            unit = 'min'

        rule = f"{value}{unit}"
        df = df.set_index("datetime")
        df = df.sort_index()
        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        resampled = resampled.reset_index()
        return resampled
    

class GeneralUtils:

    @staticmethod
    def _compute_base_interval(timeframe: str, platform: str) -> Tuple[str, bool]:
            val = int(timeframe[:-1])
            unit = timeframe[-1].lower()
    
            if unit == "m":
                target = val
            elif unit == "h":
                target = val * 60
            elif unit == "d":
                target = val * 1440
    
            candidates = ResolutionMap.transformed_timeframe[platform]
    
            best_tf = None
            best_bars = float("inf")
    
            for base_minutes, tf in candidates:
    
                if target % base_minutes != 0:
                    continue
    
                bars = target // base_minutes
    
                if bars < best_bars:
                    best_bars = bars
                    best_tf = tf
    
            # fallback: smallest available candle
            if best_tf is None:
                best_tf = candidates[0][1]
    
            is_resampled = (best_tf != timeframe)
    
            return best_tf, is_resampled


    @staticmethod
    def _validate_timeframe(timeframe: str) -> None:
        timeframe = timeframe.lower()
        pattern = r"^\d+[dmh]$"
        if not isinstance(timeframe, str) or not re.match(pattern, timeframe):
            raise InputError(
                f"Invalid timeframe format: '{timeframe}'. Expected pattern: "
                f"positive integer followed by 'd' (days), 'm' (minutes), or 'h' (hours). "
                f"Examples: '1d', '15m', '4h'."
                f"For monthly or yearly data, please call 1d and resample to 1mon, 1y."
            )

    @staticmethod
    def _route_symbol(symbol: str) -> Tuple[str, str]:
        symbol_upper = symbol.upper().strip()

        # Vietnam stock 
        if symbol_upper.startswith("VN:"): 
            return "tv_vnstock", symbol_upper[3:]
        elif len(symbol_upper)==3:
            return "tv_vnstock", symbol_upper
        elif symbol_upper in ['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX', 'UPCOMINDEX']:
            return 'tv_vnstock', symbol_upper
        
        # Vietnam futures
        if symbol_upper in ['VN30F1M', 'VN30F2M']:
            return "tv_vnfuture", symbol_upper
        elif symbol_upper.startswith("VNF:"):
            if symbol_upper[4:] not in ['VN30F1M', 'VN30F2M']:
                return (None, None)
            return "tv_vnfuture", symbol_upper[4:]
        

        # US stock
        if symbol_upper.startswith("US:"): 
            return "tv_usstock", symbol_upper[3:]

        # US futures
        if symbol_upper.startswith("USF:"): 
            return "tv_usfuture", symbol_upper[4:]
        

        # Commodities and Macro
        if symbol_upper.startswith("C&M:"):
            return "tv_commodity", symbol_upper[4:]
        
        
        # Crypto - Binance
        crypto_suffixes = ("USDT", "USDC", "BUSD", "BTC", "ETH")
        if symbol_upper.startswith("CP:"):  
            return "crypto", symbol_upper[3:]
        elif any(symbol_upper.endswith(suf) for suf in crypto_suffixes):
            return "crypto", symbol_upper

        return (None, None)


    @staticmethod
    def _to_unix_seconds(time_start: str, time_end: str, target_tf: str) -> Tuple[int, int]:

        tf = str(target_tf)[-1]
        try:
            if tf=='d':
                time_start = time_start[:10]
                time_end = time_end[:10]
                time_format = "%Y-%m-%d"
            else:
                time_format = "%Y-%m-%d %H:%M:%S"

            vn_tz = ZoneInfo("Asia/Ho_Chi_Minh")
            dt_start = datetime.strptime(time_start, time_format).replace(tzinfo=vn_tz)
            dt_end = datetime.strptime(time_end, time_format).replace(tzinfo=vn_tz)

        except ValueError as e:
            raise InputError(
                f"Timestamp format error: {e}. Expected format: 'YYYY-MM-DD HH:MM:SS'")

        return int(dt_start.timestamp()), int(dt_end.timestamp())




# ============== Component 0: DATA ADJUSTING ENGINE =================
class AdjustData:

    def __init__(self, config: Dict[str, Any], dependency_loader):
        self.config = config
        self.dependency_loader = dependency_loader

    def adjust(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.config["provider"] == "tv_vnfuture" and self.config["symbol"] == "VN30F1M":
            return self._adjust_vnf1m(data)
        if self.config["provider"] == "tv_vnstock":
            pass
        return data


    # Adjust Vietnam Stock
    def _adjust_vnstock(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    
    # Adjust VN30F1M
    def _adjust_vnf1m(self, df: pd.DataFrame) -> pd.DataFrame:

        # Call API VNF_2M
        dependency_config = self.config.copy()
        dependency_config.update({
            "original_symbol": "VN30F2M",
            "symbol": "VN30F2M",
            "target_interval": "1d",
            "base_interval": "1d",
            "requires_resampling": False,
        })

        f2m = self.dependency_loader(dependency_config) # dependency loader already raise err to bloc Null Data

        # Preprocess and Merge into VNF_1M
        if f2m is None or f2m.empty:
            return df

        
        f2m["_trade_date"] = f2m["datetime"].dt.normalize()
        is_third_thursday = (
            (f2m["datetime"].dt.weekday == 3)
            & f2m["datetime"].dt.day.between(15, 21)
        )
        roll_closes = (
            f2m.loc[is_third_thursday]
            .sort_values("datetime")
            .drop_duplicates("_trade_date", keep="last")
            [["_trade_date", "close"]]
            .rename(columns={"close": "close_f2m"})
        )
        if roll_closes.empty:
            return df

        df["_trade_date"] = df["datetime"].dt.normalize()
        df = df.merge(roll_closes, how="left", on="_trade_date", sort=False)
        last_bar_of_day = df["_trade_date"].ne(df["_trade_date"].shift(-1))
        df["_gap"] = np.where(
            last_bar_of_day & df["close_f2m"].notna(),
            df["close_f2m"] - df["close"],
            0.0,
        )

        gaps = pd.Series(df["_gap"].to_numpy(), index=df.index)
        adjustment = gaps.iloc[::-1].cumsum().iloc[::-1]
        for column in ['open', 'high', 'low', 'close']:
            df[f'{column}_adj'] = df[f"{column}"] + adjustment

        return df.drop(columns=["_trade_date", "close_f2m", "_gap"]).reset_index(drop=True)




# ============== Component 1: Input validation and configuration mapper =================
class _ValidateInputParams:
    """
    Validate if input params are format-wise and logic-wise correct

    Returns:
        Validated input, if not meet req -> raise error

    Req:
        timeframe must be int + d/m/h
        datetime must be yyyy-mm-dd h:m:s
        symbol must be available
    """

    def __init__(self, 
                 symbol: Union[str, List[str]], 
                 timeframe: Union[str, List[str]], 
                 time_start: str, 
                 time_end: str=None,
                 username: str = "None",
                 password: str = "None"):
        
        # Std symbol into list format
        if isinstance(symbol, str):
            self.symbol = [symbol]
        elif isinstance(symbol, list):
            self.symbol = symbol

        
        # Std username and password for TradingView account
        if not (isinstance(username, str) and isinstance(password, str)):
            raise InputError("username and password must be in str format")
        anonymous_values = {'', 'no', 'not', 'na', 'none', 'n/a', '0'}
        if username.strip().lower() in anonymous_values or password.strip().lower() in anonymous_values:
            self.username = None
            self.password = None
        else:
            self.username = username
            self.password = password


        # Std time start and time end
        now_str = str(pd.Timestamp.now().floor('s'))
        if isinstance(time_start, str):
            self.time_starts = [time_start] * len(self.symbol)
        elif isinstance(time_start, list):
            if len(time_start) == 1:
                self.time_starts = time_start * len(self.symbol)
            elif len(time_start) == len(self.symbol):
                self.time_starts = time_start
            else:
                raise InputError("Length of time_start list must match the number of symbols")
        else:
            raise InputError("time_start must be a string or a list of strings")

        if not time_end:
            self.time_ends = [now_str] * len(self.symbol)
        elif isinstance(time_end, str):
            self.time_ends = [time_end] * len(self.symbol)
        elif isinstance(time_end, list):
            if len(time_end) == 1:
                val = time_end[0] if time_end[0] is not None else now_str
                self.time_ends = [val] * len(self.symbol)
            elif len(time_end) == len(self.symbol):
                self.time_ends = [t if t is not None else now_str for t in time_end]
            else:
                raise InputError("Length of time_end list must match the number of symbols")
        else:
            raise InputError("time_end must be a string, None, or a list")

        for ts, te in zip(self.time_starts, self.time_ends):
            if ts >= te:
                raise InputError("time_start must be earlier than time_end")
        

        # Validate timeframe and symbol
        if isinstance(timeframe, str) and len(self.symbol) >= 1:
            self.timeframes = [timeframe] * len(self.symbol)
        elif isinstance(timeframe, list) and len(timeframe) == 1 and len(self.symbol) >= 1:
            self.timeframes = timeframe * len(self.symbol)
        elif isinstance(timeframe, list) and len(timeframe) > 1 and len(self.symbol) == 1:
            self.symbol = self.symbol * len(timeframe)
            self.timeframes = timeframe
        elif isinstance(timeframe, list) and len(timeframe)==len(self.symbol):
            self.timeframes = timeframe
        else:
            raise InputError('Must provide only 1 or same number of timeframe as number of symbol')
        for tf in self.timeframes:
            GeneralUtils._validate_timeframe(tf)
        

        # Compute interval based on available timeframe in each platform
        results = [GeneralUtils._route_symbol(sym) for sym in self.symbol]
        for idx, (provider, _) in enumerate(results):
            if provider == 'vietstock' and self.timeframes[idx][-1] != 'd' and self.time_end < '2025-06-27':
                raise InputError('Vietstock do not provide under-1d stock data for date before 2025-06-27')
        
        
        
        self.base_intervals = []
        self.requires_resampling_flags = []
        for idx, tf in enumerate(self.timeframes):
            result = results[idx][0]
            if result == 'crypto':
                platform = 'binance'
            else:
                platform = 'trading_view'
            base, requires = GeneralUtils._compute_base_interval(tf, platform=platform)
            self.base_intervals.append(base)
            self.requires_resampling_flags.append(requires)

        # For each symbol: routing, prefixed overrides, warnings
        self.symbol_configs = []

        today = datetime.today()
        try:
            one_year_ago = (today.replace(year=today.year - 1) + timedelta(days=1)).strftime("%Y-%m-%d")
        except ValueError:
            one_year_ago = today.replace(year=today.year - 1, day=28).strftime("%Y-%m-%d")
        
        if any(tf.endswith(("m", "h")) for tf in self.timeframes):
            print(f"{YELLOW}[WARNING] Intraday data is often limited to {one_year_ago} 09:15:00{RESET}")

        for sym, base_interval, requires_resampling, target_interval, start_t, end_t in zip(
                    self.symbol, self.base_intervals, 
                    self.requires_resampling_flags, 
                    self.timeframes, self.time_starts, self.time_ends):
            
            provider, clean_symbol = GeneralUtils._route_symbol(sym)
            
            # Dynamic Vietstock check using the current item's end date
            if provider == 'vietstock' and target_interval[-1] != 'd' and end_t < '2025-06-27':
                raise InputError('Vietstock do not provide under-1d stock data for date before 2025-06-27')
            
            # Convert specific list-unpacked timestamps to seconds and milliseconds precision
            start_ts_sec, end_ts_sec = GeneralUtils._to_unix_seconds(start_t, end_t, target_tf=target_interval)
            start_ts_ms = start_ts_sec * 1000
            end_ts_ms = end_ts_sec * 1000

            self.symbol_configs.append({
                "original_symbol": sym.upper().strip(),
                "symbol": clean_symbol,
                "provider": provider,
                "base_interval": base_interval,
                "requires_resampling": requires_resampling,
                "target_interval": target_interval,
                "username": self.username,
                "password": self.password,
                "time_start": start_t,
                "time_end": end_t,
                "start_ts_sec": start_ts_sec,
                "end_ts_sec": end_ts_sec,
                "start_ts_ms": start_ts_ms,
                "end_ts_ms": end_ts_ms
            })

    



# ================ Component 2: Single symbol loader (extraction worker) =====================
class _SingleScraper:
    
    # Provider‑specific max candles per request
    MAX_LIMITS = {
        "crypto": 1000,
        "trading_view": 1000
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = RobustSession._create_robust_session()


    def fetch(self) -> Union[pd.DataFrame, Tuple[str, bool, str, str]]:
        provider = self.config["provider"]
        try:
            if provider == "crypto":
                df = self._fetch_crypto()

            elif self.config.get("live_vps") and provider in {'tv_vnstock', 'tv_vnfuture'}:  
                df = self._fetch_vps()  
                
            elif (provider in {'tv_vnstock', 'tv_vnfuture'} and self.config['base_interval'][-1] == 'm' and int(self.config['base_interval'][:-1]) < 15) or\
                 (provider in {'tv_vnstock', 'tv_vnfuture'} and self.config['base_interval']=='1d'):
                df = self._fetch_vietstock()

            else:
                df = self._fetch_trading_view(username=self.config['username'], password=self.config['password'])

            if df is None:
                raise RuntimeError('Scraping is temporarily blocked. Please try again later')
            elif df.empty:
                raise InputError('Wrong ticker name or Unavailable data within defined range.') #a-b  #### ==> Tính năng scrape chỉ data thiếu làm Trading view scrape ngày cuối tuần only -> 0 data -> raise Error (bug)


            df = CleanData._standardize_dataframe(df, provider=self.config['provider'], target_tf=self.config['target_interval'])
            if self.config["requires_resampling"]:
                df = CleanData._resample_dataframe(df, target_interval=self.config['target_interval'])

            return df

        except Exception as e:
            error_name = type(e).__name__
            error_msg = str(e)
            return (self.config["original_symbol"], False, error_name, error_msg)


    
    # =================== Fetch Crypto from Binance ======================
    def _fetch_crypto(self) -> pd.DataFrame:

        symbol = self.config["symbol"]
        base_interval = self.config["base_interval"]
        start_ms = self.config["start_ts_ms"]
        end_ms = self.config["end_ts_ms"]

        resolution_map = ResolutionMap.available_timeframe['binance']
        if base_interval not in resolution_map:
            raise ValueError(f"Binance does not support {self.config['target_interval']} as of no {base_interval} interval")
        

        url = "https://www.binance.com/api/v3/uiKlines"
        all_candles = []
        current_start = start_ms
        resolution = resolution_map[base_interval]

        while current_start < end_ms:
            params = {
                "symbol": symbol,
                "interval": resolution,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": self.MAX_LIMITS["crypto"]
            }
            resp = self.session.get(url,params=params,timeout=70)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            for candle in data:
                open_time = candle[0]
                if open_time > end_ms:
                    break

                all_candles.append({
                    "datetime": open_time,
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                })

            if data[-1][0] >= end_ms:
                break

            next_start = data[-1][6] + 1
            if next_start <= current_start:
                break
            current_start = next_start

            if len(data) < self.MAX_LIMITS["crypto"]:
                break


        df = pd.DataFrame(all_candles)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
            df = (df
                .drop_duplicates(subset="datetime")
                .sort_values("datetime")
                .reset_index(drop=True)
            )

        return df


    # ============ Fetch Commodity, Stocks from Trading View ===================
    def _fetch_trading_view(self, username:str, password:str) -> pd.DataFrame:
        tv = TvSocket(username=username, password=password)

        base_symbol = self.config["symbol"]

        base_interval = self.config["base_interval"]
        resolution_map = ResolutionMap.available_timeframe['trading_view']
        if base_interval not in resolution_map:
            raise ValueError(f"Trading View does not support {self.config['target_interval']} as of no {base_interval} interval")
        interval = resolution_map[base_interval]

        start_ts = int(self.config["start_ts_sec"])
        last_ts = int(self.config["end_ts_sec"])
        end_ts = min(last_ts, int(time.time()))

        if start_ts >= end_ts:
            raise InputError("TradingView request must end no later than the current time")

        if base_interval[-1] == 'm':
            total_bars = math.ceil(
                (end_ts - start_ts) / (60 * int(base_interval[:-1]))
            )
        elif base_interval[-1] == 'h':
            total_bars = math.ceil(
                (end_ts - start_ts) / (3600 * int(base_interval[:-1]))
            )
        elif base_interval[-1] == 'd':
            total_bars = math.ceil(
                (end_ts - start_ts) / (86400 * int(base_interval[:-1]))
            )
        total_bars = total_bars + 1

        all_exc = ExchangePlatform.platform

        # =================== FILTER  ====================

        # Vietnam
        if self.config['provider'] == 'tv_vnstock':
            symbol = "301" if base_symbol == 'UPCOMINDEX' else base_symbol
            for exc in all_exc['tv_vnstock']:
                try:
                    check_data = tv.get_hist(symbol=symbol, exchange=exc, interval=interval, n_bars=total_bars)
                    if check_data is not None and not check_data.empty:
                        break
                except Exception:
                    continue  
            else:
                raise RuntimeError(
                    f"Could not scrape data for Vietnam stock {base_symbol}. "
                    f"Your requests may be blocked. Try again later")
            
        elif self.config['provider'] == 'tv_vnfuture':
            symbol = 'VN30'
            fut = [1 if base_symbol.endswith('F1M') else 2][0]
            for exc in all_exc['tv_vnfuture']:
                try:
                    check_data = tv.get_hist(symbol=symbol, exchange=exc, interval=interval, 
                                             n_bars=total_bars, fut_contract=fut)
                    if check_data is not None and not check_data.empty:
                        break
                except Exception:
                    continue  
            else:
                raise RuntimeError(
                    f"Could not scrape data for Vietnam future {base_symbol}. "
                    f"Your requests may be blocked. Try again later")

            
        # US - ongoing
        elif self.config['provider'] == 'tv_usstock':
            for exc in all_exc['tv_usstock']:
                try:
                    print
                    check_data = tv.get_hist(symbol=base_symbol, exchange=exc, interval=interval, n_bars=total_bars)
                    if check_data is not None and not check_data.empty:
                        break
                except Exception:
                    continue  
            else:
                raise RuntimeError(
                    f"Could not scrape data for US stock {base_symbol}. "
                    f"Your requests may be blocked. Try again later")


        # Commodity and Macro
        elif self.config['provider'] == 'tv_commodity':
            if base_interval[-1] != 'd':
                raise InputError(f'Item {base_symbol} does not accept tf smaller than 1d')
            for exc in all_exc['tv_commodity']:
                try:
                    check_data = tv.get_hist(symbol=base_symbol, exchange=exc, interval=interval, n_bars=total_bars)
                    if check_data is not None and not check_data.empty:
                        break
                except Exception:
                    continue  
            else:
                raise RuntimeError(
                    f"Could not scrape data for commodity or macro index {base_symbol}. "
                    f"Your requests may be blocked. Try again later")

        else:
            raise RuntimeError("Wrong ticker name, please check naming convention")


        df = check_data.copy()
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            start_date = pd.to_datetime(start_ts, unit="s", utc=True)
            last_date = pd.to_datetime(last_ts, unit="s", utc=True)
            df = df[df['datetime'].between(start_date, last_date)]
            df = (df
                .drop_duplicates(subset="datetime")
                .sort_values("datetime")
                .reset_index(drop=True))

        else:
            return None    
            
        return df
    
    # ============ Backup Fetch Vietstock for VN tf < 15m ===================
    def _fetch_vietstock(self) -> pd.DataFrame:

        symbol = self.config["symbol"]

        base_interval = self.config["base_interval"]
        start_sec = int(self.config["start_ts_sec"])
        end_sec = int(self.config["end_ts_sec"])

        resolution_map = ResolutionMap.available_timeframe['vietstock']
        if base_interval not in resolution_map:
            raise ValueError(f"Vietstock does not support {self.config['target_interval']} as of no {base_interval} interval")
        

        url = "https://api.vietstock.vn/tvnew/history"
        all_candles = []
        current_start = start_sec
        resolution = resolution_map[base_interval]

        while current_start < end_sec:

            params = {
                "symbol": symbol,
                "resolution": resolution,
                "from": current_start,
                "to": end_sec,
            }
            
            resp = self.session.get(url, params=params, headers=Headers.headers, timeout=70)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break
            if data.get("s") != "ok":
                break
            timestamps = data.get("t", [])
            if not timestamps:
                break

            opens = data.get("o", [])
            highs = data.get("h", [])
            lows = data.get("l", [])
            closes = data.get("c", [])
            volumes = data.get("v", [])

            for i, ts in enumerate(timestamps):
                if ts > end_sec:
                    break

                all_candles.append({
                    "datetime": ts,
                    "open": float(opens[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "close": float(closes[i]),
                    "volume": float(volumes[i]) if i < len(volumes) else 0.0,
                })

            
            if timestamps[-1] >= end_sec:
                break
        
            next_start = timestamps[-1] + 1
            if next_start <= current_start:
                break
            current_start = next_start


        df = pd.DataFrame(all_candles)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
            df = (df
                .drop_duplicates(subset="datetime")
                .sort_values("datetime")
                .reset_index(drop=True))
            
        return df


    # fetch live for Vietnam securities
    def _fetch_vps(self) -> pd.DataFrame:
        
        symbol = self.config["symbol"]
        start_sec = int(self.config["start_ts_sec"])
        end_sec = int(self.config["end_ts_sec"])

        

        url = "https://histdatafeed.vps.com.vn/tradingview/history"
        all_candles = []
        current_start = start_sec
        resolution = 1

        while current_start < end_sec:

            params = {
                "symbol": symbol,
                "resolution": resolution,
                "from": current_start,
                "to": end_sec,
            }
            
            resp = self.session.get(url, params=params, headers=Headers.headers_vps, timeout=70)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break
            if data.get("s") != "ok":
                break
            timestamps = data.get("t", [])
            if not timestamps:
                break

            opens = data.get("o", [])
            highs = data.get("h", [])
            lows = data.get("l", [])
            closes = data.get("c", [])
            volumes = data.get("v", [])

            for i, ts in enumerate(timestamps):
                if ts > end_sec:
                    break

                all_candles.append({
                    "datetime": ts,
                    "open": float(opens[i]) * 1000 if self.config['provider'] == 'tv_vnstock' else float(opens[i]),
                    "high": float(highs[i]) * 1000 if self.config['provider'] == 'tv_vnstock' else float(highs[i]),
                    "low": float(lows[i]) * 1000 if self.config['provider'] == 'tv_vnstock' else float(lows[i]),
                    "close": float(closes[i]) * 1000 if self.config['provider'] == 'tv_vnstock' else float(closes[i]),
                    "volume": float(volumes[i]) if i < len(volumes) else 0.0,
                })

            
            if timestamps[-1] >= end_sec:
                break
        
            next_start = timestamps[-1] + 1
            if next_start <= current_start:
                break
            current_start = next_start


        df = pd.DataFrame(all_candles)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
            df = (df
                .drop_duplicates(subset="datetime")
                .sort_values("datetime")
                .reset_index(drop=True))
            
        return df
  




# ============ Component 3: RESEARCH DATA SCRAPER ===============
class OhlcvGenerator:

    def __init__(self, 
                 symbol: Union[str, List[str]], timeframe: Union[str, List[str]], time_start: str, time_end: str=None,
                 update_data: bool=False,
                 username: str = "None", password: str = "None",
                 max_workers: int = 5,
                 cache_dir: Optional[str] = None):
        """
        Args:
            symbol: List of ticker symbols (with optional provider prefixes).
            timeframe: e.g. "5m", "2h", "1d".
            time_start, time_end: Format "%Y-%m-%d %H:%M:%S".

            update_data: If True, scrape web for new data no matter if CSV data existed or not.

            max_workers: Thread pool size.
        """

        self.update_data = update_data

        self.max_workers = max_workers

        # Accept either a single timeframe string or a list matching symbol
        if isinstance(timeframe, (str, list)):
            tf_input = timeframe
        else:
            raise InputError('timeframe must be a string or list of strings')


        validator = _ValidateInputParams(symbol, tf_input, time_start, time_end, username, password)

        self.symbol_configs = validator.symbol_configs

        # Backwards-compatible exposures: return scalar when all entries identical
        if len(set(validator.timeframes)) == 1:
            self.timeframe = validator.timeframes[0]
        else:
            self.timeframe = validator.timeframes


        if len(set(validator.base_intervals)) == 1:
            self.base_interval = validator.base_intervals[0]
        else:
            self.base_interval = validator.base_intervals


        if len(set(validator.requires_resampling_flags)) == 1:
            self.requires_resampling = validator.requires_resampling_flags[0]
        else:
            self.requires_resampling = validator.requires_resampling_flags


        self.cache_dir = cache_dir or \
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "_research_data")
        os.makedirs(self.cache_dir, exist_ok=True)


    def _load_single_symbol(self, config: Dict[str, Any], show_log:bool=True) -> Tuple[str, Optional[pd.DataFrame], Optional[Tuple[str, str]]]:
        symbol = config["original_symbol"]
        tf = config['target_interval']
        req_start = pd.to_datetime(config["time_start"])
        req_end = pd.to_datetime(config["time_end"])

        # if tf.lower().endswith("d"):
        #     req_start = req_start.normalize()
        #     req_end = req_end.normalize()

        # 1. Check if data exist -> if enough -> take from cache
        cached_df = CachingData._load_from_cache(symbol, tf, self.cache_dir)

        if cached_df is not None and not cached_df.empty:

            # lowercase cols
            cached_df = cached_df.rename(columns=str.lower)

            # Ensure datetime column
            keywords = ["t", "time", "timestamp", "timestamps", "date", "dates", "datetime", "datetimes"]
            col = next((col for col in cached_df.columns if col in keywords), None)
            cached_df = cached_df.rename(columns={col: 'datetime'})
            cached_df['datetime'] = pd.to_datetime(cached_df['datetime'])

            # if tf.lower().endswith("d"):
            #     cached_df['datetime'] = cached_df['datetime'].dt.normalize()

            
            if not ('datetime' in cached_df.columns):
                raise KeyError(f"Data {symbol}_{tf} misses 'datetime' column")

            # MECHANISM: Block auto-scrape trigger
            # Why: helpful in date sensitive case 
            # -> data end 2020-01-01 14:45:00 BUT config 2020-01-02 as day off can trigger scrape -> slow loader
            if not self.update_data:
                cached_df = (cached_df
                                .drop_duplicates(subset="datetime")
                                .dropna(how='all'))
                return (symbol, cached_df, None)

            # Proceed if exist but NOT ENOGUH DATE RANGE
            cur_data_date = cached_df["datetime"].min()
            cur_data_last_date = cached_df["datetime"].max()

            # Case 1: Non-crypto asset -> Prone to error with smart fetch -> use default
            if config['provider'] != 'crypto':
                if req_start < cur_data_date or req_end > cur_data_last_date:
                    req_start = min(cur_data_date, req_start)
                    req_end = max(cur_data_last_date, req_end)
                else:
                    cached_df = cached_df.drop_duplicates(subset="datetime").dropna(how='all')
                    return (symbol, cached_df, None)

            # Case 2: Crypto = large data -> smart fetch to reduce fetching time
            else:
                if req_start < cur_data_date and req_end <= cur_data_last_date:
                    req_end = cur_data_date
                elif req_start >= cur_data_date and req_end > cur_data_last_date:
                    req_start = cur_data_last_date
                elif req_start < cur_data_date and req_end > cur_data_last_date:
                    pass
                else:
                    cached_df = cached_df.drop_duplicates(subset="datetime").dropna(how='all')
                    return (symbol, cached_df, None)




        # 2. If not found in cache or Datetime is missing -> fetch new

        # New cfg 
        fetch_config = config.copy()
        fetch_config["time_start"] = req_start.strftime("%Y-%m-%d %H:%M:%S")
        fetch_config["time_end"] = req_end.strftime("%Y-%m-%d %H:%M:%S")
        fetch_config["start_ts_sec"], fetch_config["end_ts_sec"] = GeneralUtils._to_unix_seconds(str(req_start), str(req_end), target_tf=tf)
        fetch_config["start_ts_ms"] = fetch_config["start_ts_sec"] * 1000
        fetch_config["end_ts_ms"] = fetch_config["end_ts_sec"] * 1000

        # Scrape raw
        scraper = _SingleScraper(fetch_config)  # ---> Symbol used to fetch
        result = scraper.fetch()

        if isinstance(result, pd.DataFrame) and not result.empty:

            result["datetime"] = pd.to_datetime(result["datetime"])

            if cached_df is not None and not cached_df.empty:
                cached_df = cached_df.reindex(columns=result.columns)

                # Cache avail -> Append raw in
                result = (pd.concat([cached_df, result], ignore_index=True)
                            .drop_duplicates(subset='datetime', keep='first')
                            .dropna(how='all')
                            .reset_index(drop=True)
                            .sort_values(by='datetime')
                        )

            else:
                # No cache -> New raw
                result = (result
                            .drop_duplicates(subset='datetime')
                            .dropna(how='all')
                            .reset_index(drop=True)
                            .sort_values(by='datetime')
                        )

            # Save raw data
            CachingData._save_to_cache(symbol, result, tf, self.cache_dir) 
            if show_log:
                print(f"Downloaded {symbol}_{tf} to cache")
            return (symbol, result, None)
        
        else:
            _, _, err_name, err_msg = result
            if show_log:
                print(f"{DARK_RED}Fail to load {symbol}_{tf}{RESET}")
            return (symbol, None, (err_name, err_msg))


    def _generate_depending_data(self, dep_cfg: Dict[str, Any], dep_exe: ThreadPoolExecutor, dep_fut: Dict, dep_loc: threading.Lock) -> pd.DataFrame:
        
        key = (
            dep_cfg["original_symbol"],
            dep_cfg["target_interval"],
            dep_cfg["time_start"],
            dep_cfg["time_end"])
        with dep_loc:
            future = dep_fut.get(key)
            if future is None:
                future = dep_exe.submit(self._load_single_symbol, dep_cfg, False)
                dep_fut[key] = future

        sym, df, error = future.result()
        if error is not None:
            err_name, err_msg = error
            raise RuntimeError(
                f"Failed to load depending data {sym}_{dep_cfg['target_interval']}: "
                f"{err_name}: {err_msg}"
            )
        if df is None or df.empty:
            raise ValueError(
                f"Empty depending data {sym}_{dep_cfg['target_interval']}"
            )
        
        return df


    # ================================
    # ============== MAIN EXE ============
    # ========================================
    def generate(self) -> Dict[str, Union[pd.DataFrame, Tuple[str, str]]]:

        results = {}
        failed_symbol = []
        phase1_results = []
        dep_fut = {}
        dep_loc = threading.Lock()
        dependency_workers = max(1, min(self.max_workers, 2))

        with (ThreadPoolExecutor(max_workers=self.max_workers) as executor,
              ThreadPoolExecutor(max_workers=dependency_workers) as dep_exe):

            # PHASE 1: scrape and load all requested raw data first.
            future_to_cfg = {
                executor.submit(self._load_single_symbol, cfg): cfg
                for cfg in self.symbol_configs
            }

            for future in as_completed(future_to_cfg):
                cfg = future_to_cfg[future]
                try:
                    sym, df, error = future.result()
                except Exception as exc:
                    sym = cfg["original_symbol"]
                    df = None
                    error = (type(exc).__name__, str(exc))
                phase1_results.append((cfg, sym, df, error))



            # PHASE 2: adjust successful raw frames concurrently.
            adjust_futures = {}
            for cfg, sym, df, error in phase1_results:
                tf = cfg["target_interval"]
                if error is not None:
                    err_name, err_msg = error
                    results[f"{sym}_{tf}"] = (err_name, err_msg)
                    failed_symbol.append((sym, err_name, err_msg))
                    continue
                if df is None or df.empty:
                    error = ("ValueError", f"{sym}_{tf} returned empty dataframe")
                    results[f"{sym}_{tf}"] = error
                    failed_symbol.append((sym, *error))
                    continue

                load_dependency = lambda dep_cfg: self._generate_depending_data(dep_cfg, dep_exe, dep_fut, dep_loc)
                future = executor.submit(
                    AdjustData(cfg, load_dependency).adjust,
                    df,
                )
                adjust_futures[future] = (cfg, sym, tf)

            # Output final data
            for future in as_completed(adjust_futures):
                cfg, sym, tf = adjust_futures[future]
                try:
                    adjusted_df = future.result()
                    time_start = cfg["time_start"]
                    time_end = cfg["time_end"]
                    results[f"{sym}_{tf}"] = adjusted_df[
                        adjusted_df["datetime"].between(time_start, time_end)]  # -> Only take required portion
                    
                    if results[f"{sym}_{tf}"].empty:
                        raise ValueError("Data within specified range does not exist")
                    
                    print(f"{GREEN}Loaded {sym}_{tf}{RESET}")

                except Exception as exc:
                    err_name, err_msg = type(exc).__name__, str(exc)
                    results[f"{sym}_{tf}"] = (err_name, err_msg)
                    failed_symbol.append((sym, tf, err_name, err_msg))

                    print(f"{DARK_RED}Fail to load {sym}_{tf}{RESET}")

        if failed_symbol:
            print(' ')
            print(f"{RED}Failure reason:{RESET}")
            for sym, tf, err_name, err_msg in failed_symbol:
                print(f"'{(sym.split(":", 1)[1] if sym.split(":", 1)[0] in ['VN', 'CP', 'C&M', "VNF"] else sym)}_{tf}': {PURPLE}{err_msg}{RESET}")

        return results



# ============ Component 4: LIVE DATA SCRAPER ===============
"""
- Research co mot so bottle neck nhu:
    + Ko goi duoc Concurrent Per Symbol
    + Exchange fallback dung for loop
    + Call Data VNF2M trong adjust data VN30F1M -> block path cua VN30F1M --> Sol: CHi adjjst cuoi phien, trong phien KO ADJUst
                                                => Trong session, KO CAN ADJUST, close = fillna(close_raw)

==> Deploy Live can GIAM BOT TINH NANG, tap trung vao toc do

"""
# class LiveOhlcvGenerator(OhlcvGenerator):
#     def __init__(self, data_cfg: Dict[str, Any], max_workers: int = 5):
#         self.data_cfg = data_cfg
#         self.live_cache_dir = os.path.join(
#             os.path.dirname(os.path.abspath(__file__)),
#             "_live_data"
#         )
#         os.makedirs(self.live_cache_dir, exist_ok=True)

#         data_items = data_cfg.get("data")
#         symbols = [item["symbol"] for item in data_items]

#         now = pd.Timestamp.now()
#         start = now.normalize() - pd.DateOffset(years=1)
#         end = now

#         super().__init__(
#             symbol=symbols,
#             timeframe=["1m"] * len(symbols),
#             time_start=start.strftime("%Y-%m-%d %H:%M:%S"),
#             time_end=end.strftime("%Y-%m-%d %H:%M:%S"),
#             update_data=True,
#             username=data_cfg.get("tv_username", "None"),
#             password=data_cfg.get("tv_password", "None"),
#             max_workers=max_workers,
#             cache_dir=self.live_cache_dir
#         )

#         self._ensure_live_cache()

#     def _ensure_live_cache(self) -> None:
#         missing_symbols = []

#         for cfg in self.symbol_configs:
#             cache_path = self._get_cache_path(cfg["original_symbol"], "1m")
#             if not os.path.exists(cache_path):
#                 missing_symbols.append(cfg["original_symbol"])

#         if not missing_symbols:
#             return

#         now = pd.Timestamp.now()
#         start = now.normalize() - pd.DateOffset(years=1)
#         end = now

#         generator = OhlcvGenerator(
#             symbol=missing_symbols,
#             timeframe="1m" if len(missing_symbols) == 1 else ["1m"] * len(missing_symbols),
#             time_start=start.strftime("%Y-%m-%d %H:%M:%S"),
#             time_end=end.strftime("%Y-%m-%d %H:%M:%S"),
#             update_data=True,
#             username=self.data_cfg.get("tv_username", "None"),
#             password=self.data_cfg.get("tv_password", "None"),
#             max_workers=self.max_workers,
#             cache_dir=self.live_cache_dir
#         )
#         generator.generate()

#     def _is_vietnam_live_provider(self, provider: str) -> bool:
#         return provider in {"tv_vnstock", "tv_vnfuture"}

#     def _prepare_live_config(self, config: Dict[str, Any]) -> Dict[str, Any]:

#         """
#         Live config da:
#         - copy config cua user -> ko anh huong den config viet gi
#         - live_data tach biet research_data -> good
        

#         Issue:
#         - code fetch_vps -> Done
#         - Adjust data versus Raw data -> 
#         - Khai bao so bars de scrape chuan data hon -> Done
#         - Check self.appluy_rate_delay + tim cach balance speed va avoid_block_api -> Done
#         - LiveOhlcv KO dc raise errorn -> Done
#         """


#         live_config = config.copy()
#         symbol = config["original_symbol"]
#         cached_df = self._load_from_cache(symbol, "1m")

#         now = pd.Timestamp.now()

#         if cached_df is not None and not cached_df.empty and "datetime" in cached_df.columns:
#             cached_df["datetime"] = pd.to_datetime(cached_df["datetime"])
#             last_dt = cached_df["datetime"].max() 
#             req_start = last_dt + pd.Timedelta(minutes=1)
#         else:
#             req_start = now.normalize() - pd.DateOffset(years=1)

#         if req_start >= now:
#             req_start = now

#         live_config["time_start"] = req_start.strftime("%Y-%m-%d %H:%M:%S")
#         live_config["time_end"] = now.strftime("%Y-%m-%d %H:%M:%S")
#         live_config["target_interval"] = "1m"
#         live_config["base_interval"] = "1m"
#         live_config["requires_resampling"] = False
#         live_config["live_vps"] = self._is_vietnam_live_provider(config["provider"])
#         live_config["start_ts_sec"], live_config["end_ts_sec"] = GeneralUtils._to_unix_seconds(
#             str(req_start), str(now)
#         )
#         live_config["start_ts_ms"] = live_config["start_ts_sec"] * 1000
#         live_config["end_ts_ms"] = live_config["end_ts_sec"] * 1000

#         return live_config

#     # Just fetch data, no need bar_live
#     # MISSING Data for papertrade and livetrade would be dealt later
#     def fetch_live(self) -> Dict[str, pd.DataFrame]:
#         results = {}

#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             future_to_config = {}

#             for config in self.symbol_configs:
#                 live_config = self._prepare_live_config(config)
#                 future_to_config[executor.submit(self._load_single_symbol, live_config)] = live_config

#             for future in as_completed(future_to_config):
#                 config = future_to_config[future]
#                 symbol = config["original_symbol"]
#                 _, df, error = future.result()

#                 if error is not None:
#                     _, err_msg = error
#                     print(f"'{(symbol.split(":", 1)[1] if symbol.split(":", 1)[0] in ['VN', 'CP', 'C&M', "VNF"] else symbol)}':{PURPLE}{err_msg}{RESET}")
                                    

#                 results[f"{symbol}_1m"] = df

#         return results


# -----------------------------------------------------------------------------
# python -m DataApi.ohlcv.ohlcv_api
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    generator = OhlcvGenerator(
        symbol=['cts'],
        timeframe=['10m'],
        time_start=["2018-10-01 10:00:00"],
        time_end=["2021-10-15 10:00:00"],
        update_data = True,
        max_workers=3
    )
    data = generator.generate()
    print(data)
