import os
import sys
import re
import time
import random
import urllib3
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from .tradingview_socket import TvSocket
import math

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
YELLOW = "\033[93m"
RED = "\033[91m"
PINK = "\033[35m"
GREEN = "\033[92m"
PURPLE = "\033[95m"
RESET = "\033[0m"


# =========== Helper class ===============
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
         "tv_commodity": ["TVC", "FRED", "ECONOMICS"]
        }

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

class InputError(Exception):
    pass

class RobustSession:
    """
        Auto retry when error 429, 500, 502, 503, 504 is thrown
        Return: new session, max is 4 before raising error
    """
    @staticmethod
    def _create_robust_session(retries: int = 3, backoff_factor: float = 1) -> requests.Session:
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

        # # Std time start and end
        # self.time_start = time_start
        # self.time_end = time_end
        # if self.time_start >= self.time_end:
        #     raise InputError("time_start must be earlier than time_end")
        
        # Std username and password for TradingView account
        
        if not (isinstance(username, str) and isinstance(password, str)):
            raise InputError("username and password must be in str format")
        if not all([username, password]):
            self.username = None
            self.password = None
        elif any(x for x in [username.lower(), password.lower()]) in ['no', 'not', 'na', 'none', 'n/a', '0']:
            self.username = None
            self.password = None
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
            self._validate_timeframe(tf)
        

        # Compute interval based on available timeframe in each platform
        results = [self._route_symbol(sym) for sym in self.symbol]
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
            base, requires = self._compute_base_interval(tf, platform=platform)
            self.base_intervals.append(base)
            self.requires_resampling_flags.append(requires)

        # # Convert timestamps to seconds and milliseconds precision
        # self.start_ts_sec, self.end_ts_sec = self._to_unix_seconds(time_start, time_end)
        # self.start_ts_ms = self.start_ts_sec * 1000
        # self.end_ts_ms = self.end_ts_sec * 1000

        # For each symbol: routing, prefixed overrides, warnings
        self.symbol_configs = []
        # for sym, base_interval, requires_resampling, target_interval in zip(
        #             self.symbol, self.base_intervals, 
        #             self.requires_resampling_flags, 
        #             self.timeframes):
            
        #     provider, clean_symbol = self._route_symbol(sym)
        #     self._print_intraday_warning(provider, clean_symbol, target_interval)
        #     self.symbol_configs.append({
        #         "original_symbol": sym.upper().strip(),
        #         "symbol": clean_symbol,
        #         "provider": provider,
        #         "base_interval": base_interval,
        #         "requires_resampling": requires_resampling,
        #         "target_interval": target_interval,
        #         "username": self.username,
        #         "password": self.password,
        #         "time_start": self.time_start,
        #         "time_end": self.time_end,
        #         "start_ts_sec": self.start_ts_sec,
        #         "end_ts_sec": self.end_ts_sec,
        #         "start_ts_ms": self.start_ts_ms,
        #         "end_ts_ms": self.end_ts_ms,
        #     })
        for sym, base_interval, requires_resampling, target_interval, start_t, end_t in zip(
                    self.symbol, self.base_intervals, 
                    self.requires_resampling_flags, 
                    self.timeframes, self.time_starts, self.time_ends):
            
            provider, clean_symbol = self._route_symbol(sym)
            self._print_intraday_warning(provider, clean_symbol, target_interval)
            
            # Dynamic Vietstock check using the current item's end date
            if provider == 'vietstock' and target_interval[-1] != 'd' and end_t < '2025-06-27':
                raise InputError('Vietstock do not provide under-1d stock data for date before 2025-06-27')
            
            # Convert specific list-unpacked timestamps to seconds and milliseconds precision
            start_ts_sec, end_ts_sec = self._to_unix_seconds(start_t, end_t)
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
                "end_ts_ms": end_ts_ms,
            })


    def _compute_base_interval(self, timeframe: str, platform: str) -> Tuple[str, bool]:
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


    def _validate_timeframe(self, timeframe: str) -> None:
        timeframe = timeframe.lower()
        pattern = r"^\d+[dmh]$"
        if not isinstance(timeframe, str) or not re.match(pattern, timeframe):
            raise ValueError(
                f"Invalid timeframe format: '{timeframe}'. Expected pattern: "
                f"positive integer followed by 'd' (days), 'm' (minutes), or 'h' (hours). "
                f"Examples: '1d', '15m', '4h'."
            )


    def _to_unix_seconds(self, time_start: str, time_end: str) -> Tuple[int, int]:

        try:
            vn_tz = ZoneInfo("Asia/Ho_Chi_Minh")
            dt_start = datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=vn_tz)
            dt_end = datetime.strptime(time_end, "%Y-%m-%d %H:%M:%S").replace(tzinfo=vn_tz)

        except ValueError as e:
            raise InputError(
                f"Timestamp format error: {e}. Expected format: 'YYYY-MM-DD HH:MM:SS'")

        return int(dt_start.timestamp()), int(dt_end.timestamp())


    def _route_symbol(self, symbol: str) -> Tuple[str, str]:
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
                raise InputError('Available Vietnam future contract: VN30F1M, VN30F2M')
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


        raise InputError(f'\nAsset {symbol} does not exist. Please pass \n' 
                         f'- "US:" for us stock; "USF" for US futures \n' 
                         f'- 3-letter or "VN:" for VN stock; "VNF" for Vietnam future \n'
                         f'- usdt, usdc, busd, btc, eth for crypto \n'
                         f'- "C&M" for commodities and macro indexes'
                         )


    def _print_intraday_warning(self, provider: str, symbol: str, timeframe: str) -> None:
        unit = timeframe[-1]
        is_intraday = (unit == 'm' or unit == 'h')
        if not is_intraday:
            return
        if provider.startswith("tv_"):
            print(f"{PINK}[WARNING] Trading View's limit on INTRADAY DATA for {symbol} can cause unexpected error! {RESET}")




# ================ Component 2: Single symbol loader (extraction worker) =====================
class _OhlcvSingleLoader:
    
    # Provider‑specific max candles per request
    MAX_LIMITS = {
        "crypto": 1000,
        "trading_view": 5000
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = RobustSession._create_robust_session()


    def fetch(self) -> Union[pd.DataFrame, Tuple[str, bool, str, str]]:
        provider = self.config["provider"]
        try:
            if provider == "crypto":
                df = self._fetch_crypto()
            elif provider != 'crypto' and self.config['base_interval'] in ['1m', '3m', '5m', '15m']:
                df = self._fetch_vietstock()
            else:
                df = self._fetch_trading_view(username=self.config['username'], password=self.config['password'])

            if df.empty:
                raise ValueError("Ticker does not exist. Please check source, api url, parsing method.")


            df = self._standardize_dataframe(df)
            if self.config["requires_resampling"]:
                df = self._resample_dataframe(df)

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

            time.sleep(0.2)

        df = pd.DataFrame(all_candles)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

            df = (
                df
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
        end_ts = int(time.time())

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
                    f"All exchange variations failed. Check your symbol and try again later.")
            
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
                    f"All exchange variations failed. Check your symbol and try again later.")
            
        # US - ongoing
        if self.config['provider'] == 'tv_usstock':
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
                    f"All exchange variations failed. Check your symbol and try again later.")

        # Commodity and Macro
        if self.config['provider'] == 'tv_commodity':
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
                    f"All exchange variations failed. Check your symbol and try again later.")


        df = check_data.copy()
        if not df.empty:
            start_date = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S')
            last_date = datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M:%S')
            df = df[df['datetime'].between(start_date, last_date)]
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
            df = (df
                .drop_duplicates(subset="datetime")
                .sort_values("datetime")
                .reset_index(drop=True))
            
            
        return df
    
    # ============ Backup Fetch Vietstock for VN tf < 30m ===================
    def _fetch_vietstock(self) -> pd.DataFrame:

        symbol = self.config["symbol"]
        print(f"{PINK}[WARNING] Rechanneled to Vietstock. INTRADAY DATA for {symbol} is limited to 2025-06-27!{RESET}")

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

            time.sleep(0.2)

        df = pd.DataFrame(all_candles)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
            df = (df
                .drop_duplicates(subset="datetime")
                .sort_values("datetime")
                .reset_index(drop=True))
            
        return df
    

    
    # =================  Standardisation & resampling  =========================

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"datetime", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns after fetch: {missing}")
        
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        platform = self.config["provider"].lower()
        if platform in ["crypto", "tv_vnstock", "tv_vnfuture", "tv_commodity"]:
            df["datetime"] = (pd.to_datetime(df["datetime"])
                              .dt.tz_localize(None)
                              + pd.Timedelta(hours=7))

        elif platform in ['tv_usstock', 'tv_usfuture']:     
            df["datetime"] = (pd.to_datetime(df["datetime"], utc=True)
                            .dt.tz_convert("America/New_York")
                            .dt.tz_localize(None))

        return df

    def _resample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        target = self.config["target_interval"]
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




# ============ Component 3: MAIN ENGINE ===============
class OhlcvGenerator:

    def __init__(self, 
                 symbol: Union[str, List[str]], timeframe: Union[str, List[str]], time_start: str, time_end: str=None,
                 save_data: bool = True, update_data: bool=False,
                 username: str = "None", password: str = "None",
                 max_workers: int = 5):
        """
        Args:
            symbol: List of ticker symbols (with optional provider prefixes).
            timeframe: e.g. "5m", "2h", "1d".
            time_start, time_end: Format "%Y-%m-%d %H:%M:%S".

            save_data: If True, store CSV results in 'cached_data' folder.
            update_data: If True, scrape web for new data no matter if CSV data existed or not.

            max_workers: Thread pool size.
        """

        self.save_data = save_data
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


        self.cache_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "cached_data")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, symbol: str, timeframe: Optional[str] = None) -> str:
        base_symbol = symbol
        tf = timeframe

        suf = base_symbol.split(":", 1)[0]
        if suf in ['VN', 'CP', 'C&M', 'VNF']:
            base_symbol = base_symbol.split(":", 1)[1]

        if isinstance(tf, str) and tf:
            if base_symbol.endswith(f"_{tf}"):
                base_symbol = base_symbol[:-len(f"_{tf}")]
            elif tf.startswith(f"{base_symbol}_"):
                tf = tf[len(base_symbol) + 1:]

        safe_symbol = base_symbol.replace("/", "_").replace(":", "_")

        if tf is None:
            for cfg in self.symbol_configs:
                if cfg.get("original_symbol_with_time") == symbol:
                    tf = cfg.get("target_interval")
                    break
            if tf is None:
                tf = self.timeframe if isinstance(self.timeframe, str) else (self.timeframe[0] if self.timeframe else "")
        return os.path.join(self.cache_dir, f"{safe_symbol}_{tf}.csv")


    def _load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        if self.update_data:
            return

        cache_path = self._get_cache_path(symbol, timeframe)
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                print(f"[CACHE] Loaded {symbol} from {cache_path}")
                return df
            except Exception as e:
                print(f"[CACHE] Warning: could not read {cache_path}: {e}")
      

    def _save_to_cache(self, symbol: str, df: pd.DataFrame, timeframe: str) -> None:
        if not self.save_data:
            return

        cache_path = self._get_cache_path(symbol, timeframe)
        df.to_csv(cache_path, index=False)
        print(f"[CACHE] Scraped and Saved {symbol} -> {cache_path}")

    def _apply_rate_delay(self, provider: str) -> None:
        if provider in ("vietstock", "investing"):
            delay = random.uniform(0.1, 0.5)
            time.sleep(delay)

    def _load_single_symbol(self, config: Dict[str, Any]) -> Tuple[str, Optional[pd.DataFrame], Optional[Tuple[str, str]]]:
        
        symbol = config["original_symbol"]
        tf = config['target_interval']

        cached_df = self._load_from_cache(symbol, tf)
        if cached_df is not None:
            return (symbol, cached_df, None)

        self._apply_rate_delay(config["provider"])

        loader = _OhlcvSingleLoader(config)
        result = loader.fetch()

        if isinstance(result, pd.DataFrame):
            self._save_to_cache(symbol, result, tf)
            return (symbol, result, None)
        else:
            _, _, err_name, err_msg = result
            return (symbol, None, (err_name, err_msg))


    # ================================
    # ============== MAIN EXE ============
    # ========================================
    def generate(self) -> Dict[str, Union[pd.DataFrame, Tuple[str, str]]]:

        results = {}
        successful_symbol = []
        failed_symbol = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._load_single_symbol, cfg)
                for cfg in self.symbol_configs
            }


            for idx, future in enumerate(as_completed(future_to_symbol)):
                cfg = self.symbol_configs[idx]
                time_start = cfg["time_start"]
                time_end = cfg["time_end"]
                tf = cfg['target_interval']

                sym, df, error = future.result()

                if error is not None:
                    err_name, err_msg = error
                    results[f'{sym}_{tf}'] = (err_name, err_msg)
                    failed_symbol.append((sym, err_name, err_msg))
                else:
                    results[f'{sym}_{tf}'] = df[df['datetime'].between(time_start, time_end)]
                    successful_symbol.append(sym)

        # Final console log
        total = len(self.symbol_configs)
        success_count = len(successful_symbol)
        print(f"{GREEN}\nSuccessfully scraped {success_count}/{total} symbol{RESET}")
        if failed_symbol:
            print(f"{RED}Failed symbol:{RESET}")
            for sym, err_name, err_msg in failed_symbol:
                print(f"'{(sym.split(":", 1)[1] if sym.split(":", 1)[0] in ['VN', 'CP', 'C&M', "VNF"] else sym)}': {PURPLE}{err_name} - {err_msg}{RESET}")
            sys.exit(1)

        return results


# -----------------------------------------------------------------------------
# python -m DataApi.ohlcv.ohlcv_api
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    generator = OhlcvGenerator(
        symbol=['vn30f1m', 'vn30f1m', 'cts'],
        timeframe=['30m', '1d', '10m'],
        time_start=["2025-11-15 10:00:00", "2026-01-15 10:00:00", "2025-12-15 10:00:00"],
        time_end=["2026-05-15 11:00:00", None, None],
        save_data=True,
        update_data = False,
        max_workers=3
    )
    data = generator.generate()

    for dt, val in data.items():
        print(val)
