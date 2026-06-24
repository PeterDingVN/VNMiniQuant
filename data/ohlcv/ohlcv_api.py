import os
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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
YELLOW = "\033[93m"
RED = "\033[91m"
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
        "vietstock": { "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "45m": "45",
                        "1h": "60", "2h": "120", "3h": "180","4h": "240",
                        "1d": "1D"
                        },
        "investing": {"5m": "5", "15m": "15", "30m": "30",
                      "1h": "60",
                      "1d": "D"}
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

class InputError(Exception):
    pass

class RobustSession:
    """
        Auto retry when error 429, 500, 502, 503, 504 is thrown
        Return: new session, max is 4 before raising error
    """
    @staticmethod
    def _create_robust_session(retries: int = 4, backoff_factor: float = 1) -> requests.Session:
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

    def __init__(self, symbols: Union[str, List[str]], timeframe: Union[str, List[str]], time_start: str, time_end: str):
        
        if isinstance(symbols, str):
            self.symbols = [symbols]
        elif isinstance(symbols, list):
            self.symbols = symbols
        self.time_start = time_start
        self.time_end = time_end

        # Validate timeframe and symbol
        if isinstance(timeframe, str):
            self.timeframes = [timeframe] * len(self.symbols)
        elif isinstance(timeframe, list) and len(timeframe)==len(self.symbols):
            self.timeframes = timeframe
        else:
            raise InputError('Must provide only 1 or same number of timeframe as number of symbol')
        for tf in self.timeframes:
            self._validate_timeframe(tf)


        # Compute interval based on available timeframe in each platform
        results = [self._route_symbol(sym) for sym in self.symbols]
        self.base_intervals = []
        self.requires_resampling_flags = []
        for idx, tf in enumerate(self.timeframes):
            base, requires = self._compute_base_interval(tf, platform=results[idx][0])
            self.base_intervals.append(base)
            self.requires_resampling_flags.append(requires)

        # Convert timestamps to seconds and milliseconds precision
        self.start_ts_sec, self.end_ts_sec = self._to_unix_seconds(time_start, time_end)
        self.start_ts_ms = self.start_ts_sec * 1000
        self.end_ts_ms = self.end_ts_sec * 1000

        # For each symbol: routing, prefixed overrides, warnings
        self.symbol_configs = []
        for sym, base_interval, requires_resampling, target_interval in zip(
                    self.symbols, self.base_intervals, 
                    self.requires_resampling_flags, 
                    self.timeframes):
            
            provider, clean_symbol = self._route_symbol(sym)
            self._print_intraday_warning(provider, clean_symbol, target_interval)
            self.symbol_configs.append({
                "original_symbol": sym,
                "symbol": clean_symbol,
                "provider": provider,
                "base_interval": base_interval,
                "requires_resampling": requires_resampling,
                "target_interval": target_interval,
                "start_ts_sec": self.start_ts_sec,
                "end_ts_sec": self.end_ts_sec,
                "start_ts_ms": self.start_ts_ms,
                "end_ts_ms": self.end_ts_ms,
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
        if dt_start >= dt_end:
            raise InputError("time_start must be earlier than time_end")

        return int(dt_start.timestamp()), int(dt_end.timestamp())

    def _route_symbol(self, symbol: str) -> Tuple[str, str]:
        symbol_upper = symbol.upper()

        # Vietnam stock 
        if symbol_upper.startswith("VN:"): 
            return "vietstock", symbol_upper[3:]
        elif len(symbol_upper)==3:
            return "vietstock", symbol_upper
        elif symbol_upper.isin(['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX', 'UPCOMINDEX']):
            return "vietstock", symbol_upper

        
        # US stock
        if symbol_upper.startswith("US:"):
            return "investing", symbol_upper[3:]
        
        # Crypto
        crypto_suffixes = ("USDT", "USDC", "BUSD", "BTC", "ETH")
        if symbol_upper.startswith("CP:"):  
            return "binance", symbol_upper[3:]
        elif any(symbol_upper.endswith(suf) for suf in crypto_suffixes):
            return "binance", symbol_upper

        raise InputError(f'Asset {symbol} does not exist. Please pass US: for us stock,' 
                         f'3-letter or VN: for VN stock, and usdt or similar suffixes for crypto')


    def _print_intraday_warning(self, provider: str, symbol: str, timeframe: str) -> None:
        unit = timeframe[-1]
        is_intraday = (unit == 'm' or unit == 'h')
        if not is_intraday:
            return
        if provider == "investing":
            print(f"{YELLOW}[WARNING] Investing.com intraday data for {symbol} might be "
                  f"available only since June 2020. Please double check before use!{RESET}")
        elif provider == "vietstock":
            print(f"{YELLOW}[WARNING] Vietstock intraday data for {symbol} might be "
                  f"available since mid-2025 only. Please double check before use!{RESET}")




# ================ Component 2: Single symbol loader (extraction worker) =====================
class _OhlcvSingleLoader:
    
    # Provider‑specific max candles per request
    MAX_LIMITS = {
        "binance": 1000,
        "vietstock": 1000,  
        "investing": 1000,  
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = RobustSession._create_robust_session()
        self.headers = {
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

    def fetch(self) -> Union[pd.DataFrame, Tuple[str, bool, str, str]]:

        provider = self.config["provider"]
        try:
            if provider == "binance":
                df = self._fetch_binance()
            elif provider == "vietstock":
                df = self._fetch_vietstock()
            elif provider == "investing":
                df = self._fetch_investing()
            else:
                raise ValueError(f"Unknown provider: {provider}")

            if df.empty:
                raise ValueError("Fetched data is empty")


            df = self._standardize_dataframe(df)
            if self.config["requires_resampling"]:
                df = self._resample_dataframe(df)

            return df

        except Exception as e:
            error_name = type(e).__name__
            error_msg = str(e)
            return (self.config["original_symbol"], False, error_name, error_msg)

    
    # =================== Fetch Binance ======================
    def _fetch_binance(self) -> pd.DataFrame:

        symbol = self.config["symbol"]
        base_interval = self.config["base_interval"]
        start_ms = self.config["start_ts_ms"]
        end_ms = self.config["end_ts_ms"]

        resolution_map = ResolutionMap.available_timeframe['binance']
        if base_interval not in resolution_map:
            raise ValueError(f"Vietstock does not support {self.config['target_interval']} as of no {base_interval} interval")
        

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
                "limit": self.MAX_LIMITS["binance"]
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

            if len(data) < self.MAX_LIMITS["binance"]:
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


    # ============ Fetch Vietstock ===================
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
            
            resp = self.session.get(url, params=params, headers=self.headers, timeout=70)
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
    


    # ===================== Fetch Investing ======================
    def _fetch_investing(self) -> pd.DataFrame:

        symbol = self.config["symbol"]
        base_interval = self.config["base_interval"]
        start_sec = self.config["start_ts_sec"]
        end_sec = self.config["end_ts_sec"]

        res_map = ResolutionMap.available_timeframe['investing']
        if base_interval not in res_map:
            raise ValueError(f"Investing does not support {self.config['target_interval']} as of no {base_interval} interval")

        
        url = "https://tvc4.investing.com/127911700fc4e5afa1929fb2ab34b234/1781436887/1/1/8/history"
        all_candles = []
        current_start = start_sec
        resolution = res_map[base_interval]

        while current_start < end_sec:
            params = {
                "symbol": symbol,
                "resolution": resolution,
                "from": current_start,
                "to": end_sec
            }
            resp = self.session.get(url, params=params, timeout=70)
            resp.raise_for_status()
            data = resp.json()

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
            df["datetime"] = pd.to_datetime(
                df["datetime"], unit="s")

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
        if platform in ["binance", "vietstock"]:
            df["datetime"] = (pd.to_datetime(df["datetime"])
                              .dt.tz_localize(None)
                              + pd.Timedelta(hours=7))

        elif platform == "investing":     
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
                 symbols: Union[str, List[str]], timeframe: Union[str, List[str]], time_start: str, time_end: str=None,
                 save_data: bool = True, update_data: bool=False,
                 max_workers: int = 5):
        """
        Args:
            symbols: List of ticker symbols (with optional provider prefixes).
            timeframe: e.g. "5m", "2h", "1d".
            time_start, time_end: Format "%Y-%m-%d %H:%M:%S".

            save_data: If True, store CSV results in 'cached_data' folder.
            update_data: If True, scrape web for new data no matter if CSV data existed or not.

            max_workers: Thread pool size.
        """

        self.save_data = save_data
        self.update_data = update_data

        self.max_workers = max_workers

        # Accept either a single timeframe string or a list matching symbols
        if isinstance(timeframe, (str, list)):
            tf_input = timeframe
        else:
            raise InputError('timeframe must be a string or list of strings')

        if not time_end:
            time_end = str(pd.Timestamp.now().floor('s'))

        validator = _ValidateInputParams(symbols, tf_input, time_start, time_end)

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

    def _get_cache_path(self, symbol: str) -> str:
        safe_symbol = symbol.replace("/", "_").replace(":", "_")
        # Use the symbol's target_interval from symbol_configs when available
        tf = None
        for cfg in self.symbol_configs:
            if cfg.get("original_symbol") == symbol:
                tf = cfg.get("target_interval")
                break
        if tf is None:
            tf = self.timeframe if isinstance(self.timeframe, str) else (self.timeframe[0] if self.timeframe else "")
        return os.path.join(self.cache_dir, f"{safe_symbol}_{tf}.csv")

    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        
        cache_path = self._get_cache_path(symbol)
        if os.path.exists(cache_path) and self.update_data==False:
            try:
                df = pd.read_csv(cache_path)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                print(f"[CACHE] Loaded {symbol} from {cache_path}")
                return df
            except Exception as e:
                print(f"[CACHE] Warning: could not read {cache_path}: {e}")
        return None

    def _save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        if not self.save_data:
            return
        cache_path = self._get_cache_path(symbol)
        df.to_csv(cache_path, index=False)
        print(f"[CACHE] Scraped and Saved {symbol} -> {cache_path}")

    def _apply_rate_delay(self, provider: str) -> None:
        if provider in ("vietstock", "investing"):
            delay = random.uniform(0.1, 0.5)
            time.sleep(delay)

    def _load_single_symbol(self, config: Dict[str, Any]) -> Tuple[str, Optional[pd.DataFrame], Optional[Tuple[str, str]]]:
        
        symbol = config["original_symbol"]
        cached_df = self._load_from_cache(symbol)
        if cached_df is not None:
            return (symbol, cached_df, None)

        self._apply_rate_delay(config["provider"])

        loader = _OhlcvSingleLoader(config)
        result = loader.fetch()

        if isinstance(result, pd.DataFrame):
            self._save_to_cache(symbol, result)
            return (symbol, result, None)
        else:
            _, _, err_name, err_msg = result
            return (symbol, None, (err_name, err_msg))


    # ================================
    # ============== MAIN EXE ============
    # ========================================
    def generate(self) -> Dict[str, Union[pd.DataFrame, Tuple[str, str]]]:

        results = {}
        successful_symbols = []
        failed_symbols = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._load_single_symbol, cfg): cfg["original_symbol"]
                for cfg in self.symbol_configs
            }

            for future in as_completed(future_to_symbol):
                sym, df, error = future.result()
                if error is not None:
                    err_name, err_msg = error
                    results[sym] = (err_name, err_msg)
                    failed_symbols.append((sym, err_name, err_msg))
                else:
                    results[sym] = df
                    successful_symbols.append(sym)

        # Final console log
        total = len(self.symbol_configs)
        success_count = len(successful_symbols)
        print(f"{GREEN}\nSuccessfully scraped {success_count}/{total} symbols{RESET}")
        if failed_symbols:
            print(f"{RED}Failed symbols:{RESET}")
            for sym, err_name, err_msg in failed_symbols:
                print(f"'{sym}': {PURPLE}{err_name} - {err_msg}{RESET}")

        return results


# -----------------------------------------------------------------------------
# Example usage (if run as script)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Demo: fetch mixed symbols with caching and resampling
    generator = OhlcvGenerator(
        symbols="CTS",
        timeframe='30m',
        time_start="2025-01-01 00:00:00",
        # time_end="2026-06-14 00:00:00",
        save_data=True,
        update_data = True,
        max_workers=3
    )
    data = generator.generate()
