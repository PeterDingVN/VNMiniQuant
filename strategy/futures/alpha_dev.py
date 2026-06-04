import pandas as pd
import pandas_ta as ta
import numpy as np


class Donchian:
    """
    Alpha PM:
    - Nhận config khi init
    - Nhận dm_list khi run
    """

# CONFIG
    def __init__(self, config: dict):
        self.params = config


    def run(self, df_in):

        data = df_in.copy()
        data.columns = [c.capitalize() for c in data.columns] # Chuẩn hóa tên cột viết hóa chữ cái đầu

        pos_1 = DoubleCycleLines(config=self.params).run(data)
        pos_2 = TrendFollow(config=self.params).run(data)
        print(len(pos_1), len(pos_2))
        pos = pos_1 + pos_2
        
        data['position'] = np.where(pos>=1, 1, np.where(pos<=-1, -1, pos))
       


        return data[['Datetime','Open', 'High', 'Low', 'Close', 'position']]


class VN30:
    def __init__(self):
        self.data = pd.read_parquet(
            r'C:\Users\HP\.0_PycharmProjects\VNMiniQuant_Futures\data\cached_data\stock_price_cache\VN30.parquet')
        
        
        

class DoubleCycleLines:

    def __init__(
        self,
        config,
    ) -> None:


        
        self.rl_period = config["rl_period"]
        self.cl_period = config["cl_period"]
        self.rl_smooth = config["rl_smooth"]
        self.cl_smooth = config["cl_smooth"]
        self.method = config["method"]

   

    # ── Public API ──────────────────────────────────────────────────────────

    
    # OUTPUT
    def run(self, data: pd.DataFrame):

    
        df = data.copy()
        df.columns = [c.lower() for c in df.columns]
        close = df['close']

        dispatch = {
            "trix":   self._compute_trix,
            "linreg": self._compute_linreg,
            "hybrid": self._compute_hybrid,
        }
        if self.method not in dispatch:
            raise ValueError(
                f"Unknown method '{self.method}'. Choose: {list(dispatch)}"
            )
        RL, CL = dispatch[self.method](close)

        df['RL'] = RL
        df['CL'] = CL
        
    
        df["rl_peak"]     = (RL < RL.shift(1)) & (RL.shift(1) > RL.shift(2))
        df["rl_trough"]   = (RL > RL.shift(1)) & (RL.shift(1) < RL.shift(2))
        df["cl_peak"]     = (CL < CL.shift(1)) & (CL.shift(1) > CL.shift(2))
        df["cl_trough"]   = (CL > CL.shift(1)) & (CL.shift(1) < CL.shift(2))
        df["rl_cross_up"] = (RL > CL) & (RL.shift(1) <= CL.shift(1))
        df["rl_cross_dn"] = (RL < CL) & (RL.shift(1) >= CL.shift(1))

        # Pos generation
        pos_short = np.where(df["rl_peak"] | df["rl_cross_dn"], -1, 0)
        pos_long = np.where(df["cl_trough"], 1, 0)
        
        
        # pos_short = np.where(df["rl_peak"], -1, 0)
        # pos_long = np.where(df["rl_trough"]| df["rl_cross_up"], 1, 0)
        
        pos = pos_short + pos_long
        
        pos = pd.Series(pos).replace(0, np.nan)
        pos = pos.ffill().fillna(0)
        pos = np.where(pos>=1, 1, np.where(pos<=-1, -1, pos))
        pos = pd.Series(pos).fillna(0)
    
    
        return pos

    

    # ── Private computation methods ──────────────────────────────────────────
    

    

    def _safe_smooth(
        self, series: pd.Series, smooth: int
    ) -> pd.Series:
        """Apply EMA smoothing, preserving leading NaN values."""
        
        first_valid = series.first_valid_index()
        if first_valid is None:
            return series
            
        # Forward-fill only the tail portion to feed EMA, then restore NaNs
        filled = series.copy()
        loc = series.index.get_loc(first_valid)
        filled.iloc[:loc] = series.iloc[loc]   # backfill head only
        smoothed = ta.ema(filled, smooth)
        
        # Restore original NaN positions
        smoothed[series.isna()] = np.nan
        return smoothed



    def _compute_trix(
        self, close: pd.Series
    ):
        close = self._check_len(close)
        
        """
       TRIX 
        """
        raw_rl = _trix(close, self.rl_period)
        raw_cl = _trix(close, self.cl_period)

        RL = self._safe_smooth(raw_rl, self.rl_smooth)
        CL = self._safe_smooth(raw_cl, self.cl_smooth)
        return RL, CL



    def _compute_linreg(
        self, close: pd.Series
    ):
        
        close = self._check_len(close)
        """
        Linreg
        """
        raw_rl = _linreg_slope_pct(close, self.rl_period)
        raw_cl = _linreg_slope_pct(close, self.cl_period)

        RL = self._safe_smooth(
            self._safe_smooth(raw_rl, self.rl_smooth), self.rl_smooth
        )
        CL = self._safe_smooth(
            self._safe_smooth(raw_cl, self.cl_smooth), self.cl_smooth
        )
        return RL, CL


     
    def _compute_hybrid(
        self, close: pd.Series
    ):
        """
       TRIX (trend) + smoothed DPO (cycle).
        """
        close = self._check_len(close)
        
        # RL: TRIX captures the trend
        raw_rl = _trix(close, self.rl_period)
        RL = self._safe_smooth(raw_rl, self.rl_smooth)

        # CL: DPO isolates the cycle → SuperSmoother for noise reduction
        raw_dpo = _dpo(close, self.cl_period)
        # Double SuperSmoother for very clean cycle line
        ss1 = _super_smoother(raw_dpo.fillna(0.0), self.cl_period // 2)
        ss2 = _super_smoother(ss1, self.cl_period // 3)
        CL = ss2
        CL[raw_dpo.isna()] = np.nan

        return RL, CL






    def _check_len(self, close):    
        if self.rl_period > self.cl_period:
           close = pd.Series(np.zeros(len(close)), index=close.index)

        return close
            
    






def _trix(series: pd.Series, period: int) -> pd.Series:
    e1 = ta.ema(series, period)
    e2 = ta.ema(e1, period)
    e3 = ta.ema(e2, period)
    
    trix_val = (e3 - e3.shift(1)) / e3.shift(1) * 100
    return trix_val


def _super_smoother(series: pd.Series, period: int) -> pd.Series:
    """
    Ehlers SuperSmoother Filter (2-pole low-pass Butterworth variant).

    """
    a1 = np.exp(-1.414 * np.pi / period)
    b1 = 2.0 * a1 * np.cos(np.radians(1.414 * 180.0 / period))
    c2 = b1
    c3 = -(a1 * a1)
    c1 = 1.0 - c2 - c3

    vals = series.values.astype(float)
    n = len(vals)
    out = np.empty(n)
    out[0] = vals[0]
    if n > 1:
        out[1] = vals[1]
    for i in range(2, n):
        out[i] = (c1 * (vals[i] + vals[i - 1]) / 2.0
                  + c2 * out[i - 1]
                  + c3 * out[i - 2])
    return pd.Series(out, index=series.index)


def _linreg_slope_pct(series: pd.Series, period: int) -> pd.Series:
    """
    Rolling linear regression slope, normalized as % of current price.
    
    This directly implements the "Regression Line" (hồi quy xu hướng) concept:
    it measures how fast price is rising or falling in percentage terms,
    based on a least-squares fit over the last `period` bars.
    
    Returns values oscillating around 0:
      >0 : price trending up
      <0 : price trending down
      Magnitude: how steep the trend is (larger = stronger)
    """
    x = np.arange(period, dtype=float)
    x_bar = x.mean()
    ss_xx = float(((x - x_bar) ** 2).sum())

    vals = series.values.astype(float)
    n = len(vals)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        y = vals[i - period + 1 : i + 1]
        y_bar = y.mean()
        ss_xy = float(np.dot(x - x_bar, y - y_bar))
        slope = ss_xy / ss_xx
        # Normalize: slope per bar as % of current price level
        if vals[i] != 0:
            result[i] = slope / vals[i] * 100.0

    return pd.Series(result, index=series.index)


def _dpo(series: pd.Series, period: int) -> pd.Series:
    """
    Detrended Price Oscillator (DPO), normalized as % of price.
    
    Removes long-term trend to isolate the underlying price cycle:
      DPO = close[shifted back period/2+1 bars] - SMA(close, period)
    
    The result oscillates symmetrically around 0, with peaks/troughs
    corresponding to cyclical highs and lows.
    """
    shift = period // 2 + 1
    sma_n = series.rolling(window=period, min_periods=period).mean()
    raw = series.shift(shift) - sma_n
    return raw / series * 100.0


def _detect_peaks_troughs(
    series: pd.Series,
):
    """
    Detect local peaks and troughs using the 3-bar confirmation rule
    stated explicitly in the Simplize documentation:
    
    Peak:   RL_t < RL_{t-1}  AND  RL_{t-1} > RL_{t-2}   (confirmed at bar t)
    Trough: RL_t > RL_{t-1}  AND  RL_{t-1} < RL_{t-2}   (confirmed at bar t)
    
    Note: the actual turning point occurred at bar t-1, but we can only
    confirm it at bar t when we see the subsequent move.
    """
    peak = (series.shift(1) > series.shift(2)) & (series.shift(1) > series)
    trough = (series.shift(1) < series.shift(2)) & (series.shift(1) < series)
    return peak, trough



class TrendFollow:

    def __init__(self, config):
        self.config = config
        
    @staticmethod
    def get_third_thursday(year, month):
        import calendar
        c = calendar.monthcalendar(year, month)
        thursdays = [week[3] for week in c if week[3] != 0]
        return thursdays[2] # The third one


    def run(self, df_in):
        """
        df columns required: 'future_close', 'index_close', 'date'
        """
        config = self.config

        df = df_in.copy()
        
        n_period = config["lookback"]
        don_period = config["don_lb"]
        basis_period = config["basis_lb"]
        long_basis = config["long_basis"]
        short_basis = config["short_basis"]


        # 1. Extract parameters from your config dictionary
        # 1. Extract using your specific key names
        trend_lb       = config["trend_lb"]
        atr_lb         = config["atr_lb"]  # Your config uses "atr_lb"
        bull_threshold = config["bull_threshold"]
        bear_threshold = config["bear_threshold"]
        smooth_lb      = config["smooth_lb"]

        
        df.columns = [c.lower() for c in df.columns]
        

        vn30 = VN30().data
        df = df.merge(vn30, on='datetime', how='left', suffixes=("", "_vn30"))

        df = df.rename(columns={'datetime': 'date'})
        
        # 1. Standard Donchian Channels
        df['upper_dc'] = df['close'].rolling(window=don_period-1).max().shift(1)
        df['lower_dc'] = df['close'].rolling(window=don_period-1).min().shift(1)
        
        # 2. Basis Features
        df['basis'] = df['close'] - df['close_vn30']
        
        df['basis_mean'] = df['basis'].rolling(window=n_period).mean().shift(1)
        df['basis_std'] = df['basis'].rolling(window=n_period).std().shift(1)
        df['basis_z'] = (df['basis'] - df['basis_mean']) / df['basis_std']


        # Reindex back to intraday
        df['regime'] = (
           df['basis_z']
            .rolling(basis_period, min_periods=1)
            .median().shift(1)
        )



        # 3. Expiry Logic
        def is_near_expiry(row):
            expiry_day = TrendFollow.get_third_thursday(row['date'].year, row['date'].month)
            days_to_expiry = expiry_day - row['date'].day
            # Return True if we are in expiry week (0 to 3 days before Thursday)
            return 0 <= days_to_expiry <=  2

        df['near_expiry'] = df.apply(is_near_expiry, axis=1)


        
        # Long
        df['long_signal'] = np.where(
            (df['close'] > df['upper_dc']) & 
            (df['regime'] < long_basis) &
            (~df['near_expiry']) 
            ,1, 0
        )

        # Short
        df['short_signal'] = np.where(
            (df['close'] < df['lower_dc']) & 
            (df['regime'] > short_basis)&
            (~df['near_expiry']), 
            -1, 0
        )

     
        df['pos'] = df['long_signal']+df['short_signal']
        df['pos'] = df['pos'].replace(0, np.nan)
        pos_trend = df['pos'].ffill().fillna(0)



        
        
        mode_arr = RegimeDetector.linreg_trend_state(
                            df=df, 
                            trend_lb=trend_lb,
                            atr_lb=atr_lb,  # Mapping your atr_lb to the vol_lb argument
                            bull_threshold=bull_threshold,
                            bear_threshold=bear_threshold,
                            smooth_lb=smooth_lb
                        )
    

        
        pos = np.where(mode_arr, pos_trend, 0)

        return pos



                                                                            # BEST NOW 30M
class RegimeDetector:
   
    @staticmethod
    def _linreg_slope(series: pd.Series, window: int) -> pd.Series:
        # Ensure we are using pandas_ta linreg
        return ta.linreg(series, length=window, slope=True)

    @staticmethod
    def linreg_trend_state(
        df: pd.DataFrame,
        trend_lb: int,
        atr_lb: int,
        bull_threshold: float, # Lower for slow grinds (e.g. 0.15)
        bear_threshold: float, # Higher for violent drops (e.g. 0.35)
        smooth_lb: int      # Filters out the "dashing"/chatter
    ) -> np.ndarray:
        
        data = df.copy()
        data.columns = [c.lower() for c in data.columns]

        # 1. Calculate raw slope
        # Shifting by 1 to prevent lookahead bias if using for signals
        data['slope'] = RegimeDetector._linreg_slope(data['close'].shift(1), trend_lb)

        # 2. Calculate ATR for normalization
        data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=atr_lb)

        # 3. Normalize Slope
        # This represents: "How many ATRs did the price move per bar?"
        data['norm_slope'] = data['slope'] / data['atr']

        # 4. Smoothing Layer (The Anti-Chatter Filter)
        # Smoothing the normalized slope prevents the regime from flickering
        # during minor pullbacks in a macro trend.
        if smooth_lb > 1:
            data['norm_slope_smooth'] = data['norm_slope'].ewm(span=smooth_lb).mean()
        else:
            data['norm_slope_smooth'] = data['norm_slope']

        # 5. Asymmetric Trend Logic
        # We check positive slope against bull and negative slope against bear
        is_bull = data['norm_slope_smooth'] > bull_threshold
        is_bear = data['norm_slope_smooth'] < -bear_threshold

        trend_state = (is_bull | is_bear).astype(int)

        return trend_state.to_numpy()


class MFI_MACD:


    def __init__(self, config=None):
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    def _ma(self, series: pd.Series, period: int, ma_type: str) -> pd.Series:
        if ma_type.upper() == "EMA":
            return self._ema(series, period)
        elif ma_type.upper() == "SMA":
            return self._sma(series, period)
        else:
            raise ValueError(f"Unknown MA type: {ma_type!r}. Use 'EMA' or 'SMA'.")

    @staticmethod
    def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: int) -> pd.Series:
        """
        Money Flow Index (mirrors ta.mfi(hlc3, period) in Pine Script).
        Pine's ta.mfi uses hlc3 as the typical price, same as standard MFI.
        """
        mfi = ta.mfi(high=high, low=low, close=close, volume=volume, length=period)
     
        return mfi

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df_in : pd.DataFrame
            Must contain columns: open, high, low, close, volume
            (case-insensitive).

        Returns
        -------
        pd.DataFrame with columns:
            macd, signal, hist
        """
        df = df_in.copy()
        df.columns = [c.lower() for c in df.columns]

        cfg = self.config
        rsi_len    = cfg["rsi_len"]
        fast_len   = cfg["fast_len"]
        slow_len   = cfg["slow_len"]
        signal_len = cfg["signal_len"]
        osc_type   = cfg["osc_type"]
        sig_type   = cfg["sig_type"]

        # Step 1 – MFI (Pine: ta.mfi(hlc3, rsi_lb))
        source = self._mfi(df["high"], df["low"], df["close"], df["volume"], rsi_len)

        # Step 2 – Fast / slow MAs on the MFI series
        ma_fast = self._ma(source, fast_len, osc_type)
        ma_slow = self._ma(source, slow_len, osc_type)

        # Step 3 – MACD line, signal, histogram
        macd   = ma_fast - ma_slow
        signal = self._ma(macd, signal_len, sig_type)
        hist   = macd - signal


                # MACD crossover strategy

        long_signal = (
            (macd > signal) &
            (macd.shift(1) <= signal.shift(1))
        )

        short_signal = (
            (macd < signal) &
            (macd.shift(1) >= signal.shift(1))
        )

        state = 0.0
        position = np.zeros(len(df))

        for i in range(len(df)):
            if long_signal.iloc[i]:
                state = 1.0
            elif short_signal.iloc[i]:
                state = -1.0

            position[i] = state

        df['position'] = np.where(position==0, np.nan, position)
        df['position'] = df['position'].ffill().fillna(0)

        return df['position']
