import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class Donchian:
    def __init__(self, config):
        self.config = config

    def run(self, df_in: pd.DataFrame) -> np.ndarray:
        df = TrendTrading._prepare_df(df_in)

        trend_mode = TrendTrading.get_mode(df, self.config)
        trend_pos = trend_mode.astype(float)
        sideway_pos = SidewayGridTrading.run(df, self.config, trend_mode)
        # Trend overrides sideway. Sideway only acts when trend_mode == 0
        df['position'] = np.where(trend_mode != 0, trend_pos, sideway_pos).astype(float)


        plot = True
        if plot:
            # Calculate step-by-step PnL (excluding fees for visual simplicity)
            # You hold the position from the previous time step into the current price diff
            df['pos_held'] = df['position'].shift(1).fillna(0.0)
            df['price_diff'] = df['close'].diff().fillna(0.0)
            df['bar_pnl'] = df['pos_held'] * df['price_diff']

            x = np.arange(len(df))
            y = df["close"].to_numpy(dtype=float)
            pnl = df["bar_pnl"].to_numpy(dtype=float)

            # Build line segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Colors based on bar-by-bar PnL
            colors = []
            for i in range(1, len(df)):
                val = pnl[i]
                if val > 0:
                    colors.append("green")  # Profitable step
                elif val < 0:
                    colors.append("red")    # Losing step
                else:
                    colors.append("grey")   # Flat or no price movement

            # Create colored line collection
            lc = LineCollection(
                segments,
                colors=colors,
                linewidths=2
            )

            # Plot
            fig, ax = plt.subplots(figsize=(18, 8))
            ax.add_collection(lc)

            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(np.nanmin(y), np.nanmax(y))

            ax.set_title("Price Path Colored by Trade Profit/Loss (Green = Win, Red = Loss)")
            ax.set_xlabel("Index")
            ax.set_ylabel("Close Price")

            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

            # Clean up temporary columns used for plotting
            df = df.drop(columns=['pos_held', 'price_diff', 'bar_pnl'])


        # mode = TrendTrading.get_mode(df, self.config)

        # x = np.arange(len(df))
        # y = df["close"].to_numpy(dtype=float)

        # # Build line segments
        # points = np.array([x, y]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # # Colors based on regime
        # colors = []

        # for m in mode[:-1]:

        #     if m == 1:
        #         colors.append("green")

        #     elif m == -1:
        #         colors.append("red")

        #     else:
        #         colors.append("grey")

        # # Create colored line collection
        # lc = LineCollection(
        #     segments,
        #     colors=colors,
        #     linewidths=2
        # )

        # # Plot
        # fig, ax = plt.subplots(figsize=(18, 8))

        # ax.add_collection(lc)

        # ax.set_xlim(x.min(), x.max())
        # ax.set_ylim(np.nanmin(y), np.nanmax(y))

        # ax.set_title("Trend Detection via ta.linreg Slope")
        # ax.set_xlabel("Index")
        # ax.set_ylabel("Close Price")

        # plt.grid(alpha=0.3)
        # plt.tight_layout()
        # plt.show()


        

        return df
    
class TrendTrading:
    """
    Trend mode:
      +1 = uptrend  -> hold long
      -1 = downtrend -> hold short
       0 = no trend  -> let sideway module trade
    """

    @staticmethod
    def _prepare_df(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        df.columns = [c.lower() for c in df.columns]
        if "time" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"time": "date"})
        df = df.sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _linreg_slope(series: pd.Series, window: int) -> pd.Series:
        # pandas-ta linreg supports slope=True
        return ta.linreg(series, length=window, slope=True)

    @staticmethod
    def get_mode(df_in: pd.DataFrame, config: dict) -> np.ndarray:
        df = TrendTrading._prepare_df(df_in)

        trend_lb = int(config["trend_lb"])
        up_thr = float(config["uptrend_min_slope"])
        dn_thr = float(config["downtrend_min_slope"])  # usually negative

        slope = TrendTrading._linreg_slope(df["close"].shift(1), trend_lb)

        mode = np.select(
            [slope > up_thr, slope < dn_thr],
            [1, -1],
            default=0,
        ).astype(int)

        return mode

    # @staticmethod
    # def get_mode(df_in: pd.DataFrame, config: dict) -> np.ndarray:
    #     df = TrendTrading._prepare_df(df_in)

    #     # Extract parameters from config (with defaults if missing)
    #     adx_len = int(config.get("adx_len", 14))
    #     sma_len = int(config.get("sma_len", 100))
    #     threshold = float(config.get("adx_threshold", 23))

    #     # Calculate ADX (requires high, low, close)
    #     # pandas_ta returns a DataFrame; we extract the specific ADX column
    #     adx_df = ta.adx(df["high"], df["low"], df["close"], length=adx_len)
        
    #     if adx_df is not None and not adx_df.empty:
    #         adx_col = f"ADX_{adx_len}"
    #         adx_series = adx_df[adx_col]
    #     else:
            
    #         adx_series = pd.Series(0, index=df.index)

        
    #     sma_adx = ta.sma(adx_series, length=sma_len)
    #     mode = np.where(adx_series > threshold, 0,0).astype(int)

    #     return mode

    @staticmethod
    def run(df_in: pd.DataFrame, config: dict) -> np.ndarray:
        return TrendTrading.get_mode(df_in, config).astype(float)


class SidewayGridTrading:
    """
    Sideway mode:
      - anchor = BB middle line
      - 3 grids above and 3 grids below
      - short the upside swing with weights [0.5, 0.3, 0.2]
      - long the downside swing with weights [0.5, 0.3, 0.2]
      - close all at anchor
      - stop loss:
          short -> 4th grid above anchor
          long  -> 5th grid below anchor
    """

    WEIGHTS = np.array([0.5, 0.3, 0.2], dtype=float)

    @staticmethod
    def run(df_in: pd.DataFrame, config: dict, trend_mode: np.ndarray) -> np.ndarray:
        df = TrendTrading._prepare_df(df_in)

        bb_lb = int(config["bb_lb"])

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float) if "high" in df.columns else close.copy()
        low = df["low"].to_numpy(dtype=float) if "low" in df.columns else close.copy()

        # Bollinger reference:
        # - anchor = rolling mean (middle band)
        # - step = rolling std
        bb_mid = df["close"].rolling(bb_lb).mean().shift(1).to_numpy(dtype=float)
        bb_step = df["close"].rolling(bb_lb).std().shift(1).to_numpy(dtype=float)

        n = len(df)
        pos = np.zeros(n, dtype=float)

        # Sideway state
        grid_side = 0      # 0 flat, -1 short grid, +1 long grid
        grid_level = 0     # 0..3
        grid_anchor = np.nan

        for i in range(n):
            mode = int(trend_mode[i])

            # Trend regime: sideway is off
            if mode != 0:
                grid_side = 0
                grid_level = 0
                grid_anchor = np.nan
                pos[i] = 0.0
                continue

            # Sideway regime: initialize anchor once
            if np.isnan(grid_anchor):
                grid_anchor = bb_mid[i]
                if np.isnan(grid_anchor):
                    grid_anchor = close[i]

            if np.isnan(bb_step[i]) or np.isnan(grid_anchor) or bb_step[i] <= 0:
                pos[i] = 0.0
                continue

            anchor = grid_anchor
            step = bb_step[i]

            # 3 entry levels plus stop levels
            upper = anchor + step * np.arange(1, 6)  # +1 ... +5
            lower = anchor - step * np.arange(1, 6)  # -1 ... -5

            # Enter only if flat
            if grid_side == 0:
                if close[i] >= upper[0]:
                    grid_side = -1
                    grid_level = 1
                elif close[i] <= lower[0]:
                    grid_side = 1
                    grid_level = 1

            # Short the upside swing
            if grid_side == -1:
                # Stop loss at 4th grid above anchor
                if high[i] >= upper[3]:
                    grid_side = 0
                    grid_level = 0
                    grid_anchor = np.nan
                    pos[i] = 0.0
                    continue

                # Add short layers as price keeps rising
                if close[i] >= upper[2]:
                    grid_level = max(grid_level, 3)
                elif close[i] >= upper[1]:
                    grid_level = max(grid_level, 2)
                elif close[i] >= upper[0]:
                    grid_level = max(grid_level, 1)

                # Close all when price falls back to anchor
                if close[i] <= anchor:
                    grid_side = 0
                    grid_level = 0
                    grid_anchor = np.nan
                    pos[i] = 0.0
                    continue

                pos[i] = -SidewayGridTrading.WEIGHTS[:grid_level].sum()
                continue

            # Long the downside swing
            if grid_side == 1:
                # Stop loss at 5th grid below anchor
                if low[i] <= lower[4]:
                    grid_side = 0
                    grid_level = 0
                    grid_anchor = np.nan
                    pos[i] = 0.0
                    continue

                # Add long layers as price keeps falling
                if close[i] <= lower[2]:
                    grid_level = max(grid_level, 3)
                elif close[i] <= lower[1]:
                    grid_level = max(grid_level, 2)
                elif close[i] <= lower[0]:
                    grid_level = max(grid_level, 1)

                # Close all when price rises back to anchor
                if close[i] >= anchor:
                    grid_side = 0
                    grid_level = 0
                    grid_anchor = np.nan
                    pos[i] = 0.0
                    continue

                pos[i] = SidewayGridTrading.WEIGHTS[:grid_level].sum()
                continue

            pos[i] = 0.0

        return pos


        



