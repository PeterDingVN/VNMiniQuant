import pandas as pd
import numpy as np


class DonchianBreakout:

    def __init__(self, config: dict):
        self.cfg = config
    
    def run(self, df_):
        don_lookback = self.cfg["don_lookback"]
        ema_lookback=self.cfg["ema_lookback"]
        atr_lookback=self.cfg["atr_lookback"]
        long_atr_mult=self.cfg["long_atr_mult"]
        short_atr_mult=self.cfg["short_atr_mult"]

        df = df_.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.set_index('datetime')

        # =========================================================
        # DAILY DATA
        # =========================================================
        daily = (
            df[['open', 'high', 'low', 'close']]
            .resample('1D')
            .agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })
        )

        # Daily EMA
        daily['ema'] = (
            daily['close']
            .ewm(span=ema_lookback, adjust=False)
            .mean()
        )

        # Daily trend filters
        daily['bull'] = daily['close'] > daily['ema']
        daily['bear'] = daily['close'] < daily['ema']

        # =========================================================
        # MAP DAILY FILTER BACK TO 30M
        # =========================================================
        df['daily_bull'] = (
            daily['bull']
            .shift(1)
            .reindex(df.index, method='ffill').astype(bool)
            .fillna(False)
        )

        df['daily_bear'] = (
            daily['bear']
            .shift(1)
            .reindex(df.index, method='ffill').astype(bool)
            .fillna(False)
        )

        # =========================================================
        # DONCHIAN CHANNEL
        # =========================================================
        df['don_upper'] = (
            df['high']
            .rolling(don_lookback)
            .max()
            .shift(1)
        )

        df['don_lower'] = (
            df['low']
            .rolling(don_lookback)
            .min()
            .shift(1)
        )

        # =========================================================
        # ATR
        # =========================================================
        prev_close = df['close'].shift(1)

        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df['atr'] = tr.rolling(atr_lookback).mean()

        # =========================================================
        # ARRAYS
        # =========================================================
        long_pos = np.zeros(len(df))
        short_pos = np.zeros(len(df))

        long_in_position = False
        short_in_position = False

        long_stop = np.nan
        short_stop = np.nan

        close = df['close'].values
        upper = df['don_upper'].values
        lower = df['don_lower'].values
        atr = df['atr'].values

        daily_bull = df['daily_bull'].values
        daily_bear = df['daily_bear'].values

        # =========================================================
        # MAIN LOOP
        # =========================================================
        for i in range(len(df)):

            # skip invalid rows
            if (
                np.isnan(close[i]) or
                np.isnan(upper[i]) or
                np.isnan(lower[i]) or
                np.isnan(atr[i])
            ):
                continue

            # =====================================================
            # LONG LOGIC
            # =====================================================
            if not long_in_position:

                # ENTRY
                if (
                    daily_bull[i] and
                    close[i] > upper[i]
                ):
                    long_in_position = True
                    long_stop = close[i] - long_atr_mult * atr[i]

            else:

                # UPDATE TRAILING STOP
                new_stop = close[i] - long_atr_mult * atr[i]
                long_stop = max(long_stop, new_stop)

                # EXIT CONDITIONS
                exit_long = False

                # ATR stop
                # if close[i] < long_stop:
                #     exit_long = True

                # trend reversal + lower breakdown
                if (
                    (not daily_bull[i]) and
                    close[i] < lower[i]
                ):
                    exit_long = True

                if exit_long:
                    long_in_position = False
                    long_stop = np.nan

            long_pos[i] = 1 if long_in_position else 0

            # =====================================================
            # SHORT LOGIC
            # =====================================================
            if not short_in_position:

                # ENTRY
                if (
                    daily_bear[i] and
                    close[i] < lower[i]
                ):
                    short_in_position = True
                    short_stop = close[i] + short_atr_mult * atr[i]

            else:

                # UPDATE TRAILING STOP
                new_stop = close[i] + short_atr_mult * atr[i]
                short_stop = min(short_stop, new_stop)

                # EXIT CONDITIONS
                exit_short = False

                # ATR stop
                if close[i] > short_stop:
                    exit_short = True

                # daily EMA reversal
                if not daily_bear[i]:
                    exit_short = True

                if exit_short:
                    short_in_position = False
                    short_stop = np.nan

            short_pos[i] = -1 if short_in_position else 0


        pos = long_pos + short_pos

        # clean NaNs
        pos = np.nan_to_num(pos, nan=0.0)
        df['position'] = pos

        df_ = df.reset_index()[['datetime', 'close', 'position']]

        return pos
    


