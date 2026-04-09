import numpy as np
import pandas as pd
from typing import List
import warnings

class MonteCarlosPermutation:
    
    # validation method
    @staticmethod
    def _check_index(df: pd.DataFrame):
        if not isinstance (df.index, pd.DatetimeIndex) and not 'time' in df.columns:
            raise KeyError("Please add column 'time' to your data, or rename your datetime column to 'time'")


    # util method: get random box to shuffle in permutation process
    @staticmethod
    def _get_random_blocks(n):
        lengths = []
        total = 0
        p = 0.1  # e.g. avg block length ~10
        while total < n:
            L = np.random.geometric(p)
            if total + L > n:
                L = n - total
            lengths.append(L)
            total += L
        starts = np.cumsum([0] + lengths[:-1])
        return lengths, starts
    
    @staticmethod
    def gen_permutation(ohlc: pd.DataFrame | List[pd.DataFrame],
                    start_index: int = 1,
                    seed: float = None):

        # ---------------
        #   Validation
        # ---------------
        # Force index >= 1 to retain the first data point
        if start_index < 1:
            warnings.warn(f'start_index was forced set to 1 because your index is < 1', UserWarning)
            start_index = 1

        # Normalize input to List
        if isinstance(ohlc, pd.DataFrame):
            ohlcs = [ohlc.copy()]
        else:
            ohlcs = [df.copy() for df in ohlc]

        # Verify matching indices
        MonteCarlosPermutation._check_index(ohlcs[0])
        base_index = ohlcs[0].set_index('time').index

        for i, df in enumerate(ohlcs):
            try:
                MonteCarlosPermutation._check_index(df)
                df = df.set_index('time')
            except Exception as e:
                return f'Error {e} is caught at data number {i+1} in the list'
            if not base_index.equals(df.index):
                raise ValueError("Indexes do not match across inputs")


        # --------------------------
        #        MAin FUnc
        # --------------------------

        n_markets = len(ohlcs)
        n_bars = len(base_index)

        # If not enough bars to permute, return original
        if n_bars <= start_index + 2:
            return ohlc.copy() if n_markets == 1 else [df.copy() for df in ohlcs]

        # Require volume for the upgraded version
        for df in ohlcs:
            if 'volume' not in df.columns:
                raise ValueError("Input data must contain a 'volume' column.")

        perm_index = start_index + 1
        perm_n = n_bars - start_index - 2  # up to last-1

        # Seed for reproducibility
        np.random.seed(seed)

        # Convert to log-prices for all markets
        log_ohlc = [
            np.log(df[['open', 'high', 'low', 'close']].values.astype(float))
            for df in ohlcs
        ]
        volume_data = [df['volume'].values.copy() for df in ohlcs]

        # Save the preserved start and end bars
        start_log = np.stack([ld[start_index] for ld in log_ohlc])
        last_log = np.stack([ld[-1] for ld in log_ohlc])

        # Compute relative returns for each market
        rel_open = np.empty((n_markets, perm_n))
        rel_high = np.empty((n_markets, perm_n))
        rel_low = np.empty((n_markets, perm_n))
        rel_close = np.empty((n_markets, perm_n))

        for i, logs in enumerate(log_ohlc):
            # Gap (open - prior close)
            rel_open[i, 0] = logs[perm_index, 0] - logs[perm_index - 1, 3]
            for j in range(1, perm_n):
                rel_open[i, j] = logs[perm_index + j, 0] - logs[perm_index + j - 1, 3]

            # Intraday (high/low/close relative to open)
            slice_logs = logs[perm_index:perm_index + perm_n]
            rel_high[i] = slice_logs[:, 1] - slice_logs[:, 0]
            rel_low[i] = slice_logs[:, 2] - slice_logs[:, 0]
            rel_close[i] = slice_logs[:, 3] - slice_logs[:, 0]

        # Generate and shuffle blocks for intraday returns
        lengths_hl, starts_hl = MonteCarlosPermutation._get_random_blocks(perm_n)
        perm_order_hl = np.random.permutation(len(lengths_hl))

        # Generate and shuffle blocks for gap returns
        lengths_o, starts_o = MonteCarlosPermutation._get_random_blocks(perm_n)
        perm_order_o = np.random.permutation(len(lengths_o))

        # Apply block permutation to intraday returns
        rel_high_new = np.empty_like(rel_high)
        rel_low_new = np.empty_like(rel_low)
        rel_close_new = np.empty_like(rel_close)
        vol_new = np.empty((n_markets, perm_n), dtype=volume_data[0].dtype)

        pos = 0
        for b in perm_order_hl:
            L = lengths_hl[b]
            seg = slice(starts_hl[b], starts_hl[b] + L)
            rel_high_new[:, pos:pos + L] = rel_high[:, seg]
            rel_low_new[:, pos:pos + L] = rel_low[:, seg]
            rel_close_new[:, pos:pos + L] = rel_close[:, seg]
            for i in range(n_markets):
                vol_new[i, pos:pos + L] = volume_data[i][perm_index + starts_hl[b]:perm_index + starts_hl[b] + L]
            pos += L

        # Apply block permutation to gap returns
        rel_open_new = np.empty_like(rel_open)
        pos = 0
        for b in perm_order_o:
            L = lengths_o[b]
            seg = slice(starts_o[b], starts_o[b] + L)
            rel_open_new[:, pos:pos + L] = rel_open[:, seg]
            pos += L

        # Reconstruct log-prices from permuted returns (vectorized)
        permuted = []
        for i in range(n_markets):
            base_close = start_log[i, 3]

            # Build array A whose cumsum gives log(open prices)
            A = np.empty(perm_n)
            A[0] = base_close + rel_open_new[i, 0]
            A[1:] = rel_close_new[i, :-1] + rel_open_new[i, 1:]
            open_seq = np.cumsum(A)

            # Compute high/low/close from open and intraday moves
            high_seq = open_seq + rel_high_new[i]
            low_seq = open_seq + rel_low_new[i]
            close_seq = open_seq + rel_close_new[i]

            # Assemble final log-price matrix
            perm_log = np.empty((n_bars, 4))
            perm_volume = np.empty(n_bars, dtype=vol_new.dtype)

            # Preserve initial segment
            perm_log[:start_index + 1] = log_ohlc[i][:start_index + 1]
            perm_volume[:start_index + 1] = volume_data[i][:start_index + 1]

            # Fill permuted portion
            perm_log[perm_index:perm_index + perm_n, 0] = open_seq
            perm_log[perm_index:perm_index + perm_n, 1] = high_seq
            perm_log[perm_index:perm_index + perm_n, 2] = low_seq
            perm_log[perm_index:perm_index + perm_n, 3] = close_seq
            perm_volume[perm_index:perm_index + perm_n] = vol_new[i]

            # Preserve final bar
            perm_log[-1] = last_log[i]
            perm_volume[-1] = volume_data[i][-1]

            # Exponentiate OHLC and pack into DataFrame
            out = pd.DataFrame(
                np.exp(perm_log),
                columns=['open', 'high', 'low', 'close']
            )
            out['volume'] = perm_volume

            # Put datetime back as a column
            out.insert(0, 'time', base_index.to_numpy())

            permuted.append(out)

        return permuted[0] if n_markets == 1 else permuted



# =======================
#       TEST CASE
# =======================

if __name__ == '__main__':

    from data.stock_price import AccessData
    agr = AccessData(symbol='AGR').access_data()
    print(agr.describe())

    print('='*50, 'PERM DES','='*50)
    for i in range(3):
        new_agr = MonteCarlosPermutation.gen_permutation(agr)
        print(new_agr.index.name)
        print(new_agr.describe())


# Run cmd: python -m strategy_backtest.utils.mcpt

