import numpy as np
import pandas as pd
from typing import List


class MonteCarlosPermutation:

    """
    Summary:
        The class is used to produce N different permutations, used for testing if trade strategy is not pure luck
        The permutation preserve start and end points to ensure trend is retained
        However, dependency or serial corr cannot be retained
        Plus, stat properties change across folds (none same as original data)
        In case of multiple datasets input -> they will be mutated together so correlation retained among them.

    Main func: MonteCarlosPermutation.gen_permutation()
    ---------
    Params:
    ---------
    ohlc: pd.DataFrame | List[pd.DataFrame]
    start_index: where the data start -> default to 0 -> this makes the 2nd (index=1) to be the first index mutated
    end_index: where the data stop permutating -> default to 1 -> stop permutating at the last data point
    seed: random seed to reproduce result

    Output:
    ---------
    Dataframe or A list of dataframe being mutated.

    """
    
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
    
    
    # MAIN FUNC: Generate permutation
    @staticmethod
    def gen_permutation(ohlc: pd.DataFrame | List[pd.DataFrame],
                    start_index: int = 0,
                    end_index: int = 1,
                    seed: float = None):

        # ---------------
        #   Validation
        # ---------------

        # _____ Start index must be at least 0, end must be 1 min _____
        assert start_index >= 0
        assert end_index >= 1

        # ______ Normalize input to List __________
        if isinstance(ohlc, pd.DataFrame):
            ohlcs = [ohlc.copy()]
        else:
            ohlcs = [df.copy() for df in ohlc]

        # ______ Verify matching indices _________
        MonteCarlosPermutation._check_index(ohlcs[0])
        base_index = ohlcs[0].reset_index().set_index('time').index

        data_order = []
        for i, df in enumerate(ohlcs):
            try:
                MonteCarlosPermutation._check_index(df)
                df = df.reset_index().set_index('time')
                if not base_index.equals(df.index):
                    raise ValueError("Indexes do not match across inputs")
                
                data_order.append(i)
            except Exception as e:
                return f'Error {e} is caught at data number {i+1} in the list'


        # --------------------------
        #        MAin FUnc
        # --------------------------

        # _____ Key params _____
        np.random.seed(seed)

        n_markets = len(ohlcs)
        n_bars = len(base_index)

        perm_index = start_index + 1
        perm_n = n_bars - perm_index 


        # ______ Calculate "relative" to remain most structure of data after perm ______
        # _______ But dependency, serial corr, ... cannot be retained __________________

        # Stack into (n_markets, T, 5) - all markets have same column order
        log_arr = np.log(np.stack([df[['open', 'high', 'low', 'close', 'volume']].to_numpy() for df in ohlcs]))

        o, h, l, c, v = log_arr[:, :, 0], log_arr[:, :, 1], log_arr[:, :, 2], log_arr[:, :, 3], log_arr[:, :, 4]

        # Open relative to prev close; volume relative to prev volume; high, low, close relative to open
        prev_v = np.concatenate([np.full((v.shape[0], 1), np.nan), v[:, :-1]], axis=1)
        r_v = v - prev_v

        prev_c = np.concatenate([np.full((c.shape[0], 1), np.nan), c[:, :-1]], axis=1)
        r_o = o - prev_c

        r_h = h - o
        r_l = l - o
        r_c = c - o

        # Slice: (n_markets, T - perm_index)
        relative_open  = r_o[:, perm_index:]
        relative_high  = r_h[:, perm_index:]
        relative_low   = r_l[:, perm_index:]
        relative_close = r_c[:, perm_index:]
        relative_volume = r_v[:, perm_index:]


        # ______________________ PERfrom Shuffling _________________________
        idx = np.arange(perm_n)

        # Shuffle intrabar relative values (high/low/close)
        perm1 = np.random.permutation(idx)
        relative_high = relative_high[:, perm1]
        relative_low = relative_low[:, perm1]
        relative_close = relative_close[:, perm1]
        relative_volume = relative_volume[:, perm1]

        # Shuffle last close to open (gaps) seprately
        perm2 = np.random.permutation(idx)
        relative_open = relative_open[:, perm2]


        # _______________________ ADD Relative into Log Price to get perm Price __________________________

        # Stack all markets into (n_markets, n_bars, 5) log-space array
        log_bars_all = np.log(
            np.stack([reg_bars[['open', 'high', 'low', 'close', 'volume']].to_numpy() for reg_bars in ohlcs])
        )

        # Init output array
        perm_bars_all = np.empty((n_markets, n_bars, 5))

        # Init start -> start bar
        perm_bars_all[:, :start_index+1, :] = log_bars_all[:, :start_index+1, :]

        # MAIN CALCULATION:
        # 1. Price calculation
        start_close   = perm_bars_all[:, start_index, 3]                  
        cumsum_delta  = np.cumsum(relative_open + relative_close, axis=1)                                
        shifted_cumsum = np.concatenate(
            [np.zeros((n_markets, 1)), cumsum_delta[:, :-1]], axis=1
        )                                                                               

        open_perm  = start_close + shifted_cumsum + relative_open
        close_perm = start_close + cumsum_delta
        high_perm  = open_perm + relative_high
        low_perm   = open_perm + relative_low

        perm_bars_all[:, perm_index:, 0] = open_perm
        perm_bars_all[:, perm_index:, 1] = high_perm
        perm_bars_all[:, perm_index:, 2] = low_perm
        perm_bars_all[:, perm_index:, 3] = close_perm

        # 2. Volume calculation
        start_log_volume = perm_bars_all[:, start_index, 4]   
        cumsum_vol = np.cumsum(relative_volume, axis=1)       
        log_volume_perm = start_log_volume[:, None] + cumsum_vol

        perm_bars_all[:, perm_index:, 4] = log_volume_perm


        # Revert end bar(s) to original values
        perm_bars_all[:, -end_index:, :] = log_bars_all[:, -end_index:, :]


        # __________________ COnvert back to Nominal Price ______________________
        # Exponentiate and wrap in DataFrames
        perm_bars_all = np.exp(perm_bars_all)
        perm_ohlc = [
            pd.DataFrame(perm_bars_all[i], index=base_index, columns=['open', 'high', 'low', 'close', 'volume'])
            for i in data_order
        ]

        return perm_ohlc if n_markets > 1 else perm_ohlc[0]



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

