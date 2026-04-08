import numpy as np
import warnings


class WalkForwardSplit:
    """
    Walk-forward splitter where EACH fold has its own train/test split.

    Parameters
    ----------
    test_size : float (default=0.2)
        Fraction of each fold used as test

    k_fold : int
        Number of folds

    Output
    ---------
    A list of dataframe each nested inside a tuple
    """

    def __init__(self, test_size=0.2, k_fold=5):
        self.test_size = test_size
        self.k_fold = int(k_fold)

    def split(self, data):
        self._validate_inputs(data)

        n = len(data)

        # derive window size from k_fold
        window_size = n // self.k_fold

        # Train and test size of each window (each fold)
        test_len = int(window_size * self.test_size)
        train_len = window_size - test_len
        self._warn_small_folds(train_len, test_len)

        # ensure no overlap of test sets, default step-size = test-len instead of >
        step_size = test_len
        

        splits = []

        start = 0

        for i in range(self.k_fold):
            end = start + window_size

            if end > n:
                break

            window = data.iloc[start:end]

            train_data = window.iloc[:train_len]
            test_data = window.iloc[train_len:]

            splits.append((train_data, test_data))

            start += step_size  # move forward

        return splits

    # -------------------------
    # Validation logic
    # -------------------------

    def _validate_inputs(self, data):
        if data is None or len(data) == 0:
            raise ValueError("Data must be non-empty")

        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1")

        if self.k_fold <=0:
            raise ValueError("k_fold must be positive")


    def _warn_small_folds(self, train_len, test_len):
        if test_len + train_len < 500:
            warnings.warn(
                f"Data total size ({train_len+test_len}) is small with train {train_len} and {test_len} → noisy evaluation",
                UserWarning
            )


# ----------------------
#       TEST CASE
# ----------------------
if __name__ == '__main__':
    from data.stock_price import AccessData
    agr = AccessData(symbol='AGR').access_data()
    agr_ls = WalkForwardSplit(k_fold=20, test_size=0.6).split(data=agr)
    print(len(agr_ls))
    print(type(agr_ls[0][0]))

# Run cmd: python -m strategy_backtest.utils.walk_forward_split
