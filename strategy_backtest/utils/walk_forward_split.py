import numpy as np
import warnings


class WalkForwardSplit:
    """
    Walk-forward splits the raw input data into K different datasets, each has train and test sets.
    Its only usage is to split raw big data into K chunks, no internal splitting (train-val split) 
        for each fold is performed.
    The aim is to provide data for walk forward performance evaluation Financially and Statistically.


    Parameters
    ----------
    test_size : float (default=0.2)
        Fraction of each fold used as test

    k_fold : int
        Number of folds

    Output
    ---------
    A list of dataframe.
    """

    def __init__(self, test_size=0.2, k_fold=5):
        self.test_size = test_size
        self.k_fold = int(k_fold)

    def split(self, data):
        self._validate_inputs(data)

        n = len(data)

        # Train and test size of each window (each fold) 
        # -> only use for validation not a real train-test split
        train_len = n / (1 + self.k_fold * self.test_size / (1-self.test_size))
        test_len = train_len * self.test_size / (1-self.test_size)
        self._warn_small_folds(train_len, test_len)

        # derive window size from k_fold
        window_size = int(train_len + test_len)

        # ensure no overlap of test sets, default step-size = test-len instead of >
        step_size = int(test_len)


        splits = []
        start = 0
        for i in range(self.k_fold):
            end = start + window_size
            if i+1 == self.k_fold:
                end = n
                
            slice_ = data.iloc[start:end+1]
            splits.append(slice_)

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
                f"Data total size ({int(train_len+test_len)}) is small with train {int(train_len)}" 
                f"and {int(test_len)} → noisy evaluation",
                UserWarning
            )


# ----------------------
#       TEST CASE
# ----------------------
if __name__ == '__main__':
    from data_api.stock_price import AccessData
    agr = AccessData(symbol='AGR').access_data()
    agr_ls = WalkForwardSplit(k_fold=20, test_size=0.5).split(data=agr)

    print(agr_ls[-3].tail(), len(agr_ls[-2]))
    print(agr_ls[-2].iloc[-125:, :].head(), len(agr_ls[-1]))


# Run cmd: python -m strategy_backtest.utils.walk_forward_split
