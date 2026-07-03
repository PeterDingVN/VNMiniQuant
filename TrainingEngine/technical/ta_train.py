# Expose data config
# Split data Train Test
# Train with w4w and record the best param
# Overwrite config
# Backtest on Outsample alongside Insample
# Automate trial.suggest part? (simplify at best -> auto detect int/float/cat -> let user define range)


class TrainTA:
    def __init__(self,
                 oos_ratio: float = 0.15,
                 w4w_val_ratio: float = 0.15,
                 n_fold: int = 5,
                 n_trials: int = 110,
                 opt_dir: str = 'maximize',
                 opt_metric: str = 'sharpe'):
        
        

class TrainML:
    pass