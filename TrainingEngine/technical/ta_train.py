import ast
import re
from typing import List, Tuple
import copy
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from IPython.display import clear_output

from TrainingEngine.utils.data_split import TrainTestSplit, WalkForwardSplit
from Backtest import FinanceMetrics, ZeroPosError


@dataclass
class Metric:
    metric_map = {
        "sharpe": lambda bt: bt.Sharpe_after_fee()[0],
        "sortino": lambda bt: bt.Sharpe_after_fee()[1],
        "volatility": lambda bt: bt.Sharpe_after_fee()[2],
        "calmar": lambda bt: bt.Calmar(),
        "cagr": lambda bt: bt.Return()[2],
        "mdd": lambda bt: bt.MDD()[1],
        "return": lambda bt: bt.Return()[1],
        "hitrate": lambda bt: bt.Hitrate()[2],
        "total_trades": lambda bt: bt.Total_Trade()[0] + bt.Total_Trade()[1],
        "profit": lambda bt: bt.Profit()[1]
    }

    @classmethod
    def _normalize_metric_name(cls, metric_name: str) -> str:
        return re.sub(r"\s+", "", metric_name.lower())

    @classmethod
    def _safe_eval(cls, expr: str, bt) -> float:
        value_map = {name: func(bt) for name, func in cls.metric_map.items() if name != "custom"}
        value_map.update({"abs": abs, "min": min, "max": max})

        safe_expr = expr
        replacements = {}
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr):
            token = match.group(0)
            if token in value_map:
                placeholder = f"__metric_{len(replacements)}__"
                replacements[placeholder] = value_map[token]
                safe_expr = safe_expr.replace(token, placeholder, 1)

        tree = ast.parse(safe_expr, mode="eval")

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id not in replacements:
                    raise ValueError(f"Unsupported metric token: {node.id}")
                return replacements[node.id]
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                operand = _eval(node.operand)
                return operand if isinstance(node.op, ast.UAdd) else -operand
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    return left ** right
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                func = replacements.get(node.func.id)
                if func is None:
                    raise ValueError(f"Unsupported function: {node.func.id}")
                args = [_eval(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

        return float(_eval(tree))

    @classmethod
    def score(cls, metric_name: str, bt) -> float:
        normalized = cls._normalize_metric_name(metric_name)

        if normalized in cls.metric_map:
            return float(cls.metric_map[normalized](bt))

        if "*" in normalized or "+" in normalized or "-" in normalized or "/" in normalized or "**" in normalized:
            return cls._safe_eval(normalized, bt)

        raise KeyError(f"Unknown metric: {metric_name}")


class NoImproveStop:
    def __init__(self, max_patience):
        self.patience = max_patience
        self.best_vals = None
        self.best_trial = None
        self.last_step = 0

    def __call__(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_vals = trial.value
            self.best_trial = trial.number

        # Only applies Max_patience after 600 folds
        current_step = trial.number // 600

        if current_step > self.last_step:
            self.patience = int(self.patience * 0.4) 
            self.last_step = current_step

        if self.best_trial is not None:
            if trial.number - self.best_trial >= self.patience:
                print(f"\033[93mTraining stopped. No improvement for {self.patience} trials.\033[0m")
                study.stop()


class TrainTA:
    def __init__(self,
                 alpha: None,
                 config: dict,
                 oos_ratio: float = 0.15,
                 w4w_val_ratio: float = 0.15,
                 w4w_gap: int = 0,
                 n_fold: int = 5,
                 n_trials: int = 110,
                 opt_dir: str = 'maximize',
                 opt_metric: str = 'sharpe'):

        self.class_alpha = alpha
        self.config = config
        self.oos_ratio = oos_ratio
        self.w4w_val_ratio = w4w_val_ratio
        self.w4w_gap = w4w_gap
        self.n_fold = n_fold
        self.n_trials = n_trials
        self.opt_dir = opt_dir
        self.opt_metric = opt_metric

    def start_training(self, data:pd.DataFrame, param_range:dict):
        is_list, _ = self._split_data(data)
        best_params = self._optimize(is_list, param_range)
        return best_params


    # ----- HELPER --------

    # Split data
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame | List[pd.DataFrame]]:
        is_df, os_df = TrainTestSplit(test_size=self.oos_ratio).split(data)
        is_list = WalkForwardSplit(test_size=self.w4w_val_ratio, gap=self.w4w_gap, k_fold=self.n_fold).split(is_df)
        return is_list, os_df

    # Optimize
    def _optimize(
        self,
        is_data_list: List[pd.DataFrame],
        param_range: dict
    ) -> dict:
        
        fold_results = []

        def _suggest_from_spec(trial, name: str, spec, default_value=None):
            if spec is None:
                return default_value

            if isinstance(spec, list):
                return trial.suggest_categorical(name, spec)

            if isinstance(spec, tuple):
                if len(spec) not in (2, 3):
                    raise ValueError(f"{name}: tuple is only for Int or Float, and must have length 2 or 3.")

                low, high = spec[0], spec[1]
                low, high = min(low, high), max(low, high)
                step = spec[2] if len(spec) == 3 else None

                kwargs = {}
                if step is not None:
                    kwargs["step"] = step

                if any(isinstance(x, float) for x in (low, high, step) if x is not None):
                    return trial.suggest_float(name, float(low), float(high), **kwargs)
                return trial.suggest_int(name, int(low), int(high), **kwargs)

            raise TypeError(f"{name}: str or bool must be in List, int or float in Tuple")


        def report_best_trial(study, trial):
            clear_output(wait=True)
            print(f"Current trial {trial.number + 1}/{self.n_trials}")
            if study.best_trials:
                best_trial = study.best_trial
                print("Best trial:")
                print(f"- Trial {best_trial.number}")
                print(f"- {self.opt_metric.title()} {best_trial.value:.5f}")
                print(f"- Parameters: {json.dumps(best_trial.params)}")


        def objective(trial):

            current_params = self.config['alpha_cfg']['params']
            params = current_params.copy()

            for name, spec in param_range.items():
                params[name] = _suggest_from_spec(
                    trial=trial,
                    name=name,
                    spec=spec,
                    default_value=current_params.get(name),
                )

            config = copy.deepcopy(self.config)
            config["alpha_cfg"]["params"] = params

            alpha = self.class_alpha(config["alpha_cfg"]["params"])

            scores = []
            for fold_df in is_data_list:
                df = fold_df.copy()
                if isinstance(df.index, pd.RangeIndex):
                    df = df.reset_index()

                pos = alpha.run(df)
                if isinstance(pos, pd.DataFrame):
                    raise ValueError("Check your alpha .py file. Alpha output must be array or Series, not DataFrame")
                df["position"] = np.asarray(pos)

                try:
                    bt = FinanceMetrics(df=df, **self.config["bt_cfg"])
                    score_fold = Metric.score(self.opt_metric, bt)
                except ZeroPosError:
                    score_fold = np.nan
                scores.append(score_fold)

            if not scores or any(pd.isna(s) for s in scores):
                return -9999

            score = np.asarray(scores)
            mean_score = np.mean(score)
            median_score = np.median(score)

            # Median Absolute Deviation (MAD)
            penalty = np.mean(np.abs(score - median_score))

            if self.opt_dir == "maximize":
                score = mean_score - 0.65 * penalty
                fold_results.append(score)
                return score
            else:
                score = mean_score + 0.65 * penalty
                fold_results.append(score)
                return score


        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=self.opt_dir)

        # In-tune print
        study.optimize(objective,
            n_trials=self.n_trials,
            callbacks=[report_best_trial, NoImproveStop(max_patience=700)],
        )

        # Summary print
        fold_results = np.array(fold_results)
        fold_results = fold_results[fold_results > -100]
        training_results = pd.Series(fold_results[fold_results>-100], name=self.opt_metric)
        tab = pd.DataFrame(training_results.describe())
        tab.loc['median'] = training_results.median()
        tab.loc['skew'] = training_results.skew()
        tab.loc['ex_kurt'] = training_results.kurt()

        print("\nTraining summary")
        print(tab)
        plt.figure(figsize=(8, 4.5))
        plt.hist(training_results, bins=10, color="#2f56c1", edgecolor='none')

        plt.title(f"Distribution of {self.opt_metric}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(False)

        plt.axvline(x=0, color="#ff471e", linestyle='--')
        plt.plot([], [], linestyle='none', label=f"Mean: {training_results.mean():.3f}")
        plt.plot([], [], linestyle='none', label=f"Median: {training_results.median():.3f}")
        plt.plot([], [], linestyle='none', label=f"Stdev: {training_results.std():.3f}")
        

        plt.legend()
        plt.tight_layout()
        plt.show()

        # Save best params
        best_params = self.config['alpha_cfg']['params'].copy()
        best_params.update(study.best_params)
        return best_params
    
# python -m TrainingEngine.technical.ta_train
if __name__ == '__main__':
    print(TrainTA().alpha)
