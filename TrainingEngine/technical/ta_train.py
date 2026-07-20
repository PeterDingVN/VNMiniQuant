import ast
import re
from typing import List, Tuple
import copy
from dataclasses import dataclass

import pandas as pd
import numpy as np
import optuna

from TrainingEngine.utils.data_split import TrainTestSplit, WalkForwardSplit
from Backtest import FinanceMetrics, ZeroPosError
from AlphaBase import AlphaBase



@dataclass
class Metric:
    metric_map = {
        "sharpe": lambda bt: bt.Sharpe_after_fee()[0],
        "sortino": lambda bt: bt.Sharpe_after_fee()[1],
        "calmar": lambda bt: bt.Calmar(),
        "cagr": lambda bt: bt.Return()[2],
        "mdd": lambda bt: bt.MDD()[1],
        "return": lambda bt: bt.Return()[1],
        "hitrate": lambda bt: bt.Hitrate()[2],
        "total_trades": lambda bt: bt.Total_Trade()[0] + bt.Total_Trade()[1],
        "profit": lambda bt: bt.Profit()[1],
        "custom": lambda bt: None,
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
    def score(cls, metric_name: str, bt, expr: str | None = None) -> float:
        if expr is not None:
            return cls._safe_eval(expr, bt)

        normalized = cls._normalize_metric_name(metric_name)
        if normalized == "custom":
            if expr is None:
                raise ValueError("Custom metric requires an expression.")
            return cls._safe_eval(expr, bt)

        if normalized in cls.metric_map:
            return float(cls.metric_map[normalized](bt))

        if "*" in normalized or "+" in normalized or "-" in normalized or "/" in normalized or "**" in normalized:
            return cls._safe_eval(normalized, bt)

        raise KeyError(f"Unknown metric: {metric_name}")



class TrainTA(AlphaBase):
    def __init__(self,
                 oos_ratio: float = 0.15,
                 w4w_val_ratio: float = 0.15,
                 w4w_gap: int = 0,
                 n_fold: int = 5,
                 n_trials: int = 110,
                 opt_dir: str = 'maximize',
                 opt_metric: str = 'sharpe'):
        
        super().__init__()
        
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

        # overwrite cfg
        cfg = self._load_config()
        cfg['alpha_cfg']['params'] = best_params
        self._dump_config(cfg)



    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame | List[pd.DataFrame]]:
        is_df, os_df = TrainTestSplit(test_size=self.oos_ratio).split(data)
        is_list = WalkForwardSplit(test_size=self.w4w_val_ratio, gap=self.w4w_gap, k_fold=self.n_fold).split(is_df)
        return is_list, os_df


    def _optimize(
        self,
        is_data_list: List[pd.DataFrame],
        param_range: dict
    ) -> dict:

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

        def objective(trial):
            current_params = self._construct_params_set()
            params = {}

            # Sample every parameter from either param_range or default value
            all_param_names = set(current_params) | set(param_range)

            for name in all_param_names:
                if name in param_range:
                    params[name] = _suggest_from_spec(
                        trial=trial,
                        name=name,
                        spec=param_range[name],
                        default_value=current_params.get(name),
                    )
                else:
                    default_value = current_params[name]

                    if isinstance(default_value, bool):
                        params[name] = trial.suggest_categorical(name, [default_value])
                    elif isinstance(default_value, str):
                        params[name] = trial.suggest_categorical(name, [default_value])
                    elif isinstance(default_value, int):
                        params[name] = trial.suggest_int(name, default_value, default_value)
                    elif isinstance(default_value, float):
                        params[name] = trial.suggest_float(name, default_value, default_value)
                    else:
                        params[name] = default_value

            config = copy.deepcopy(self.config)
            config["alpha_cfg"]["params"] = params

            alpha = self.class_alpha(config["alpha_cfg"]["params"])

            scores = []
            for fold_df in is_data_list:
                df = fold_df.copy()

                pos = alpha.run(df)
                df["position"] = np.asarray(pos)

                try:
                    bt = FinanceMetrics(df=df, **self.config["bt_cfg"])
                    score_fold = Metric.score(self.opt_metric, bt)
                except ZeroPosError:
                    score_fold = np.nan
                scores.append(score_fold)

            if not scores or any(pd.isna(s) for s in scores):
                return -9999

            return float(np.mean(scores))

        study = optuna.create_study(direction=self.opt_dir)
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params
    
# python -m TrainingEngine.technical.ta_train
if __name__ == '__main__':
    print(TrainTA().alpha)
