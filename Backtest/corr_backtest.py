from .finance_backtest import FinanceMetrics
from pathlib import Path

import glob
import os
import pandas as pd


class CorrTest:
    def __init__(self, config):

        self.config_for_bt = config
        self.folder = Path(__file__).resolve().parent.parent/ "Alpha_Repo" / self.config_for_bt ['fee_type']

    def check_alpha_corr(self, df: pd.DataFrame):
        mas = df.copy()
        mas.columns = [c.lower() for c in mas.columns]

        if not all(col in mas.columns for col in ['datetime', 'position', 'close']):
            raise ValueError("Input data must have datetime, position, close")
        
        mas["datetime"] = pd.to_datetime(mas["datetime"])

        if 'gain_after_fee' not in mas.columns:
            mas = FinanceMetrics(df=mas, **self.config_for_bt).df
            mas = mas.reset_index()
            mas = mas[['datetime', 'gain_after_fee']]
        elif 'gain_after_fee' in mas.columns:
            mas = mas.reset_index()
            mas = mas[['datetime', 'gain_after_fee']]


        mas = (mas.set_index("datetime")
            .resample("D")
            .sum(min_count=1)
            .dropna())
        

        results = []

        for file in sorted(glob.glob(os.path.join(self.folder, "*.csv"))):
            try:
                df = pd.read_csv(file)
                df.columns = [c.lower() for c in df.columns]

                if not all(col in df.columns for col in ['datetime', 'position', 'close']):
                    raise ValueError(f"Data {file} must have all: datetime, position, close")
                df["datetime"] = pd.to_datetime(df["datetime"])

                if 'gain_after_fee' not in df.columns:
                    df = FinanceMetrics(df=df, **self.config_for_bt).df
                    df = df.reset_index()
                    df = df[['datetime', 'gain_after_fee']]
                elif 'gain_after_fee' in df.columns:
                    df = df.reset_index()
                    df = df[['datetime', 'gain_after_fee']]

                df = (df.set_index("datetime").resample("D").sum(min_count=1).dropna())

                # ----------  Start calculating Corr  ----------
                merged = df[["gain_after_fee"]].join(mas[["gain_after_fee"]],
                    how="inner",
                    lsuffix="_alpha",
                    rsuffix="_mas")

                corr = merged["gain_after_fee_alpha"].corr(merged["gain_after_fee_mas"])
                corr_sp = merged["gain_after_fee_alpha"].corr(merged["gain_after_fee_mas"], method='spearman')

                results.append({
                    "Alpha": os.path.splitext(os.path.basename(file))[0],
                    "Pearson": corr,
                    "Spearman": corr_sp
                    })


            except Exception as e:
                raise Exception(f"Error: {e}")


        result_df = pd.DataFrame(results)
        if result_df.empty:
            RED = "\033[91m"
            RESET = "\033[0m"
            print(f"{RED}NO DATA FOR CORR CHECK{RESET}")
            return

        result_df = (
            result_df
            .sort_values("Pearson", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        result_df["Pearson"] = result_df["Pearson"].map(lambda x: f"{x:.4f}")
        result_df["Spearman"] = result_df["Spearman"].map(lambda x: f"{x:.4f}")

        # Output table
        table = result_df.to_string(index=False, col_space=9)
        lines = table.splitlines()
        separator = "-" * len(lines[0])

        print(lines[0])
        print(separator)
        print("\n".join(lines[1:]))