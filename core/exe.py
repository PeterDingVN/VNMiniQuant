from data import AccessData

from strategy_backtest import MonteCarlosPermutation, WalkForwardSplit, StatTest, TrainTestSplit, FinanceTest
import numpy as np
import pandas as pd
from typing import Dict
from config import SysConfig, DonchianCfg

from strategy_sample import DonchianBreakout


class SystemExecute:

    """
    Under dev: 
    - A single config files gathering every run so you can PLUG AND PLAY ANY STRAT INSTANTLY
    - Future update: MULTIPLE ASSET PERMUTATIONS and TRAINING
    
    """

    def __init__(self,
                 strategy: None,
                 config = SysConfig
            ):

        self.cfg = config
        self.strategy = strategy

        assert self.strategy != None
    # ---------------
    #      MAIN 
    # ---------------
    def execute(self, asset: str, asset_type: str) -> Dict: # --> Future update needs stock = list not str

        # Read Data
        data_stock = AccessData(symbol=asset).access_data(purpose='train')[0]['data'] # -> need update to run multiple assets

        if data_stock.empty:
            raise ValueError(f"Data for {asset} is empty")
        elif not isinstance(data_stock, pd.DataFrame):
            raise ValueError("AccessData auto return data(s) in a dict -> take data by data_stock[<data_id>]['data']")

       
        # IS/OOS
        ins, oos = TrainTestSplit(
                                test_size=self.cfg.oos_testsize, 
                                has_lookahead=False,
                                lookahead=0
                                ).split(data_stock)
        

        # W4W Training
        print("START W4W TRAINING")
        pf_w4w_is = self._walkforward_train(strategy=self.strategy,  # --> Need update to generate permutation for miltipel assets a time
                                          insample=ins, 
                                          k_fold=self.cfg.k_fold, 
                                          test_size=self.cfg.w4w_testsize,
                                          gap=self.cfg.gap)


        # MCPT Stat Test
        print("START STAT TEST")
        pvalue = self._mcpt_stat_test(pf_original=pf_w4w_is, insample=ins)


        # OOS Finance Test
        print(" ")
        print("  Result can be optimistic from reality  ")
        print("============= STRATEGY RESULT ============")
        oos_rep = self._oos_finance_test(strategy=self.strategy, outsample=oos, asset_type=asset_type)

        # final report payload
        return f"""Strategy validity pval: {pvalue}
Return per year: {oos_rep["strat_ret"]}
Sharpe: {oos_rep["strat_sharpe"]}
MDD: {oos_rep["strat_mdd"]}"""
    

    # ------------------
    #        HELPER
    # ------------------
    def _walkforward_train(self,
                           insample: pd.DataFrame,
                           strategy,  # --> Need update to generate permutation for miltipel assets a time 
                           k_fold: int, 
                           test_size: float,
                           gap: int):

        folds = WalkForwardSplit(
            k_fold=k_fold,
            test_size=test_size,
            gap=gap,
        ).split(insample)


        pf_each_fold = []

        for fold_df in folds:
            fold_df = fold_df.reset_index()

            r = strategy.run(fold_df) # -->> Strategy return a dateset with cols
            
            profit_factor = FinanceTest.profit_factor(df_ = r, pos_col = 'position')

            pf_each_fold.append(profit_factor)

        return np.mean(pf_each_fold)


    def _mcpt_stat_test(self, pf_original: float, insample: pd.DataFrame) -> Dict:
        
        pf_better = 0

        for _ in range(self.cfg.n_perm):

            ins_perm = MonteCarlosPermutation.gen_permutation(
                ohlc=insample, start_index=self.cfg.perm_start_index,
                end_index=self.cfg.perm_end_index
            )

            r = self._walkforward_train(strategy=self.strategy,
                                        insample=ins_perm, 
                                        k_fold=self.cfg.k_fold,
                                        test_size=self.cfg.w4w_testsize,
                                        gap = self.cfg.gap)

            if r > pf_original:
                pf_better+=1

        return StatTest.quasi_pvalue(mcpt_better=pf_better, all_trials=self.cfg.n_perm)
    

    def _oos_finance_test(self, strategy, outsample: pd.DataFrame, asset_type:str) -> Dict:

        o = strategy.run(outsample).reset_index()

        # Strategy Finance + Stat Test result
        perf = FinanceTest.fixed_capital_fp(o, asset_type=asset_type)

        strat_ret_pct = perf['return_per_year']
        sharpe = perf['sharpe']
        mdd = perf['max_drawdown']

        # buy&hold from log returns: exp(sum(log_ret))-1
        bh_ret_pct = ((o['close'].iloc[-1] - o['close'].iloc[0])/ o['close'].iloc[0]) * 100

        return {
            "strat_ret": f"{strat_ret_pct:.3f}%",
            "strat_sharpe": f"{sharpe:.3f}",
            "strat_mdd": f"{mdd:.3f}%",
            "bh_ret": f"{bh_ret_pct:.3f}%"
        }

# TEST CASE
# CMD: python -m core.exe
if __name__ == '__main__':
    exe = SystemExecute(strategy=DonchianBreakout(config=DonchianCfg.config), 
                        config=SysConfig).execute("VN30F1M", asset_type='future')
    print(exe)