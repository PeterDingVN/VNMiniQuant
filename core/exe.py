from data import AccessData

from strategy_backtest import MonteCarlosPermutation, WalkForwardSplit, FinanceTest, StatTest, TrainTestSplit
import numpy as np
import pandas as pd
from typing import Dict
from config import EmaMacdCfg, SysConfig

from strategy import EmaMacdStrategy


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
    def execute(self, stock: str) -> Dict: # --> Future update needs stock = list not str

        # Read Data
        data_stock = AccessData(symbol=stock).access_data(purpose='train')[0]['data'] # -> need update to run multiple assets

        if data_stock.empty:
            raise ValueError(f"Data for {stock} is empty")
        elif not isinstance(data_stock, pd.DataFrame):
            raise ValueError("AccessData auto return data(s) in a dict -> take data by data_stock[<data_id>]['data']")

       
        # IS/OOS
        ins, oos = TrainTestSplit(
                                test_size=self.cfg.oos_testsize, 
                                has_lookahead=False,
                                lookahead=0
                                ).split(data_stock)
        

        # W4W Training
        pf_w4w_is = self._walkforward_train(strategy=self.strategy,  # --> Need update to generate permutation for miltipel assets a time
                                          insample=ins, 
                                          k_fold=self.cfg.k_fold, 
                                          test_size=self.cfg.w4w_testsize,
                                          gap=self.cfg.gap)


        # MCPT Stat Test
        pvalue = self._mcpt_stat_test(pf_original=pf_w4w_is, insample=ins)


        # OOS Finance Test
        oos_rep = self._oos_finance_test(strategy=self.strategy, outsample=oos)

        # final report payload
        return {
            "Strategy validity pval": pvalue,
            "My Strategy": {oos_rep["strat_ret"]},
            "Buy Hold Strategy": {oos_rep["bh_ret"]}
        }
    

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

            r = strategy.run(fold_df) # -->> Strategy return a dateset with cols: signal, log_ret, ...
            req_col = [c for c in r.columns if c in ['log_return', 'real_return']] # -->> soon put them in config

            if len(req_col) != 2: 
                raise ValueError("We need the following cols for SystemExecute: 'log_return', 'real_return'")
            
            profit_factor = FinanceTest.profit_factor(ret=r['real_return'])

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
    

    def _oos_finance_test(self, strategy, outsample: pd.DataFrame) -> Dict:

        o = strategy.run(outsample)
        strat_ret_pct = FinanceTest.total_return(o["real_return"])

        # buy&hold from log returns: exp(sum(log_ret))-1
        bh_ret_pct = float((np.exp(o["log_return"].cumsum().iloc[-1]) - 1.0) * 100.0)
        pf_oos = FinanceTest.profit_factor(o["real_return"])

        return {
            "pf_oos": pf_oos,
            "strat_ret": round(strat_ret_pct, 4),
            "bh_ret": round(bh_ret_pct,4)
        }

# TEST CASE
# CMD: python -m core.exe
if __name__ == '__main__':
    exe = SystemExecute(strategy=EmaMacdStrategy(config=EmaMacdCfg), config=SysConfig).execute("AGR")
    print(exe)