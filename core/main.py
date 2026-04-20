from data import AccessData
from data.data_features import FeatureBuilderNonML
from signal_generation import generate_signals
from strategy_backtest import MonteCarlosPermutation, WalkForwardSplit, FinanceTest, StatTest
import pandas as pd
import numpy as np
from _0_testing_note.targetvar_diagnosis import get_permutation
from config import PORTFOLIO

    # Set up dataframe
ls_ofdata = AccessData(symbol=PORTFOLIO).access_data()

# btc = pd.read_csv(r'C:\Users\HP\Downloads\ETHUSDT.csv')
# ls_ofdata = [btc]

# date_range = [('2015-01-01', '2019-04-01'), ('2019-04-15', '2023-04-15'), ('2023-04-19', '2026-04-19')]


for df_dict in ls_ofdata:
    print(f'STOCK: {df_dict['symbol']}')
    data_stock = df_dict['data']

    df_split_base = WalkForwardSplit(k_fold=3, test_size=0.5, gap=0).split(data_stock)
    # df = data_stock[data_stock['time'].between(tup[0], tup[1])]

    # BASE RESUKT
    for i in df_split_base:

        df = i.copy()
        df_fe  = FeatureBuilderNonML().build_features(df)
        df_signal = generate_signals(df_fe, start_sig=283)
        pf_base = FinanceTest.profit_factor(df_signal['real_return'])

        # MCPT
        pf_result = []
        all_test = 0

        for _ in range(1000):
            # print(f'----------------- RESULT {i+1} ---------------------')
            all_test+=1

            # Data
            data = MonteCarlosPermutation.gen_permutation(df, start_index=283)
            # data = get_permutation(df)
            data_fe_ = FeatureBuilderNonML().build_features(data.reset_index())
            data_signal_ = generate_signals(data_fe_, start_sig=283)


            # Combine bacj
            res = FinanceTest.profit_factor(data_signal_['real_return'])
            if res >= pf_base:
                pf_result.append(res)

        print(len(pf_result))
        p_val = StatTest.quasi_pvalue(mcpt_better=pf_result, all_trials=all_test)


        tot_ret_strat = (np.exp(df_signal['strat_ret'].iloc[-1])-1)*100
        buy_hold = ((df_signal['close'].iloc[-1] - df_signal['close'].iloc[0]) / df_signal['close'].iloc[0])*100

        print('Buy Hold Strat', f'{buy_hold:.3f} %')
        print('Strat EMA MACD', f'{tot_ret_strat:.3f} %')
        print(p_val)
    print('-----------------------------')
    print(' ')




