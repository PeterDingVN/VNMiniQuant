WARMUP = 283  # 233 + 50

for tup in date_ranges:
    start_pos  = data_stock.index[data_stock['time'] >= tup[0]][0]
    end_pos    = data_stock.index[data_stock['time'] <= tup[1]][-1] + 1
    seed_start = max(0, start_pos - WARMUP)

    # Full window: warmup + chunk
    df_full = data_stock.iloc[seed_start:end_pos].copy()

    # Features built on full window — EWM is properly seeded
    df_fe_full = FeatureBuilderNonML().build_features(df_full)

    # Trim to actual test window AFTER features exist
    df_fe      = df_fe_full.iloc[start_pos - seed_start:]
    df_signal  = generate_signals(df_fe)
    pf_base    = FinanceTest.profit_factor(df_signal['log_return'])

    # Permutation: shuffle only post-warmup bars
    for i in range(1000):
        perm      = get_permutation(df_full, start_index=WARMUP)
        perm_fe   = FeatureBuilderNonML().build_features(perm)
        perm_fe   = perm_fe.iloc[WARMUP:]          # trim warmup
        perm_sig  = generate_signals(perm_fe)
        res       = FinanceTest.profit_factor(perm_sig['log_return'])
        ...





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

date_range = [('2015-01-01', '2019-04-01'), ('2019-04-15', '2023-04-15'), ('2023-04-19', '2026-04-19')]

for df_dict in ls_ofdata:
    print(f'STOCK: {df_dict['symbol']}')
    data_stock = df_dict['data']
    # data_stock = df_dict.copy()
    for tup in date_range:
        # df_split_base = WalkForwardSplit(k_fold=3, test_size=0.2, gap=0).split(df)

        df = data_stock[data_stock['time'].between(tup[0], tup[1])]
        df_fe  = FeatureBuilderNonML().build_features(df)
        df_signal = generate_signals(df_fe)

        # ORIGINAL
        # # Data
        # res_base=[]
        # for data_base in df_split_base:
        #     df_fe  = FeatureBuilderNonML().build_features(data_base)
        #     df_signal = generate_signals(df_fe)
        #     res_base.append(df_signal)

        # df_signal_base = pd.concat(res_base)
        # print(df_signal_base.info())

        # Result
        # print('Win rate', FinanceTest.winrate(df_signal['log_return']))
        # print('Profit factor', FinanceTest.profit_factor(df_signal['log_return']))
        # print('Sharpe', FinanceTest.sharpe_ratio(df_signal['log_return']))

        # MCPT

        pf_result = []
        pf_base = FinanceTest.profit_factor(df_signal['log_return'])

        for i in range(1000):
            # print(f'----------------- RESULT {i+1} ---------------------')

            # Data
            # data = MonteCarlosPermutation.gen_permutation(df, end_index=0)
            data = get_permutation(df, start_index=233+50)

            # W4w
            # data_split = WalkForwardSplit(k_fold=3, test_size=0.2, gap=0).split(data)
            # signal = []
            # for df_ in data_split:

                # FE and signal gen (optimizaing)
            data_fe_ = FeatureBuilderNonML().build_features(data)
            data_signal_ = generate_signals(data_fe_)
            #     signal.append(data_signal_)

            # # Data signal combined back
            # data_sig = pd.concat(signal)

            # Combine bacj
            res = FinanceTest.profit_factor(data_signal_['log_return'])
            if res >= pf_base:
                pf_result.append(res)


        p_val = StatTest.quasi_pvalue(finance_perf=pf_base, mcpt_perf=pf_result)
        tot_ret_strat = (np.exp(df_signal['strat_ret'].iloc[-1])-1)*100
        buy_hold = ((df_signal['close'].iloc[-1] - df_signal['close'].iloc[0]) / df_signal['close'].iloc[0])*100

        print('Buy Hold Strat', f'{buy_hold:.3f} %')
        print('Strat EMA MACD', f'{tot_ret_strat:.3f} %')
        print(p_val)
        print('-----------------------------')
        print(' ')
            # print('Win rate', FinanceTest.winrate(data_signal['strat_ret']))
            # print('Profit factor', FinanceTest.profit_factor(data_signal['strat_ret']))
            # print('Sharpe', FinanceTest.sharpe_ratio(data_signal['strat_ret']))

            # print(' --------------------------------------------------- ')
            # print(' ')




