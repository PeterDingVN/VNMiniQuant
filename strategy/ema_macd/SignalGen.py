import numpy as np

def generate_signals(df, start_sig: int=1):

    """
    Generates trend signals based on weighted EMA and MACD states,
    and computes the strategy's returns.
    """

    assert start_sig >= 1
    df = df.copy()
    
    # ---------------------------------------------------------
    # EMA Signal (fast cross up and down slow)
    # ---------------------------------------------------------
    conditions_ema = [
        df['ema_55'] > df['ema_233'],
        df['ema_55'] < df['ema_233']
    ]
    df['ema_signal'] = np.select(conditions_ema, [1, -1], default=0)
    

    # ---------------------------------------------------------
    # MACD Signal (macd cross up and down signal line)
    # ---------------------------------------------------------
    conditions_macd = [
        df['macd'] > df['macd_signal'],
        df['macd'] < df['macd_signal']
    ]
    df['macd_signal_state'] = np.select(conditions_macd, [1, -1], default=0)
    

    # ---------------------------------------------------------
    # Weights for each indicators + signal generation
    # ---------------------------------------------------------
    weight_ema = 0.55
    weight_macd = 0.45
    
    df['weighted_score'] = (df['ema_signal'] * weight_ema) + (df['macd_signal_state'] * weight_macd)
    
    conditions_final = [
        df['weighted_score'] >= 0.5  
    ]
    df['final_signal'] = np.select(conditions_final, [1], default=0)
    df['final_signal'] = np.where(df.index > start_sig-1, df['final_signal'], 0) # -> skip n periods before 
                                                                                 #   MA MACD has meaning (>= slow len)


    # ---------------------------------------------------------
    # Output cols: Return Computations
    # ---------------------------------------------------------
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)) # -> log return at each bar
    df['real_return'] = df['log_return'] * df['final_signal'].shift(1)  # -> real return = log * signal_yesterday
                                                                            # -> avoid lookahead bias
    df['strat_ret'] = np.cumsum(df['real_return'])  # -> total return from real signal
    
    return df

def generate_signal_futures(df, start_sig: int=1):
    assert start_sig >= 1
    df = df.copy()
    
    # ---------------------------------------------------------
    #                       Signal gen
    # ---------------------------------------------------------
    # EMA crossover
    conditions_ema = [
        df['ema_55'] > df['ema_233'],
        df['ema_55'] < df['ema_233']
    ]
    df['ema_signal'] = np.select(conditions_ema, [1, -1], default=0)
    

    # MACD crossover
    conditions_macd = [
        df['macd'] > df['macd_signal'],
        df['macd'] < df['macd_signal']
    ]
    df['macd_signal_state'] = np.select(conditions_macd, [1, -1], default=0)
    

    # Weight assignment
    weight_ema = 0.55
    weight_macd = 0.45
    
    df['weighted_score'] = (df['ema_signal'] * weight_ema) + (df['macd_signal_state'] * weight_macd)
    
    conditions_final = [
        df['weighted_score'] >= 0.5,
        df['weighted_score'] <= -0.55  
    ]

    # Signal generation
    df['final_signal'] = np.select(conditions_final, [1, -1], default=0)
    df['final_signal'] = np.where(df.index > start_sig-1, df['final_signal'], 0) # -> skip n periods before 
                                                                                 #   MA MACD has meaning (>= slow len)


    # ========================================================
    #           Output cols: Return Computations
    # ========================================================

    point_ret = np.array(df['close'].shift(-1) - df['close'])
    position = np.array(df['final_signal'])
    position_nxt = np.array(df['final_signal'].shift(-1))

    cost = np.where(abs(position-position_nxt)==1, 0.44*10**(-3),
                    np.where(abs(position-position_nxt)==2, 0.8*10**(-3), 0))

    df['real_return'] = point_ret*position - cost  # -> real return = log * signal_yesterday
                                                                            # -> avoid lookahead bias
    df['strat_ret'] = np.cumsum(df['real_return'])  # -> total return from real signal
    
    return df