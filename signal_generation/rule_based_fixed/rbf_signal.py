import pandas as pd
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
        df['weighted_score'] >= 0.5,   
        df['weighted_score'] <= -0.5   
    ]
    df['final_signal'] = np.select(conditions_final, [1, 0], default=0)
    df['final_signal'] = np.where(df.index > start_sig-1, df['final_signal'], 0) # -> skip n periods before 
                                                                                 #   MA MACD has meaning (>= slow len)


    # ---------------------------------------------------------
    # Output cols: Return Computations
    # ---------------------------------------------------------
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)) # -> log return at each bar
    df['real_return'] = df['log_return'] * abs(df['final_signal'])  # -> real return = log * signal
    df['strat_ret'] = np.cumsum(df['real_return'])  # -> total return from real signal
    
    
    return df

