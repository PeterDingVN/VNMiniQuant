import pandas as pd
import numpy as np

def generate_signals(df, start_sig: int=1):
    """
    Generates trend signals based on weighted EMA and MACD states,
    and computes the strategy's logarithmic returns.
    """
    assert start_sig >= 1

    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. EMA Signal (State: Over / Under)
    # ---------------------------------------------------------
    conditions_ema = [
        df['ema_55'] > df['ema_233'],
        df['ema_55'] < df['ema_233']
    ]
    df['ema_signal'] = np.select(conditions_ema, [1, -1], default=0)
    
    # ---------------------------------------------------------
    # 2. MACD Signal (State: Over / Under)
    # ---------------------------------------------------------
    conditions_macd = [
        df['macd'] > df['macd_signal'],
        df['macd'] < df['macd_signal']
    ]
    df['macd_signal_state'] = np.select(conditions_macd, [1, -1], default=0)
    
    # ---------------------------------------------------------
    # 3. Apply Weights & Final Signal
    # ---------------------------------------------------------
    weight_ema = 0.55
    weight_macd = 0.45
    
    df['weighted_score'] = (df['ema_signal'] * weight_ema) + (df['macd_signal_state'] * weight_macd)
    
    # Final signal: 1 (Trend Up), -1 (Trend Down), 0 (Unclear/Conflicting)
    conditions_final = [
        df['weighted_score'] >= 0.5,   
        df['weighted_score'] <= -0.5   
    ]
    df['final_signal'] = np.select(conditions_final, [1, 0], default=0)

    df['final_signal'] = np.where(df.index > start_sig-1, df['final_signal'], 0)
    # ---------------------------------------------------------
    # 4. Return Computations
    # ---------------------------------------------------------
    # Calculate continuous logarithmic return of the asset
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # # Calculate strategy return
    # # Shifting the signal by 1 ensures we trade today based on yesterday's closing signal
    df['real_return'] = df['log_return'] * abs(df['final_signal'])
    df['strat_ret'] = np.cumsum(df['real_return'])
    
    
    return df

