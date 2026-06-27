import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import pandas as pd

from dataclasses import dataclass

# Helper
@dataclass
class Fee():
    fee = {
        'vn_futures': 0.4,
        'vn_stock': 0.5,
        'crypto': 0.00035,
        'us_stock': 0.5
    }
    

# =================================
# Standardize Input
# =================================
class StandardizeInput:

    REQUIRED_COLS = ["datetime", "close", "position"]

    DATETIME_ALIASES = {
        "time": "datetime",
        "timestamp": "datetime",
        "date": "datetime",
        "datetime": "datetime",
    }

    POSITION_ALIASES = {
        "pos": "position",
        "side": "position",
        "signal": "position",
        "sig": "position",
        "position": "position",
    }

    @staticmethod
    def column_std(df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # lowcase
        df.columns = [str(col).strip().lower() for col in df.columns]

        # rename cols
        rename_map = {}

        for col in df.columns:
            if col in StandardizeInput.DATETIME_ALIASES:
                rename_map[col] = "datetime"

            elif col in StandardizeInput.POSITION_ALIASES:
                rename_map[col] = "position"

        df = df.rename(columns=rename_map)

        # check cols
        missing_cols = [
            col
            for col in StandardizeInput.REQUIRED_COLS
            if col not in df.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Missing required column(s): {', '.join(missing_cols)}"
            )

        # convert datetime
        try:
            df["datetime"] = pd.to_datetime(
                df["datetime"],
                errors="raise"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to parse datetime column: {e}"
            )

        # conver other cols into float
        numeric_cols = [c for c in df.columns if c != "datetime"]

        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="raise"
                ).astype(float)
            except Exception as e:
                raise ValueError(
                    f"Column '{col}' cannot be converted to float: {e}"
                )

        # drop na
        nan_count = df.isna().any(axis=1).sum()

        if nan_count > 0:
            warnings.warn(
                f"{nan_count} row(s) containing NaN values were dropped.",
                UserWarning
            )
            df = df.dropna()

        # Sort and output
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.set_index('datetime')

        return df
    
    


# =================================
# Finance metric calculation
# =================================
class FinanceMetrics:

    def __init__(self,
                 df: pd.DataFrame, 
                 asset_type: str = 'vn_futures', 
                 initial_capital: float = 100_000_000,
                 exposure: float = 1.0,
                 currency: str = 'VND', 
                 annual_sessions_in_days: float = 252,
                 risk_free_rate: float = 0.0
                 ):
        
        
        std_data = StandardizeInput.column_std(df)
        self.trade_period = annual_sessions_in_days
        self.rf_rate = risk_free_rate
        self.currency = currency
        self.initial_capital = initial_capital

        if exposure > 1 or exposure <=0:
            raise ValueError('You must expose >0 and <=1 of your initial capital')
        self.exposure = exposure
        
        if self.currency.lower() == 'usd':
            self.initial_capital = self.initial_capital / 26_000
        elif self.currency.lower() not in ['usd', 'vnd']:
            raise ValueError('currency only accepts "vnd" or "usd"')
        
        if asset_type not in ['vn_futures', 'crypto', 'vn_stock', 'us_stock']:
            raise ValueError('asset type only accepts: vn_stock, us_stock, vn_futures, crypto')
        
        if asset_type == 'vn_futures':
            self.initial_capital = self.initial_capital / 100_000
            if self.initial_capital < std_data['close'].iloc[0]:
                raise ValueError(f'Initial capital too small to buy 1 contract at {std_data['close'].iloc[0]} points')
        else:
            if self.initial_capital < std_data['close'].iloc[0]:
                raise ValueError(f'Initial capital too small to buy 1 unit at price {std_data['close'].iloc[0]}')
        
        self.one_way_fee = Fee().fee[asset_type]

        self.df = self.Gains_Calculation_Simple(std_data)
        self.year_count = len(self.df.resample('D').sum(min_count=1).dropna())/ annual_sessions_in_days

        
    def Gains_Calculation_Simple(self, df):
        df['pos_change'] = df['position'].diff().fillna(df['position'].iloc[0])

        # Absolute PnL (1 contract)
        df['gain'] = df['position'].shift(1) * df['close'].diff()
        df['fee'] = self.one_way_fee * df['pos_change'].abs()
        df['gain_after_fee'] = df['gain'] - df['fee']
        df['cum_gain_after_fee'] = df['gain_after_fee'].cumsum()
        df['total_equity'] = self.initial_capital + df['cum_gain_after_fee']

        # Scaled PnL (n% capital invested, fixed-notional, no compounding)
        # Scale by exposure over close price
        df['scaler'] =  self.initial_capital * self.exposure / df['close']
        df['scaled_gain_after_fee'] = df['gain_after_fee'] * df['scaler']
        df['scaled_cum_gain_after_fee'] = df['scaled_gain_after_fee'].cumsum()
        df['scaled_equity'] = self.initial_capital + df['scaled_cum_gain_after_fee']
        
        return df


    def Sharpe_after_fee(self):
        
        daily_gain = (self.df['scaled_gain_after_fee'].resample('D').sum(min_count=1).dropna())
        daily_close = (self.df['close'].resample('D').last().dropna())
        daily_gain, daily_close = daily_gain.align(daily_close,join='inner')


        if len(daily_gain) < 2:
            return np.nan
        
        yearly_max = daily_close.groupby(daily_close.index.year).transform('max')
        cash_max = yearly_max.mean()
        year_total = self.year_count

        daily_return = daily_gain / cash_max / year_total
        daily_rf = (1 + self.rf_rate) ** (1 / self.trade_period) - 1


        daily_ret = daily_return - daily_rf
        std_ret = daily_ret.std()

        if std_ret == 0 or np.isnan(std_ret):
            return np.nan

        sharpe = (daily_ret.mean()/ std_ret) * np.sqrt(self.trade_period)
        return sharpe


    def MDD(self):
        # Abs mdd
        equity = self.df['total_equity']
        peak = equity[equity!=0].cummax()
        dd_abs = (peak - equity)
        mdd_abs = dd_abs.max()
        
        # Pct mdd
        equity_pct = self.df['scaled_equity']
        peak = equity_pct[equity_pct!=0].cummax()
        mdd_pct = ((peak - equity_pct)/self.initial_capital * 100).max()
        

        # Date
        mdd_trough_date = dd_abs.idxmax()
        mdd_peak_date = equity.loc[:mdd_trough_date].idxmax()
        mdd_date = f'Time: {mdd_peak_date} -> {mdd_trough_date}'

        return mdd_abs, mdd_pct, mdd_date


    def Total_Trade(self):
        long_count = (
                    (self.df['position'].shift(1).fillna(0).isin([0, -1])) &
                    (self.df['position'] == 1)
                    ).sum()
            

        short_count = (
                    (self.df['position'].shift(1).fillna(0).isin([0, 1])) &
                    (self.df['position'] == -1)
                    ).sum()

        return long_count, short_count


    def Profit(self):
        final_gain = self.df['cum_gain_after_fee'].iloc[-1]

        total_profit = final_gain
        profit_after_fee_per_year = final_gain / self.year_count
        profit_after_fee_per_day = final_gain / (self.year_count * self.trade_period)

        return total_profit, profit_after_fee_per_year, profit_after_fee_per_day


    def Return(self):
        final_cap = self.df['scaled_equity'].iloc[-1]
        total_ret = ((final_cap / self.initial_capital) - 1)
        year_no = self.year_count

        total_return = total_ret * 100
        return_per_year = total_return / year_no
        cagr = ((final_cap / self.initial_capital) ** (1 / year_no) - 1)*100

        return total_return, return_per_year, cagr
    
    
    def Hitrate(self):
        positions = self.df['position'].values
        gains = self.df['gain_after_fee'].values
        signs = np.sign(positions)
        sign_changes = np.diff(signs, prepend=signs[0] + 1) != 0
        block_ids = np.cumsum(sign_changes)
        trade_gains = np.bincount(block_ids, weights=gains)[1:]
        block_signs = signs[sign_changes]

        # Long win
        long_mask = block_signs > 0
        long_trades = np.sum(long_mask)
        long_wins = np.sum(long_mask & (trade_gains > 0))

        # Short win
        short_mask = block_signs < 0
        short_trades = np.sum(short_mask)
        short_wins = np.sum(short_mask & (trade_gains > 0))

        # Hitrate
        long_hitrate = (long_wins / long_trades * 100) if long_trades > 0 else 0.0
        short_hitrate = (short_wins / short_trades * 100) if short_trades > 0 else 0.0

        return long_hitrate, short_hitrate


    def Longest_streak(self):
        active_gains = self.df['gain_after_fee'].to_numpy()
        
        if active_gains.size == 0:
            return 0, 0

        # -- Calculate Longest Win Streak --
        is_win = active_gains > 0
        win_groups = np.cumsum(~is_win)[is_win]
        longest_win_streak = int(np.bincount(win_groups).max()) if win_groups.size > 0 else 0

        # -- Calculate Longest Loss Streak --
        is_loss = active_gains < 0
        loss_groups = np.cumsum(~is_loss)[is_loss]
        longest_loss_streak = int(np.bincount(loss_groups).max()) if loss_groups.size > 0 else 0

        return longest_win_streak, longest_loss_streak



class FinanceBacktest(FinanceMetrics):
    def __init__(self,
                 df: pd.DataFrame, 
                 asset_type: str, 
                 inital_capital: float = 100_000_000, 
                 exposure: float = 1.0,
                 currency: str = 'VND', 
                 annual_sessions_in_days: float = 252,
                 risk_free_rate: float = 0.0
                 ):
        super().__init__(df, asset_type, inital_capital, exposure, currency, annual_sessions_in_days, risk_free_rate)

    def dashboard(self):
        sharpe = self.Sharpe_after_fee()
        mdd_3 = self.MDD()
        profit_3 = self.Profit()
        return_3 = self.Return()
        hitrate_2 = self.Hitrate()
        trade_2 = self.Total_Trade()
        streak_2 = self.Longest_streak()

        return f"""
======================================================
                 Financial Backtest 
======================================================
    Initial capital: {self.initial_capital:,.2f}
     Ending capital: {self.df['total_equity'].iloc[-1]:,.2f}
             Sharpe: {sharpe:.2f}
                MDD: {mdd_3[0]:.2f} ({mdd_3[1]:.2f}%); {mdd_3[2]}
       Total Profit: {profit_3[0]:,.2f}
      Annual Profit: {profit_3[1]:,.2f}
       Daily Profit: {profit_3[2]:,.2f}
       Total Return: {return_3[0]:.2f}%
      Annual Return: {return_3[1]:.2f}%
               CAGR: {return_3[2]:.2f}%
       Hitrate Long: {hitrate_2[0]:.2f}%
      Hitrate Short: {hitrate_2[1]:.2f}%
 Longest win streak: {streak_2[0]}
Longest lose streak: {streak_2[1]}
        Long trades: {trade_2[0]}
       Short trades: {trade_2[1]}

"""

    def plot_equity(self):
        figsize = (25,5)
        
        sharpe = self.Sharpe_after_fee()
        
        _, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [6, 4]}, sharex=True)
        
        # 1. Return
        equity = self.df['scaled_equity'][~self.df['scaled_equity'].isin([np.nan, np.inf, -np.inf])]
        equity = equity.resample('D').last().dropna()
        ret = (equity / equity.iloc[0] - 1) * 100
        axs[0].plot(ret.index, ret, label=f"Strategy (Sharpe_after_fee: {sharpe:.2f})", color="blue")
        axs[0].set_ylabel("Return (%)")
        axs[0].legend(); axs[0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        peak = equity[equity!=0].cummax()
        daily_dd = (peak - equity)/self.initial_capital * 100
        daily_dd = daily_dd.resample('D').last().dropna()
        axs[1].fill_between(daily_dd.index, daily_dd, 0, color='red', alpha=0.4)
        axs[1].set_ylabel("Drawdown %"); axs[1].grid(True, alpha=0.3)

        plt.tight_layout(); plt.show()

    # --- PNL REPORT MAIN ---
    def pnl_report(self, plot=True):
        dash = self.dashboard()
        print(dash)
        if plot:
            self.plot_equity()



# python -m backtest.backtest_utils.finance_backtest
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\HP\.0_PycharmProjects\VNMiniQuant_main\data\cached_data\alphaT6_s247.csv')
    FinanceBacktest(df, asset_type='vn_futures', 
                    currency='vnd', 
                    inital_capital=124_000_000_000, 
                    exposure=1, risk_free_rate=0).pnl_report(plot=True)
    
    
