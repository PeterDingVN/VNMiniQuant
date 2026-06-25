import numpy as np
import pandas as pd

import warnings
import pandas as pd


# Helper
class Fee():
    def __init__(self):
        self.vn_futures = 0.4
        self.vn_stock = 2



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
                 one_way_fee: float, 
                 inital_capital: float = 100_000_000, 
                 annual_sessions_in_days: float = 252,
                 risk_free_rate: float = 0.0
                 ):
        
        
        self.trade_period = annual_sessions_in_days
        self.rf_rate = risk_free_rate
        self.initial_capital = inital_capital
        self.one_way_fee = one_way_fee # calculation method with fee_compute function


        self.df = self.Gains_Calculation_Simple(StandardizeInput.column_std(df))
        self.year_count = len(self.df.resample('D').sum(min_count=1).dropna())/ annual_sessions_in_days

        
    def Gains_Calculation_Simple(self, df):
        df['pos_change'] = df['position'].diff().fillna(df['position'].iloc[0])

        df['gain'] = df['position'].shift(1) * df['close'].diff()
        # scale = df['position'] * alloc_per_trade / df['close']
        # df['gains_scaled'] = df['position'].shift(1) * 

        df['fee'] = self.one_way_fee * df['pos_change'].abs() 
        df['gain_after_fee'] = df['gain'] - df['fee']

        df['cum_gain_after_fee'] = df['gain_after_fee'].cumsum()
        df['total_equity'] = df['cum_gain_after_fee'] + self.initial_capital
        
        return df
    


    def Sharpe_after_fee(self):
        
        daily_gain = (self.df['gain_after_fee'].resample('D').sum(min_count=1).dropna())
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
        # Abs
        equity = self.df['cum_gain_after_fee']
        peak = equity[equity!=0].cummax()
        dd_abs = (peak - equity)
        mdd_abs = dd_abs.max()
        
        # Pct
        daily_close = (self.df['close'].resample('D').last().dropna())
        yearly_max = daily_close.groupby(daily_close.index.year).mean()
        cash_max = yearly_max.mean()
        mdd_pct = ((dd_abs/ cash_max) * 100).max()

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
        final_cap = self.df['total_equity'].iloc[-1]
        total_ret = ((final_cap / self.initial_capital) - 1)
        year_no = self.year_count

        total_return = total_ret * 100
        return_per_year = total_return / year_no
        cagr = ((final_cap / self.initial_capital) ** (1 / year_no) - 1)*100

        return total_return, return_per_year, cagr
    

    def Hitrate(self):
        long_mask = self.df['position'] > 0
        short_mask = self.df['position'] < 0
        long_trades = long_mask.sum()
        short_trades = short_mask.sum()

        long_wins = ((self.df['gain_after_fee'] > 0) & long_mask).sum()
        short_wins = ((self.df['gain_after_fee'] > 0) & short_mask).sum()

        long_hitrate = (long_wins / long_trades * 100) if long_trades > 0 else 0.0
        short_hitrate = (short_wins / short_trades * 100) if short_trades > 0 else 0.0

        return long_hitrate, short_hitrate


    def Longest_streak(self):
        gain = self.df['gain_after_fee'].to_numpy()
        active_gains = gain[gain != 0]

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
                 one_way_fee: float, 
                 inital_capital: float = 100_000_000, 
                 annual_sessions_in_days: float = 252,
                 risk_free_rate: float = 0.0
                 ):
        super().__init__(df, one_way_fee, inital_capital, annual_sessions_in_days, risk_free_rate)

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
    Initial capital: {self.initial_capital:.2f}
     Ending capital: {self.df['total_equity'].iloc[-1]:.2f}
             Sharpe: {sharpe:.2f}
                MDD: {mdd_3[0]:.2f} ({mdd_3[1]:.2f}%); {mdd_3[2]}
       Total Profit: {profit_3[0]:.2f}
      Annual Profit: {profit_3[1]:.2f}
       Daily Profit: {profit_3[2]:.2f}
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

    def pnl_report(self, plot=True):
        dash = self.dashboard()
        print(dash)
        if plot:
            self.plot_equity()
        

    def plot_equity():
        pass



# python -m backtest.backtest_utils.finance_backtest
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\HP\.0_PycharmProjects\VNMiniQuant_Futures\data\cached_data\stock_price_cache\cty_pos_T6_15m.csv')
    perf = FinanceBacktest(df, one_way_fee=0.4, inital_capital=1_000).dashboard()
    print(perf)
    

