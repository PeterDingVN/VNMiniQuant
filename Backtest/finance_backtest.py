import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
class LengthError(Exception):
    pass

from dataclasses import dataclass
from datetime import time




# ==========================================================================================
#                                       SUPPORTING FUNC 
# ==========================================================================================

# ---- Plot config - font size -----
TITLE_SIZE = 15
LABEL_SIZE = 16.5
TICK_SIZE = 16.5
LEGEND_SIZE = 16.5


# ---- Fee One way by Asset class ----
@dataclass
class Fee():
    fee = {
    'vn_future': 0.4,  # absolute 0.4 pts
    'vn_stock': 0.005,  # 0.35% including tax + 0.12% advanced cash (this fee been averaged based on holiday duration in Vietnam)
    'vn_stock_no_adv': 0.0035, # 0.35% including tax not using advanced cash 
    'crypto': 0.00035,
    'us_stock': 0.5,
    'us_future': 0.3
    }



#  ------ Market regime rule ----

# VN stock market
@dataclass
class VnStockRule:
    advanced_cash: bool = True
    settlement_bdays: int = 2
    cutoff_time: str = "13:00"
    pos_col: str = "position"
    dt_col: str = "datetime"

    def normalize_signal(self, signal) -> pd.Series:
        s = pd.Series(signal)
        return (s.fillna(0) > 0).astype(float)
    

    def _settlement_ready_dt(self, trade_dt) -> pd.Timestamp:
        def _parse_hhmm(s: str) -> time:
            hh, mm = map(int, s.split(":"))
            return time(hh, mm)
        
        base = pd.Timestamp(trade_dt).normalize() + pd.offsets.BDay(self.settlement_bdays)
        cutoff = _parse_hhmm(self.cutoff_time)
        return base.replace(
            hour=cutoff.hour,
            minute=cutoff.minute,
            second=0,
            microsecond=0,
        )


    def _infer_bar_mode(self, dts: pd.Series) -> str:
        dts = pd.Series(pd.to_datetime(dts)).sort_values().reset_index(drop=True)

        # If all timestamps are date-only, treat as daily.
        if dts.dt.time.nunique() == 1 and dts.dt.time.iloc[0] == time(0, 0):
            return "daily"

        median_delta = dts.diff().dropna().median()
        return "daily" if median_delta >= pd.Timedelta("20h") else "intraday"


    def _can_exit_now(self, now: pd.Timestamp, ready_dt: pd.Timestamp, bar_mode: str) -> bool:
        now = pd.Timestamp(now)
        ready_dt = pd.Timestamp(ready_dt)
        if bar_mode == "daily":
            return now.normalize() >= ready_dt.normalize()
        return now >= ready_dt


    def _can_enter_now(self, now: pd.Timestamp, cash_ready_dt: pd.Timestamp, bar_mode: str) -> bool:
        if cash_ready_dt == pd.Timestamp.min:
            return True
        now = pd.Timestamp(now)
        cash_ready_dt = pd.Timestamp(cash_ready_dt)
        if bar_mode == "daily":
            return now.normalize() >= cash_ready_dt.normalize()
        return now >= cash_ready_dt


    def apply(
        self,
        data: pd.DataFrame,
        bar_mode: str = "auto",
    ) -> pd.DataFrame:

        signal_col = self.pos_col
        dt_col = self.dt_col

        if dt_col in data.columns:
            dts = pd.Series(pd.to_datetime(data[dt_col]), index=data.index)
        else:
            dts = pd.Series(pd.to_datetime(data.index), index=data.index)

        desired = self.normalize_signal(data[signal_col]).to_numpy()

        mode = self._infer_bar_mode(dts) if bar_mode == "auto" else bar_mode.lower()

        out = np.zeros(len(data), dtype=np.int8)

        in_pos = False
        entry_ready_dt = pd.Timestamp.min   # earliest time we are allowed to exit this long
        cash_ready_dt = pd.Timestamp.min    # earliest time we are allowed to enter next long

        for i, (now, want) in enumerate(zip(dts, desired)):
            now = pd.Timestamp(now)

            if in_pos:
                # Hold the position until T+2.5; only after that can we obey a 0 signal.
                if self._can_exit_now(now, entry_ready_dt, mode) and want == 0:
                    in_pos = False
                    cash_ready_dt = self._settlement_ready_dt(now) if not self.advanced_cash else now
                out[i] = 1 if in_pos else 0
            else:
                # Flat state: only enter if cash is available and signal wants 1.
                if self._can_enter_now(now, cash_ready_dt, mode) and want == 1:
                    in_pos = True
                    entry_ready_dt = self._settlement_ready_dt(now)
                    out[i] = 1
                else:
                    out[i] = 0

        data['position'] = out
        return data



# -------------  Standardize Input -------------
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
    def market_rule(data: pd.DataFrame, fee_type: str) -> pd.DataFrame:
        if fee_type not in ['vn_future', 'vn_stock', 'vn_stock_no_adv', 'crypto', 'us_stock', 'us_future']:
            raise ValueError("fee_type only accepts: 'vn_future', 'vn_stock', 'vn_stock_no_adv', 'crypto', 'us_stock', 'us_future'")
        if fee_type == 'vn_stock':
            data_out = VnStockRule().apply(data)
        elif fee_type == 'vn_stock_no_adv':
            data_out =  VnStockRule(advanced_cash=False).apply(data)
        else:
            return data
        return data_out


    # ---- MAIN METHOD -----
    @staticmethod
    def column_std(df: pd.DataFrame, fee_type: str) -> pd.DataFrame:

        df = df.copy()
        if len(df) < 10:
            raise LengthError("Input data is too short. Please double check your alpha")

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
        
        # take necessary cols
        df = df[StandardizeInput.REQUIRED_COLS]

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

        # Sort by datetime
        df = df.sort_values("datetime").reset_index(drop=True)


        # drop na
        nan_count = df.isna().any(axis=1).sum()
        if nan_count > 0:
            warnings.warn(
                f"{nan_count} row(s) containing NaN values were dropped.",
                UserWarning
            )
            df = df.dropna()


        # Convert pos according to market
        df = StandardizeInput.market_rule(df, fee_type=fee_type)

        return df.set_index('datetime')
    
# ==========================================================================================
#                                        MAIN FUNC 
# ==========================================================================================

#   Finance metric calculation

class FinanceMetrics:

    def __init__(self,
                 df: pd.DataFrame, 
                 fee_type: str = 'vn_future', 
                 initial_capital: float = 100_000_000,
                 allocation_per_trade: float = 1.0,
                 fixed_allocation: bool = True,
                 currency: str = 'VND', 
                 risk_free_rate: float = 0.0
                 ):
        
        self.fixed_allocation = fixed_allocation
        
        std_data = StandardizeInput.column_std(df, fee_type=fee_type)

        if fee_type in ['crypto']:
            self.trade_period = 365
        else:
            self.trade_period = 252


        self.rf_rate = risk_free_rate
        self.currency = currency
        self.initial_capital = initial_capital

        if not (0 < allocation_per_trade <= 1):
            raise ValueError('You must expose >0 and <=1 of your initial capital')
        self.allocation_per_trade = allocation_per_trade

        
        if self.currency.lower() == "usd":
            self.initial_capital /= 26_000
        elif self.currency.lower() not in ["vnd", "usd"]:
            raise ValueError('currency only accepts "vnd" or "usd"')
        
        if fee_type not in ['vn_future', 'vn_stock', 'vn_stock_no_adv', 'crypto', 'us_stock', 'us_future']:
            raise ValueError("Fee type only accepts: 'vn_future', 'vn_stock', 'vn_stock_no_adv', 'crypto', 'us_stock', 'us_future'")
        
        # Check cash
        self.available_capital = self.initial_capital * self.allocation_per_trade

        if fee_type == "vn_future":
            self.initial_capital = self.initial_capital / 100_000
            self.available_capital = self.available_capital / 100_000

        if self.available_capital < std_data["close"].iloc[0]:
            raise ValueError(f'Not enough cash to buy 1 contract at {std_data["close"].iloc[0]}')

        
        self.fee_type = fee_type

        self.df = self.Gains_Calculation_Simple(std_data)
        self.year_count = len(self.df.resample('D').sum(min_count=1).dropna())/ self.trade_period

        
    def Gains_Calculation_Simple(self, df): 
        df['pos_change'] = df['position'].diff().ffill().fillna(df['position'].iloc[0])

        # Gain
        df['gain'] = df['position'].shift(1) * df['close'].diff()

        # Fee
        if self.fee_type in ['vn_stock', 'vn_stock_no_adv']:
            # Fee in pct
            one_way_fee = Fee.fee[self.fee_type] * df['close']
        else:
            # fee in abs
            one_way_fee = Fee.fee[self.fee_type]

        df['fee'] = one_way_fee * df['pos_change'].abs()

        df['gain_after_fee'] = df['gain'] - (df['fee'] * 1.01)  # slippage 1%

        # Absolute Pnl
        df['cum_gain_after_fee'] = df['gain_after_fee'].cumsum().ffill().fillna(0)
        df['total_equity'] = self.initial_capital + df['cum_gain_after_fee']


        # Scale "position" by alloc_per_trade -> pos = 1 with price x but I could buy 2x -> pos = 2
        # Fix allocation: allocate a fix pct of fixed initial capital 
        if self.fixed_allocation:
            df['scaler'] =  self.available_capital / df['close']

        # Growing equity: allocate a fix pct of growing current equity
        else:
            df['scaler'] = (df['total_equity'] * self.allocation_per_trade) / df['close']

        df['scaled_gain_after_fee'] = df['gain_after_fee'] * df['scaler']
        df['scaled_cum_gain_after_fee'] = df['scaled_gain_after_fee'].cumsum().ffill().fillna(0)
        df['scaled_equity'] = self.initial_capital + df['scaled_cum_gain_after_fee']

        return df
    

    def Margin(self):
        trade_turnover = (self.df['pos_change'].abs() * self.df['close']).sum()
        total_profit_after_fee = self.df['gain_after_fee'].sum()
        net_margin_bps = (total_profit_after_fee / trade_turnover) * 10000

        return net_margin_bps


    def Sharpe_after_fee(self):
        daily_gain = (self.df['scaled_gain_after_fee'].resample('D').sum(min_count=1).dropna())
        daily_return = daily_gain / self.available_capital
        daily_rf = (1 + self.rf_rate) ** (1 / self.trade_period) - 1

        daily_ret = daily_return - daily_rf
        std_ret = daily_ret.std()
        std_loss_ret = daily_ret[daily_ret<0].std()

        if std_ret == 0 or np.isnan(std_ret):
            return np.nan, np.nan

        sharpe = (daily_ret.mean()/ std_ret) * np.sqrt(self.trade_period)
        sortino = (daily_ret.mean()/ std_loss_ret) * np.sqrt(self.trade_period)
        return sharpe, sortino


    def Calmar(self):
        if self.fixed_allocation:
            ret = self.Return()[1]
        else:
            ret = self.Return()[2]
        mdd = self.MDD()[1]
        return ret/mdd if mdd != 0 else 0


    def MDD(self):
        # Abs mdd
        equity = self.df['scaled_equity']
        peak = equity[equity!=0].cummax()
        dd_abs = (peak - equity)
        mdd_abs = dd_abs.max()
        
        # Pct mdd
        if self.fixed_allocation:
            cash_max = self.available_capital
        else:
            cash_max = peak

        mdd_pct = ((peak - equity)/cash_max * 100).max()
        

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
    
    def Trade_per_day(self):
        all_trades = self.Total_Trade()[0] + self.Total_Trade()[1]
        all_days = self.year_count * self.trade_period
        return all_trades / all_days if all_days > 0 else 0


    def Profit(self):
        final_gain = self.df['scaled_cum_gain_after_fee'].iloc[-1]

        total_profit = final_gain
        profit_after_fee_per_year = final_gain / self.year_count

        return total_profit, profit_after_fee_per_year


    def Return(self):
        pnl = self.df['scaled_cum_gain_after_fee'].iloc[-1]
        year_no = self.year_count

        daily_close = self.df['close'].resample('D').last().dropna()
        max_close = daily_close.max()

        if self.fixed_allocation:
            capital_base = (self.available_capital * (max_close / self.df['close'])).mean()
        else:
            capital_base = ((df['total_equity'] * self.allocation_per_trade) * (max_close / self.df['close'])).mean()

        if capital_base == 0 or pd.isna(capital_base):
            return 0.0, 0.0, 0.0
 
        total_return = (pnl / capital_base) * 100
        return_per_year = total_return / year_no
        cagr = ((1 + total_return / 100) ** (1 / year_no) - 1) * 100

        return total_return, return_per_year, cagr

    
    def Hitrate(self):
        longs = self.df[self.df['position']>0]['gain_after_fee']
        shorts = self.df[self.df['position']<0]['gain_after_fee']
        alls = self.df[self.df['position']!=0]['gain_after_fee']
        # Long hitrate
        long_hitrate = len(longs[longs>0])/len(longs) * 100 if len(longs)>0 else 0

        # Short hitrate
        short_hitrate = len(shorts[shorts>0])/len(shorts) * 100 if len(shorts)>0 else 0

        # Total hr
        hitrate_total = len(alls[alls>0])/len(alls) * 100 if len(alls)>0 else 0

        return long_hitrate, short_hitrate, hitrate_total


    def Longest_streak(self):
        active_gains = self.df['gain_after_fee'].resample('D').sum(min_count=1).dropna().to_numpy()
        
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



class FinanceBacktest:
    def __init__(self, 
                 fee_type: str, 
                 initial_capital: float = 100_000_000, 
                 allocation_per_trade: float = 1.0,
                 currency: str = 'VND', 
                 fixed_allocation: bool = True,
                 risk_free_rate: float = 0.0
                 ):

        self.fee_type = fee_type
        self.initial_capital = initial_capital
        self.allocation_per_trade = allocation_per_trade
        self.fixed_allocation = fixed_allocation
        self.currency = currency
        self.risk_free_rate = risk_free_rate

    def dashboard(self, data: pd.DataFrame):
        fin_bt = FinanceMetrics(df=data, 
                                fee_type=self.fee_type, 
                                initial_capital=self.initial_capital, 
                                allocation_per_trade=self.allocation_per_trade,
                                currency=self.currency,
                                fixed_allocation=self.fixed_allocation,
                                risk_free_rate=self.risk_free_rate)
        
        name = self.fee_type.replace('_', ' ').title()

        margin = fin_bt.Margin()

        sharpe_and_sor = fin_bt.Sharpe_after_fee()
        calmar = fin_bt.Calmar()
        mdd_3 = fin_bt.MDD()

        return_3 = fin_bt.Return()
        profit_3 = fin_bt.Profit()

        hitrate_2 = fin_bt.Hitrate()

        trade_2 = fin_bt.Total_Trade()
        tpd = fin_bt.Trade_per_day()

        streak_2 = fin_bt.Longest_streak()


        return f"""
======================================================
             Financial Backtest {name}
======================================================
    Initial capital: {fin_bt.initial_capital:,.2f}
     Ending capital: {fin_bt.df['scaled_equity'].iloc[-1]:,.2f}
             Sharpe: {sharpe_and_sor[0]:.2f}
            Sortino: {sharpe_and_sor[1]:.2f}
             Calmar: {calmar:.2f}
                MDD: {mdd_3[0]:,.2f} ({mdd_3[1]:.2f}%); {mdd_3[2]}
       Total Profit: {profit_3[0]:,.2f}
   Margin per Trade: {margin:.2f} bps
       Total Return: {return_3[0]:.2f}%
    Return per year: {return_3[1]:.2f}%
               CAGR: {return_3[2]:.2f}%
       Hitrate Long: {hitrate_2[0]:.2f}%
      Hitrate Short: {hitrate_2[1]:.2f}%
      Total Hitrate: {hitrate_2[2]:.2f}%
        Longest Win: {streak_2[0]} days
       Longest Loss: {streak_2[1]} days
      Trade per Day: {tpd:.2f}
        Long Trades: {trade_2[0]}
       Short Trades: {trade_2[1]}
"""

    def plot_equity(self, data: pd.DataFrame):
        fin_bt = FinanceMetrics(df=data, 
                                fee_type=self.fee_type, 
                                initial_capital=self.initial_capital, 
                                allocation_per_trade=self.allocation_per_trade,
                                currency=self.currency,
                                fixed_allocation=self.fixed_allocation,
                                risk_free_rate=self.risk_free_rate)
        
        figsize = (22, 10)
        
        sharpe = fin_bt.Sharpe_after_fee()[0]
        
        _, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [6, 4]}, sharex=True)
        

        # 1. Return
        equity = fin_bt.df['scaled_equity'][~fin_bt.df['scaled_equity'].isin([np.nan, np.inf, -np.inf])]
        ret = equity / equity.iloc[0] - 1
        
        axs[0].plot(ret.index, 1+ret, label=f"Strategy (Sharpe_after_fee: {sharpe:.2f})", color="blue")
        axs[0].set_title("Strategy Performance", fontsize=TITLE_SIZE)
        axs[0].set_ylabel("Return", fontsize=LABEL_SIZE)
        axs[0].tick_params(axis='both', labelsize=TICK_SIZE)
        axs[0].legend(fontsize=LEGEND_SIZE, loc="upper left")
        axs[0].grid(True, alpha=0.3)
        

        # 2. Drawdown
        peak = equity[equity!=0].cummax()
        if self.fixed_allocation:
            daily_dd = (peak - equity)/fin_bt.available_capital * 100
        else:
            daily_dd = (peak - equity)/peak * 100
        
        axs[1].fill_between(daily_dd.index, daily_dd, 0, color='red', alpha=0.4, label="Drawdown")
        axs[1].set_ylabel("Drawdown (%)", fontsize=LABEL_SIZE)
        axs[1].set_xlabel("Date", fontsize=LABEL_SIZE)  # Added X-label for the bottom plot
        axs[1].tick_params(axis='both', labelsize=TICK_SIZE)
        axs[1].legend(fontsize=LEGEND_SIZE, loc="lower left")
        axs[1].grid(True, alpha=0.3)
        axs[1].invert_yaxis()

        plt.tight_layout(); plt.show()


    # --- PNL REPORT MAIN ---
    def pnl_report(self, data: pd.DataFrame, plot=True):
        dash = self.dashboard(data)
        print(dash)
        if plot:
            self.plot_equity(data)



# python -m Backtest.finance_backtest
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\HP\.0_PycharmProjects\VNMiniQuant_main\DataApi\cached_data\cci.csv')
    rep = FinanceBacktest(fee_type='vn_future', 
                    currency='vnd', 
                    initial_capital=123_099_000_200, 
                    allocation_per_trade=0.4828,
                    fixed_allocation=True,
                    risk_free_rate=0)

    out = rep.pnl_report(data=df, plot=True)

