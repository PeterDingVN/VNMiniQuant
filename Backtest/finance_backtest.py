import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
import pandas as pd

from dataclasses import dataclass

# ======================================  Helper =======================================
# Plot config - font size
TITLE_SIZE = 15
LABEL_SIZE = 16.5
TICK_SIZE = 16.5
LEGEND_SIZE = 16.5

# Fee by asset class
@dataclass
class Fee():
    fee = {
        'vn_future': 0.4,
        'vn_stock': 0.5,
        'crypto': 0.00035,
        'us_stock': 0.5,
        'us_future': 0.3
    }

# Market rule regime
class MarketRule:
    pass
    

# ==========================================  MAIN FUNC ==================================
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
                 fee_type: str = 'vn_future', 
                 initial_capital: float = 100_000_000,
                 allocation_per_trade: float = 1.0,
                 fixed_allocation: bool = True,
                 currency: str = 'VND', 
                 risk_free_rate: float = 0.0
                 ):
        
        self.fixed_allocation = fixed_allocation
        
        std_data = StandardizeInput.column_std(df)

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
        
        if fee_type not in ['vn_future', 'crypto', 'vn_stock', 'us_stock']:
            raise ValueError('Fee type only accepts: vn_stock, us_stock, vn_future, crypto')
        
        # Check cash
        self.available_capital = self.initial_capital * self.allocation_per_trade

        if fee_type == "vn_future":
            self.initial_capital = self.initial_capital / 100_000
            self.available_capital = self.available_capital / 100_000

        if self.available_capital < std_data["close"].iloc[0]:
            raise ValueError(f'Not enough cash to buy 1 contract at {std_data["close"].iloc[0]}')

        
        self.one_way_fee = Fee().fee[fee_type]

        self.df = self.Gains_Calculation_Simple(std_data)
        self.year_count = len(self.df.resample('D').sum(min_count=1).dropna())/ self.trade_period

        
    def Gains_Calculation_Simple(self, df): 
        df['pos_change'] = df['position'].diff().ffill().fillna(df['position'].iloc[0])

        # Gain
        df['gain'] = df['position'].shift(1) * df['close'].diff()
        df['fee'] = self.one_way_fee * df['pos_change'].abs()
        df['gain_after_fee'] = df['gain'] - (df['fee'] * 1.01)

        # Absolute Pnl
        df['cum_gain_after_fee'] = df['gain_after_fee'].cumsum().ffill().fillna(0)
        df['total_equity'] = self.initial_capital + df['cum_gain_after_fee']


        # Scale by allocation_per_trade over close price
        # Fix notional 
        if self.fixed_allocation:
            df['scaler'] =  self.available_capital / df['close']

        # Growing equity
        else:
            df['scaler'] = (df['total_equity'] * self.allocation_per_trade) / df['close']

        df['scaled_gain_after_fee'] = df['gain_after_fee'] * df['scaler']
        df['scaled_cum_gain_after_fee'] = df['scaled_gain_after_fee'].cumsum().ffill().fillna(0)
        df['scaled_equity'] = self.initial_capital + df['scaled_cum_gain_after_fee']
        
        return df


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
        equity = self.df['scaled_equity'].iloc[-1]
        year_no = self.year_count
 
        total_return = ((equity / self.initial_capital) - 1) * 100
        return_per_year = total_return / year_no

        cagr = ((equity / self.initial_capital) ** (1 / year_no) - 1)*100

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
       Total Return: {return_3[0]:.2f}%
      Annual Return: {return_3[1]:.2f}%
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
        equity = equity.resample('D').last().dropna()
        ret = (equity / equity.iloc[0] - 1) * 100
        
        axs[0].plot(ret.index, ret, label=f"Strategy (Sharpe_after_fee: {sharpe:.2f})", color="blue")
        
        # Make fonts larger for Return plot
        axs[0].set_title("Strategy Performance", fontsize=TITLE_SIZE)
        axs[0].set_ylabel("Return (%)", fontsize=LABEL_SIZE)
        axs[0].tick_params(axis='both', labelsize=TICK_SIZE)
        axs[0].legend(fontsize=LEGEND_SIZE, loc="upper left")
        axs[0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        peak = equity[equity!=0].cummax()
        daily_dd = (peak - equity)/fin_bt.available_capital * 100
        daily_dd = daily_dd.resample('D').last().dropna()
        
        axs[1].fill_between(daily_dd.index, daily_dd, 0, color='red', alpha=0.4, label="Drawdown")
        
        # Make fonts larger for Drawdown plot
        axs[1].set_ylabel("Drawdown %", fontsize=LABEL_SIZE)
        axs[1].set_xlabel("Date", fontsize=LABEL_SIZE)  # Added X-label for the bottom plot
        axs[1].tick_params(axis='both', labelsize=TICK_SIZE)
        axs[1].legend(fontsize=LEGEND_SIZE, loc="lower left")
        axs[1].grid(True, alpha=0.3)
        
        # Optional: Standard convention is to have drawdown go downwards
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
    df = pd.read_csv(r'C:\Users\HP\.0_PycharmProjects\VNMiniQuant_main\DataApi\cached_data\SuperMac.csv')
    rep = FinanceBacktest(fee_type='vn_future', 
                    currency='vnd', 
                    initial_capital=10_000_000_000, 
                    allocation_per_trade=1,
                    fixed_allocation=True,
                    risk_free_rate=0)

    out = rep.pnl_report(data=df, plot=False)

    
# TASK
# - Fee for vietnam market
# - Pos never -1
# - T+2.5 negate pos if change too frequent



# HINT for SYS
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import warnings
# from datetime import datetime
# from typing import Optional
# from dataclasses import dataclass

# # --- Font Size Configuration ---
# TITLE_SIZE = 15
# LABEL_SIZE = 16.5
# TICK_SIZE = 16.5
# LEGEND_SIZE = 16.5

# @dataclass
# class Fee:
#     fee = {
#         'vn_future': 0.4,
#         'vn_stock': 0.5,
#         'crypto': 0.00035,
#         'us_stock': 0.5,
#         'us_future': 0.3
#     }

# # ============================================================
# # MARKET MICROSTRUCTURE ENGINE (SCRUM-ready Strategy Pattern)
# # ============================================================

# class AssetRule(ABC):
#     """Strategy interface. New asset classes implement this; nothing else changes."""
#     @abstractmethod
#     def enforce_position_limits(self, target_position: float) -> float: ...

#     @abstractmethod
#     def get_tradable_inventory(self, current_date: datetime, inventory_ledger: list) -> float: ...

#     @abstractmethod
#     def process_corporate_actions(self, position: float, events: Optional[dict]) -> dict: ...


# class VNStockRule(AssetRule):
#     """
#     - No shorting: position floored at 0.
#     - T+2.5 settlement: a lot bought on T clears at 13:00 on T+2 (trading days).
#       Running on daily-close data, a lot is tradable once >=2 trading days have elapsed.
#     """
#     SETTLEMENT_DAYS = 2

#     def enforce_position_limits(self, target_position: float) -> float:
#         return max(0.0, target_position)

#     def get_tradable_inventory(self, current_date: datetime, inventory_ledger: list) -> float:
#         if not inventory_ledger:
#             return 0.0
#         purchase_dates = np.array(
#             [batch['date'].date() for batch in inventory_ledger], dtype='datetime64[D]'
#         )
#         shares = np.array([batch['shares'] for batch in inventory_ledger], dtype=float)
#         days_passed = np.busday_count(purchase_dates, np.datetime64(current_date.date()))
#         return float(shares[days_passed >= self.SETTLEMENT_DAYS].sum())

#     def process_corporate_actions(self, position: float, events: Optional[dict]) -> dict:
#         if not events:
#             return {'position': position, 'cash_gained': 0.0, 'ratio': 0.0}

#         event_type = events.get('type')
#         if event_type in ('stock_dividend', 'split'):
#             ratio = events.get('ratio', 0.0)
#             return {'position': position * (1 + ratio), 'cash_gained': 0.0, 'ratio': ratio}
#         if event_type == 'cash_dividend':
#             amount = events.get('amount', 0.0)
#             return {'position': position, 'cash_gained': position * amount, 'ratio': 0.0}
#         return {'position': position, 'cash_gained': 0.0, 'ratio': 0.0}


# class DefaultCryptoRule(AssetRule):
#     """Fallback rule for Instant T+0 markets like Crypto or Futures allowing shorting."""
#     def enforce_position_limits(self, target_position: float) -> float:
#         return target_position

#     def get_tradable_inventory(self, current_date: datetime, inventory_ledger: list) -> float:
#         return sum(batch['shares'] for batch in inventory_ledger)

#     def process_corporate_actions(self, position: float, events: Optional[dict]) -> dict:
#         return {'position': position, 'cash_gained': 0.0, 'ratio': 0.0}


# class MarketRule:
#     """Facade / Factory pattern coordinating structural matching."""
#     _REGISTRY = {
#         'VN_STOCK': VNStockRule,
#         'VN_FUTURE': DefaultCryptoRule,
#         'CRYPTO': DefaultCryptoRule,
#         'US_STOCK': DefaultCryptoRule,
#     }

#     def __init__(self, asset_type: str):
#         key = asset_type.upper()
#         if key not in self._REGISTRY:
#             raise ValueError(f"Asset type '{asset_type}' not supported yet.")
#         self.strategy: AssetRule = self._REGISTRY[key]()

#     def validate_trade(self, requested_pos_change: float, current_date: datetime, inventory_ledger: list) -> float:
#         """Caps a sell (negative) order to the settled/tradable balance. Buys pass through."""
#         if requested_pos_change >= 0:
#             return requested_pos_change
#         tradable = self.strategy.get_tradable_inventory(current_date, inventory_ledger)
#         return -min(abs(requested_pos_change), tradable)


# # =================================
# # Standardize Input
# # =================================
# class StandardizeInput:
#     REQUIRED_COLS = ["datetime", "close", "position"]
#     DATETIME_ALIASES = {"time": "datetime", "timestamp": "datetime", "date": "datetime", "datetime": "datetime"}
#     POSITION_ALIASES = {"pos": "position", "side": "position", "signal": "position", "sig": "position", "position": "position"}

#     @staticmethod
#     def column_std(df: pd.DataFrame) -> pd.DataFrame:
#         df = df.copy()
#         df.columns = [str(col).strip().lower() for col in df.columns]
#         rename_map = {}

#         for col in df.columns:
#             if col in StandardizeInput.DATETIME_ALIASES:
#                 rename_map[col] = "datetime"
#             elif col in StandardizeInput.POSITION_ALIASES:
#                 rename_map[col] = "position"

#         df = df.rename(columns=rename_map)
#         missing_cols = [col for col in StandardizeInput.REQUIRED_COLS if col not in df.columns]
#         if missing_cols:
#             raise ValueError(f"Missing required column(s): {', '.join(missing_cols)}")

#         try:
#             df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
#         except Exception as e:
#             raise ValueError(f"Failed to parse datetime column: {e}")

#         numeric_cols = [c for c in df.columns if c != "datetime" and c != "events"]
#         for col in numeric_cols:
#             try:
#                 df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
#             except Exception as e:
#                 raise ValueError(f"Column '{col}' cannot be converted to float: {e}")

#         nan_count = df[numeric_cols].isna().any(axis=1).sum()
#         if nan_count > 0:
#             warnings.warn(f"{nan_count} row(s) containing NaN values were dropped.", UserWarning)
#             df = df.dropna(subset=numeric_cols)

#         df = df.sort_values("datetime").reset_index(drop=True)
#         df = df.set_index('datetime')
#         return df


# # =================================
# # Finance metric calculation
# # =================================
# class FinanceMetrics:

#     def __init__(self,
#                  df: pd.DataFrame, 
#                  fee_type: str = 'vn_future', 
#                  initial_capital: float = 100_000_000,
#                  allocation_per_trade: float = 1.0,
#                  currency: str = 'VND', 
#                  risk_free_rate: float = 0.0
#                  ):
        
#         std_data = StandardizeInput.column_std(df)
#         self.trade_period = annual_sessions_in_days
#         self.rf_rate = risk_free_rate
#         self.currency = currency
#         self.initial_capital = initial_capital

#         if not (0 < allocation_per_trade <= 1):
#             raise ValueError('You must expose >0 and <=1 of your initial capital')
#         self.allocation_per_trade = allocation_per_trade

#         if self.currency.lower() == "usd":
#             self.initial_capital /= 26_000
#         elif self.currency.lower() not in ["vnd", "usd"]:
#             raise ValueError('currency only accepts "vnd" or "usd"')
        
#         if fee_type not in ['vn_future', 'crypto', 'vn_stock', 'us_stock']:
#             raise ValueError('Fee type only accepts: vn_stock, us_stock, vn_future, crypto')
        
#         self.available_capital = self.initial_capital * self.allocation_per_trade
#         if fee_type == "vn_future":
#             self.available_capital = self.available_capital / 100_000
#             if self.available_capital < std_data["close"].iloc[0]:
#                 raise ValueError(f'Not enough cash to buy 1 contract at {std_data["close"].iloc[0]}')
#         else:
#             if self.available_capital < std_data["close"].iloc[0]:
#                 raise ValueError(f'Not enough cash to buy 1 stock/unit at {std_data["close"].iloc[0]}')
            
#         self.one_way_fee = Fee().fee[fee_type]
#         self.fee_type = fee_type

#         # Initialize the agile MarketRule Engine
#         self.market_rule = MarketRule(asset_type=self.fee_type)

#         self.df = self.Gains_Calculation_Microstructure(std_data)
#         self.year_count = len(self.df.resample('D').sum(min_count=1).dropna()) / annual_sessions_in_days


#     def Gains_Calculation_Microstructure(self, df):
#         """
#         Chronologically simulates asset matching, enforcing T+2.5 rules, 
#         short-selling bans, and tracking corporate actions without breaking vector calculations.
#         """
#         # Prepare storage arrays
#         adjusted_positions = []
#         pos_changes = []
#         gains = []
#         fees = []
#         corporate_cash_gained = []
        
#         inventory_ledger = [] # Format: [{'date': datetime, 'shares': float}]
#         last_position = 0.0
        
#         # Chronological loop required to track path-dependent T+2.5 settlement arrays
#         for idx, row in enumerate(df.itertuples()):
#             current_date = row.Index
#             close_price = row.close
#             target_pos = row.position
            
#             # 1. Enforce No-Shorting Rule at input layer
#             target_pos = self.market_rule.strategy.enforce_position_limits(target_pos)
            
#             # 2. Check for Corporate Action Events on this current timestamp
#             event = getattr(row, 'events', None) if hasattr(row, 'events') else None
#             # Handle parsed string dict wrappers if present
#             if isinstance(event, str):
#                 import ast
#                 try: event = ast.literal_eval(event)
#                 except: event = None

#             corp_act = self.market_rule.strategy.process_corporate_actions(last_position, event)
            
#             # Apply adjustments to current track balances before determining tracking diffs
#             if corp_act.get('ratio', 0.0) > 0.0:
#                 ratio = corp_act['ratio']
#                 last_position *= (1 + ratio)
#                 for batch in inventory_ledger:
#                     batch['shares'] *= (1 + ratio)
            
#             cash_from_corp_actions = corp_act.get('cash_gained', 0.0)
#             corporate_cash_gained.append(cash_from_corp_actions)

#             # 3. Handle Trade Adjustments and T+2.5 Settlement Validation Rules
#             desired_change = target_pos - last_position
            
#             # Cap selling requests based on cleared inventory matching
#             validated_change = self.market_rule.validate_trade(desired_change, current_date, inventory_ledger)
#             actual_position = last_position + validated_change
            
#             # Update FIFO/LIFO Ledger tracking allocation units
#             if validated_change > 0:
#                 inventory_ledger.append({'date': current_date, 'shares': validated_change})
#             elif validated_change < 0:
#                 # Deduct inventory lots sequentially
#                 shares_to_deduct = abs(validated_change)
#                 while shares_to_deduct > 0 and inventory_ledger:
#                     if inventory_ledger[0]['shares'] <= shares_to_deduct:
#                         shares_to_deduct -= inventory_ledger[0]['shares']
#                         inventory_ledger.pop(0)
#                     else:
#                         inventory_ledger[0]['shares'] -= shares_to_deduct
#                         shares_to_deduct = 0

#             # 4. Compute Financial Metrics
#             if idx == 0:
#                 current_gain = 0.0
#             else:
#                 # Asset pricing drops during ex-date distributions are countered by captured cash outputs
#                 current_gain = last_position * (close_price - df['close'].iloc[idx - 1])
                
#             current_fee = self.one_way_fee * abs(validated_change)
            
#             adjusted_positions.append(actual_position)
#             pos_changes.append(validated_change)
#             gains.append(current_gain)
#             fees.append(current_fee)
            
#             last_position = actual_position

#         # Re-attach computed historical properties back into dataframe
#         df['position'] = adjusted_positions
#         df['pos_change'] = pos_changes
#         df['gain'] = gains
#         df['fee'] = fees
#         df['corp_cash'] = corporate_cash_gained
        
#         # Calculate Equity Paths accounting for fees and capital scaling metrics
#         df['gain_after_fee'] = df['gain'] - df['fee'] + df['corp_cash']
#         df['cum_gain_after_fee'] = df['gain_after_fee'].cumsum()
#         df['total_equity'] = self.available_capital + df['cum_gain_after_fee']

#         df['scaler'] = self.available_capital / df['close']
#         df['scaled_gain_after_fee'] = df['gain_after_fee'] * df['scaler']
#         df['scaled_cum_gain_after_fee'] = df['scaled_gain_after_fee'].cumsum()
#         df['scaled_equity'] = self.available_capital + df['scaled_cum_gain_after_fee']
        
#         return df

#     def Sharpe_after_fee(self):
#         daily_gain = (self.df['scaled_gain_after_fee'].resample('D').sum(min_count=1).dropna())
#         daily_close = (self.df['close'].resample('D').last().dropna())
#         daily_gain, daily_close = daily_gain.align(daily_close, join='inner')

#         if len(daily_gain) < 2:
#             return np.nan, np.nan
        
#         yearly_max = daily_close.groupby(daily_close.index.year).transform('max')
#         cash_max = yearly_max.mean()
#         year_total = self.year_count

#         daily_return = daily_gain / cash_max / year_total
#         daily_rf = (1 + self.rf_rate) ** (1 / self.trade_period) - 1

#         daily_ret = daily_return - daily_rf
#         std_ret = daily_ret.std()
#         std_loss_ret = daily_ret[daily_ret < 0].std()

#         if std_ret == 0 or np.isnan(std_ret):
#             return np.nan, np.nan

#         sharpe = (daily_ret.mean() / std_ret) * np.sqrt(self.trade_period)
#         sortino = (daily_ret.mean() / std_loss_ret) * np.sqrt(self.trade_period)
#         return sharpe, sortino

#     def MDD(self):
#         equity = self.df['scaled_equity']
#         peak = equity[equity != 0].cummax()
#         dd_abs = (peak - equity)
#         mdd_abs = dd_abs.max()
        
#         mdd_pct = ((peak - equity) / self.available_capital * 100).max()

#         mdd_trough_date = dd_abs.idxmax()
#         mdd_peak_date = equity.loc[:mdd_trough_date].idxmax()
#         mdd_date = f'Time: {mdd_peak_date} -> {mdd_trough_date}'

#         return mdd_abs, mdd_pct, mdd_date

#     def Total_Trade(self):
#         long_count = ((self.df['position'].shift(1).fillna(0).isin([0, -1])) & (self.df['position'] == 1)).sum()
#         short_count = ((self.df['position'].shift(1).fillna(0).isin([0, 1])) & (self.df['position'] == -1)).sum()
#         return long_count, short_count

#     def Profit(self):
#         final_gain = self.df['scaled_cum_gain_after_fee'].iloc[-1]
#         total_profit = final_gain
#         profit_after_fee_per_year = final_gain / self.year_count
#         profit_after_fee_per_day = final_gain / (self.year_count * self.trade_period)
#         return total_profit, profit_after_fee_per_year, profit_after_fee_per_day

#     def Return(self):
#         final_cap = self.df['scaled_equity'].iloc[-1]
#         total_ret = ((final_cap / self.available_capital) - 1)
#         year_no = self.year_count

#         total_return = total_ret * 100
#         return_per_year = total_return / year_no
#         cagr = ((final_cap / self.available_capital) ** (1 / year_no) - 1) * 100
#         return total_return, return_per_year, cagr
    
#     def Hitrate(self):
#         positions = self.df['position'].values
#         gains = self.df['gain_after_fee'].values
#         signs = np.sign(positions)
#         sign_changes = np.diff(signs, prepend=signs[0] + 1) != 0
#         block_ids = np.cumsum(sign_changes)
#         trade_gains = np.bincount(block_ids, weights=gains)[1:]
#         block_signs = signs[sign_changes]

#         long_mask = block_signs > 0
#         long_trades = np.sum(long_mask)
#         long_wins = np.sum(long_mask & (trade_gains > 0))

#         short_mask = block_signs < 0
#         short_trades = np.sum(short_mask)
#         short_wins = np.sum(short_mask & (trade_gains > 0))

#         long_hitrate = (long_wins / long_trades * 100) if long_trades > 0 else 0.0
#         short_hitrate = (short_wins / short_trades * 100) if short_trades > 0 else 0.0
#         hitrate_total = ((long_wins + short_wins) / (long_trades + short_trades) * 100) if (long_trades > 0 or short_trades > 0) else 0

#         return long_hitrate, short_hitrate, hitrate_total

#     def Longest_streak(self):
#         active_gains = self.df['gain_after_fee'].to_numpy()
#         if active_gains.size == 0:
#             return 0, 0

#         is_win = active_gains > 0
#         win_groups = np.cumsum(~is_win)[is_win]
#         longest_win_streak = int(np.bincount(win_groups).max()) if win_groups.size > 0 else 0

#         is_loss = active_gains < 0
#         loss_groups = np.cumsum(~is_loss)[is_loss]
#         longest_loss_streak = int(np.bincount(loss_groups).max()) if loss_groups.size > 0 else 0

#         return longest_win_streak, longest_loss_streak


# class FinanceBacktest:
#     def __init__(self, 
#                  fee_type: str, 
#                  initial_capital: float = 100_000_000, 
#                  allocation_per_trade: float = 1.0,
#                  currency: str = 'VND', 

#                  risk_free_rate: float = 0.0
#                  ):
#         self.fee_type = fee_type
#         self.initial_capital = initial_capital
#         self.allocation_per_trade = allocation_per_trade
#         self.currency = currency
#         self.trade_period = annual_sessions_in_days
#         self.risk_free_rate = risk_free_rate

#     def dashboard(self, data: pd.DataFrame):
#         fin_bt = FinanceMetrics(df=data, 
#                                 fee_type=self.fee_type, 
#                                 initial_capital=self.initial_capital, allocation_per_trade=self.allocation_per_trade,
#                                 currency=self.currency,
#                                 annual_sessions_in_days=self.trade_period,
#                                 risk_free_rate=self.risk_free_rate)

#         sharpe_and_sor = fin_bt.Sharpe_after_fee()
#         mdd_3 = fin_bt.MDD()
#         profit_3 = fin_bt.Profit()
#         return_3 = fin_bt.Return()
#         hitrate_2 = fin_bt.Hitrate()
#         trade_2 = fin_bt.Total_Trade()
#         streak_2 = fin_bt.Longest_streak()

#         return f"""
# ======================================================
#                  Financial Backtest 
# ======================================================
#     Initial capital: {fin_bt.available_capital:,.2f}
#      Ending capital: {fin_bt.df['scaled_equity'].iloc[-1]:,.2f}
#              Sharpe: {sharpe_and_sor[0]:.2f}
#             Sortino: {sharpe_and_sor[1]:.2f}
#                 MDD: {mdd_3[0]:,.2f} ({mdd_3[1]:.2f}%); {mdd_3[2]}
#        Total Profit: {profit_3[0]:,.2f}
#       Annual Profit: {profit_3[1]:,.2f}
#        Daily Profit: {profit_3[2]:,.2f}
#        Total Return: {return_3[0]:.2f}%
#       Annual Return: {return_3[1]:.2f}%
#                CAGR: {return_3[2]:.2f}%
#        Hitrate Long: {hitrate_2[0]:.2f}%
#       Hitrate Short: {hitrate_2[1]:.2f}%
#       Total Hitrate: {hitrate_2[2]:.2f}%
#  Longest win streak: {streak_2[0]}
# Longest lose streak: {streak_2[1]}
#         Long trades: {trade_2[0]}
#        Short trades: {trade_2[1]}
# """

#     def plot_equity(self, data: pd.DataFrame):
#         fin_bt = FinanceMetrics(df=data, 
#                                 fee_type=self.fee_type, 
#                                 initial_capital=self.initial_capital, allocation_per_trade=self.allocation_per_trade,
#                                 currency=self.currency,
#                                 annual_sessions_in_days=self.trade_period,
#                                 risk_free_rate=self.risk_free_rate)
        
#         figsize = (22, 10)
#         sharpe = fin_bt.Sharpe_after_fee()[0]
#         _, axs = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [6, 4]}, sharex=True)
        
#         equity = fin_bt.df['scaled_equity'][~fin_bt.df['scaled_equity'].isin([np.nan, np.inf, -np.inf])]
#         equity = equity.resample('D').last().dropna()
#         ret = (equity / equity.iloc[0] - 1) * 100
        
#         axs[0].plot(ret.index, ret, label=f"Strategy (Sharpe_after_fee: {sharpe:.2f})", color="blue")
#         axs[0].set_title("Strategy Performance", fontsize=TITLE_SIZE)
#         axs[0].set_ylabel("Return (%)", fontsize=LABEL_SIZE)
#         axs[0].tick_params(axis='both', labelsize=TICK_SIZE)
#         axs[0].legend(fontsize=LEGEND_SIZE, loc="upper left")
#         axs[0].grid(True, alpha=0.3)
        
#         peak = equity[equity != 0].cummax()
#         daily_dd = (peak - equity) / fin_bt.available_capital * 100
#         daily_dd = daily_dd.resample('D').last().dropna()
        
#         axs[1].fill_between(daily_dd.index, daily_dd, 0, color='red', alpha=0.4, label="Drawdown")
#         axs[1].set_ylabel("Drawdown %", fontsize=LABEL_SIZE)
#         axs[1].set_xlabel("Date", fontsize=LABEL_SIZE)
#         axs[1].tick_params(axis='both', labelsize=TICK_SIZE)
#         axs[1].legend(fontsize=LEGEND_SIZE, loc="lower left")
#         axs[1].grid(True, alpha=0.3)
#         axs[1].invert_yaxis()

#         plt.tight_layout()
#         plt.show()

#     def pnl_report(self, data: pd.DataFrame, plot=True):
#         dash = self.dashboard(data)
#         print(dash)
#         if plot:
#             self.plot_equity(data)


# if __name__ == "__main__":
#     # Example Generation containing raw execution paths & a corporate action event
#     dates = pd.bdate_range(start="2026-01-01", periods=10)
#     mock_df = pd.DataFrame({
#         'datetime': dates,
#         'close': [10000, 10200, 10100, 5100, 5200, 5300, 5400, 5500, 5450, 5600],
#         'position': [1, 2, 2, 2, 0, 0, 1, 1, 0, 0],
#         # Example corporate action: A 1:1 stock split on Day 4 (price halved from 10100 -> 5100)
#         'events': [None, None, None, {'type': 'split', 'ratio': 1.0}, None, None, None, None, None, None]
#     })
    
#     rep = FinanceBacktest(
#         fee_type='vn_stock', 
#         currency='vnd', 
#         initial_capital=100_000_000, 
#         allocation_per_trade=1,
#         risk_free_rate=0
#     )
    
#     rep.pnl_report(data=mock_df, plot=False)