import sys
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from datetime import timedelta, time, date
from time import mktime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from typing import Optional, Tuple, Dict, Union, List
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

import matplotlib.dates as mdates
from matplotlib import gridspec
import matplotlib.cm as cm

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and format date/time columns in the DataFrame."""
    df['Datetime'] = pd.to_datetime(df.Date)
    df = df.sort_values(by='Datetime')
    df['Date'] = df.Datetime.dt.date
    df['Date'] = df['Date'].apply(lambda x: x.strftime("%Y-%m-%d"))
    df['time'] = df.Datetime.dt.time
    df['time'] = df['time'].apply(lambda x: x.strftime('%H:%M:%S'))
    return df

def resample(df: pd.DataFrame, sample_duration: int, type_data: str) -> pd.DataFrame:
    """Resample OHLCV data by a given duration and type."""
    df.Date = pd.to_datetime(df.Date)
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    df = pd.DataFrame(df.resample(f'{sample_duration}{type_data}', on='Date', label='left').apply(ohlc_dict).dropna()).reset_index()
    return df

# =====================
# MAIN BACKTEST CLASS: SINGLE + MULTI SYMBOL + PORTFOLIO + PYRAMIDING
# =====================
class BacktestEngine:
    """
        Vectorized Backtest Engine - Single & Multi-Symbol

        Hỗ trợ:
            • Pyramiding (DCA)
            • Binance Futures: Leverage, Margin, Liquidation, Funding
            • Multi-symbol portfolio
            • Export trade log → QuantStats

        Cách dùng:
            bt = BacktestEngine(...)
            bt.analyze()          # Metrics + Dashboard + Heatmap
            bt.export_trade_log() # CSV

        Parameters
        ----------
        Datetime : pd.Series or pd.DatetimeIndex
            Chung cho tất cả symbol (UTC).
        Position : pd.Series or Dict[str, pd.Series]
            Signal: 0=flat, >0=long, <0=short. Pyramiding: 0.3, 0.6, 1.0...
        Close : pd.Series or Dict[str, pd.Series]
            Giá đóng cửa khớp Datetime.
        fee : float, default 0.0004
            Phí round-trip (0.04%).
        use_pyramiding : bool, default True
            True = DCA, False = -1/0/1.
        run_portfolio : bool, default True
            True = Binance sim, False = vectorized.
        initial_capital : float, default 20000.0
            Vốn ban đầu.
        leverage : float, default 10.0
            Đòn bẩy.
        maintenance_ratio : float, default 0.005
            Tỷ lệ duy trì margin.
        contract_size : float, default 1.0
            Kích thước hợp đồng.
        margin_buffer : float, default 0.01
            Buffer margin.
        use_funding_rate : bool, default True
            Bật funding.
        funding_rate : float | pd.Series | Dict, default 0.0001
            Funding rate 8h.
        use_binance_netting : bool, default True
            Phí trên new position.
        alloc_per_trade : float | Dict, default 100000.0
            Notional mỗi thay đổi.
        hedge_type : {'notional', 'unit'}, default 'notional'
            Cách scale position.
        symbols : List[str] | None, default None
            Tên symbol (auto nếu None).

        Examples
        --------
        # 1. SINGLE SYMBOL - SIMPLE VECTORIZED (như cũ)
        >>> bt = BacktestEngine(
        ...     Datetime=df['Datetime'],
        ...     Position=signal_series,           # pd.Series: [0, 0.5, 1.0, 0]
        ...     Close=df['Close'],
        ...     fee=0.0004,
        ...     use_pyramiding=True,
        ...     run_portfolio=False,
        ...     initial_capital=20000.0,
        ...     leverage=10.0,
        ...     maintenance_ratio=0.005,
        ...     contract_size=1.0,
        ...     margin_buffer=0.01,
        ...     use_funding_rate=False,
        ...     funding_rate=0.0,
        ...     use_binance_netting=True,
        ...     alloc_per_trade=100000.0,
        ...     hedge_type='notional',
        ...     symbols=None
        ... )
        >>> bt.analyze()

        # 2. SINGLE SYMBOL - BINANCE FUTURES (full realism)
        >>> bt = BacktestEngine(
        ...     Datetime=dt_index,
        ...     Position=pos_btc,
        ...     Close=close_btc,
        ...     fee=0.0004,
        ...     use_pyramiding=True,
        ...     run_portfolio=True,
        ...     initial_capital=10000.0,
        ...     leverage=20.0,
        ...     maintenance_ratio=0.005,
        ...     contract_size=1.0,
        ...     margin_buffer=0.01,
        ...     use_funding_rate=True,
        ...     funding_rate=0.00015,             # 0.015% mỗi 8h
        ...     use_binance_netting=True,
        ...     alloc_per_trade=50000.0,
        ...     hedge_type='notional',
        ...     symbols=['BTCUSDT']
        ... )
        >>> bt.analyze()

        # 3. MULTI-SYMBOL PORTFOLIO (BTC + ETH + SOL)
        >>> Positions = {
        ...     'BTCUSDT': pos_btc,
        ...     'ETHUSDT': pos_eth,
        ...     'SOLUSDT': pos_sol
        ... }
        >>> Closes = {
        ...     'BTCUSDT': close_btc,
        ...     'ETHUSDT': close_eth,
        ...     'SOLUSDT': close_sol
        ... }
        >>> funding_rates = {
        ...     'BTCUSDT': 0.00012,
        ...     'ETHUSDT': 0.00008,
        ...     'SOLUSDT': 0.00010
        ... }
        >>> bt = BacktestEngine(
        ...     Datetime=common_datetime,
        ...     Position=Positions,
        ...     Close=Closes,
        ...     fee=0.0004,
        ...     use_pyramiding=True,
        ...     run_portfolio=True,
        ...     initial_capital=50000.0,
        ...     leverage=15.0,
        ...     maintenance_ratio=0.005,
        ...     contract_size=1.0,
        ...     margin_buffer=0.01,
        ...     use_funding_rate=True,
        ...     funding_rate=funding_rates,
        ...     use_binance_netting=True,
        ...     alloc_per_trade={
        ...         'BTCUSDT': 60000.0,
        ...         'ETHUSDT': 30000.0,
        ...         'SOLUSDT': 10000.0
        ...     },
        ...     hedge_type='notional',
        ...     symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        ... )
        >>> bt.analyze()
        >>> bt.export_trade_log('portfolio_log.csv')
    """
    def __init__(
        self,
        Datetime: Union[pd.Series, pd.DatetimeIndex],
        Position: Union[pd.Series, Dict[str, pd.Series]],   # SINGLE or {sym: series}
        Close: Union[pd.Series, Dict[str, pd.Series]],      # SINGLE or {sym: series}
        fee: float = 0.00088,  # 0.04% total round-trip → /2 = 0.02% per side
        
        # === YÊU CẦU 1: PYRAMIDING ===
        use_pyramiding: bool = True,  # True: DCA | False: Force -1/0/1
        
        # === YÊU CẦU 2: PORTFOLIO SIMULATION ===
        run_portfolio: bool = False,   # True: Full Binance sim | False: Simple vectorized
        
        # === PORTFOLIO PARAMS (Binance Futures) ===
        initial_capital: float = 10000.0,
        leverage: float = 10.0,
        maintenance_ratio: float = 0.005,  # 0.5%
        contract_size: float = 1.0,        # BTC=1, ETH=1, etc.
        margin_buffer: float = 0.01,       # 1% buffer
        use_funding_rate: bool = True,
        funding_rate: Union[float, pd.Series, Dict[str, Union[float, pd.Series]]] = 0.0001,  # 0.01%
        use_binance_netting: bool = True,  # True: Fee on new_pos | False: on delta
        alloc_per_trade: Union[float, Dict[str, float]] = 100000.0,  # Notional per signal change
        hedge_type: str = 'notional',      # 'notional' or 'unit'
        
        # === MULTI-SYMBOL CONTROL ===
        symbols: Optional[List[str]] = None  # Auto-infer if None
    ):
        # =====================
        # 1. INPUT NORMALIZATION
        # =====================
        self.fee = fee / 2  # One-way fee
        self.use_pyramiding = use_pyramiding
        self.run_portfolio = run_portfolio
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.maintenance_ratio = maintenance_ratio
        self.contract_size = contract_size
        self.margin_buffer = margin_buffer
        self.use_funding_rate = use_funding_rate
        self.use_binance_netting = use_binance_netting
        self.hedge_type = hedge_type
        
        # =====================
        # 1. DATETIME & ALIGNMENT (ROBUST)
        # =====================
        # Chuyển thành Series để xử lý an toàn
        # =====================
        # 1. DATETIME & ALIGNMENT (ROBUST - FIX 100% ALL CASES)
        # =====================
        # BƯỚC 1: Chuyển thành Series (bắt buộc để dùng .reset_index())
        if isinstance(Datetime, pd.Index):
            datetime_series = pd.Series(Datetime.values, index=Datetime, name='Datetime')
        else:
            datetime_series = pd.Series(Datetime, name='Datetime')
        
        # BƯỚC 2: Chuyển thành datetime
        datetime_series = pd.to_datetime(datetime_series, errors='coerce', utc=True)
        
        # BƯỚC 3: Ffill/bfill
        if datetime_series.isna().any():
            datetime_series = datetime_series.ffill().bfill()
        
        # BƯỚC 4: drop_duplicates + sort + reset → LUÔN LÀ SERIES VỚI RangeIndex
        datetime_series = (
            datetime_series
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        
        # BƯỚC 5: Tạo DatetimeIndex sạch
        datetime_idx = pd.DatetimeIndex(datetime_series.values, name='Datetime')
 
        # =====================
        # 2. SYMBOL SETUP (Single → Multi) - ROBUST REINDEX
        # =====================
        if isinstance(Position, pd.Series) and isinstance(Close, pd.Series):
            # SINGLE SYMBOL MODE
            symbols = symbols or ['DEFAULT']
            self.symbols = symbols
            self.is_multi = False
 
            # Reindex Position & Close to common datetime_idx
            Position_aligned = Position.reindex(datetime_idx, method='ffill')
            Close_aligned = Close.reindex(datetime_idx, method='ffill')
 
            # Gán lại index đúng
            Position_aligned.index = datetime_idx
            Close_aligned.index = datetime_idx
 
            Positions = {symbols[0]: Position_aligned}
            Closes = {symbols[0]: Close_aligned}
 
        else:
            # MULTI SYMBOL MODE
            if not isinstance(Position, dict) or not isinstance(Close, dict):
                raise ValueError("For multi-symbol, pass dicts: {'BTC': pos, 'ETH': pos}")
            symbols = symbols or list(Position.keys())
            if set(symbols) != set(Position.keys()) or set(symbols) != set(Close.keys()):
                raise ValueError("Symbols must match keys in Position and Close dicts")
            self.symbols = symbols
            self.is_multi = True
 
            Positions = {}
            Closes = {}
            for sym in symbols:
                pos = Position[sym]
                close = Close[sym]
 
                # Reindex to common datetime_idx
                pos_aligned = pos.reindex(datetime_idx, method='ffill')
                close_aligned = close.reindex(datetime_idx, method='ffill')
 
                # Gán lại index đúng
                pos_aligned.index = datetime_idx
                close_aligned.index = datetime_idx
 
                Positions[sym] = pos_aligned
                Closes[sym] = close_aligned
        
        # =====================
        # 3. PYRAMIDING CONTROL
        # =====================
        if not use_pyramiding:
            for sym in symbols:
                Positions[sym] = np.sign(Positions[sym]) * 1.0
        
        # =====================
        # 4. SCALE POSITION (Notional/Unit)
        # =====================
        alloc_dict = alloc_per_trade if isinstance(alloc_per_trade, dict) else {sym: alloc_per_trade for sym in symbols}
        for sym in symbols:
            close = Closes[sym]
            alloc = alloc_dict.get(sym, 100000.0)
            if hedge_type == 'notional':
                scale = alloc / close
            elif hedge_type == 'unit':
                scale = alloc / close.iloc[0]  # Fixed at first price
            else:
                raise ValueError("hedge_type must be 'notional' or 'unit'")
            Positions[sym] = Positions[sym] * scale
        
        # =====================
        # 5. BUILD MAIN DF
        # =====================
        df = pd.DataFrame({'Datetime': datetime_idx})
        for sym in symbols:
            df[f'Position_{sym}'] = Positions[sym].values
            df[f'Close_{sym}'] = Closes[sym].values
        df = df.dropna().reset_index(drop=True)
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        
        # =====================
        # 6. FUNDING RATE HANDLER
        # =====================
        if use_funding_rate:
            if isinstance(funding_rate, (float, int)):
                funding_dict = {sym: pd.Series(funding_rate, index=df['Datetime']) for sym in symbols}
            elif isinstance(funding_rate, pd.Series):
                funding_dict = {sym: funding_rate.reindex(df['Datetime'], method='ffill') for sym in symbols}
            elif isinstance(funding_rate, dict):
                funding_dict = {}
                for sym in symbols:
                    rate = funding_rate.get(sym, 0.0001)
                    if isinstance(rate, (float, int)):
                        funding_dict[sym] = pd.Series(rate, index=df['Datetime'])
                    else:
                        funding_dict[sym] = rate.reindex(df['Datetime'], method='ffill')
            else:
                funding_dict = {sym: pd.Series(0.0001, index=df['Datetime']) for sym in symbols}
            for sym in symbols:
                df[f'funding_rate_{sym}'] = funding_dict[sym].values
        else:
            for sym in symbols:
                df[f'funding_rate_{sym}'] = 0.0
        
        # Funding multiplier (8H → x3 daily)
        freq = pd.infer_freq(df['Datetime'])
        df['funding_multiplier'] = 3.0 if freq in ['D', '1D'] and '8H' in str(funding_rate) else 1.0
        
        # =====================
        # 7. INIT COLUMNS
        # =====================
        for sym in symbols:
            df[f'input_pos_{sym}'] = 0.0
            df[f'gain_{sym}'] = 0.0
            df[f'fee_cost_{sym}'] = 0.0
        df['gain'] = 0.0
        df['fee_cost'] = 0.0
        df['funding_pnl'] = 0.0
        df['gain_after_fee'] = 0.0
        df['total_gain_after_fee'] = 0.0
        df['total_funding'] = 0.0
        df['total_gain_after_fee_funding'] = 0.0
        df['equity'] = initial_capital
        df['notional_total'] = 0.0
        df['initial_margin'] = 0.0
        df['maintenance_margin'] = 0.0
        df['insufficient_margin_flag'] = False
        df['liquidation_flag'] = False
        
        # =====================
        # 8. RUN BACKTEST
        # =====================
        n = len(df)
        if n < 2:
            print("Data too short")
            self.df = df.set_index('Datetime')
            self.df2 = pd.DataFrame()
            return
        
        if not run_portfolio:
            # === SIMPLE VECTORIZED MODE (OLD LOGIC) ===
            for sym in symbols:
                pos = df[f'Position_{sym}']
                df[f'input_pos_{sym}'] = pos.diff().fillna(pos.iloc[0])
                df[f'gain_{sym}'] = pos.shift(1).fillna(0) * df[f'Close_{sym}'].diff()
                df[f'fee_cost_{sym}'] = self.fee * df[f'input_pos_{sym}'].abs()
            df['gain'] = sum(df[f'gain_{sym}'] for sym in symbols)
            df['fee_cost'] = sum(df[f'fee_cost_{sym}'] for sym in symbols)
            df['gain_after_fee'] = df['gain'] - df['fee_cost']
            df['total_gain_after_fee'] = df['gain_after_fee'].cumsum()
            df['equity'] = initial_capital + df['total_gain_after_fee']
        else:
            # === FULL PORTFOLIO SIMULATION ===
            equity = initial_capital
            for i in range(1, n):
                # 1. Calculate changes
                changing_syms = []
                total_required = 0.0
                for sym in symbols:
                    input_pos = df.iloc[i][f'Position_{sym}'] - df.iloc[i-1][f'Position_{sym}']
                    df.iloc[i, df.columns.get_loc(f'input_pos_{sym}')] = input_pos
                    if input_pos != 0:
                        changing_syms.append(sym)
                        # Fee estimate
                        trade_vol = abs(df.iloc[i][f'Position_{sym}']) if use_binance_netting else abs(input_pos)
                        fee_est = self.fee * trade_vol * df.iloc[i][f'Close_{sym}'] * contract_size
                        # Notional change
                        notional_old = abs(df.iloc[i-1][f'Position_{sym}']) * df.iloc[i-1][f'Close_{sym}'] * contract_size
                        notional_new = abs(df.iloc[i][f'Position_{sym}']) * df.iloc[i][f'Close_{sym}'] * contract_size
                        net_inc = max(0, notional_new - notional_old)
                        required = (net_inc / leverage) + (notional_new / leverage * maintenance_ratio) + fee_est
                        total_required += required
                
                # 2. Margin check
                required_total = total_required * (1 + margin_buffer)
                if equity < required_total:
                    df.iloc[i, df.columns.get_loc('insufficient_margin_flag')] = True
                    for sym in changing_syms:
                        df.iloc[i, df.columns.get_loc(f'Position_{sym}')] = df.iloc[i-1][f'Position_{sym}']
                    continue
                
                # 3. Calculate PNL
                gain_total = 0.0
                fee_total = 0.0
                funding_total = 0.0
                for sym in symbols:
                    prev_pos = df.iloc[i-1][f'Position_{sym}']
                    close_diff = df.iloc[i][f'Close_{sym}'] - df.iloc[i-1][f'Close_{sym}']
                    df.iloc[i, df.columns.get_loc(f'gain_{sym}')] = prev_pos * close_diff * leverage * contract_size
                    gain_total += df.iloc[i][f'gain_{sym}']
                    
                    trade_vol = abs(df.iloc[i][f'Position_{sym}']) if use_binance_netting else abs(df.iloc[i][f'input_pos_{sym}'])
                    df.iloc[i, df.columns.get_loc(f'fee_cost_{sym}')] = self.fee * trade_vol * df.iloc[i][f'Close_{sym}'] * contract_size
                    fee_total += df.iloc[i][f'fee_cost_{sym}']
                    
                    funding_total -= df.iloc[i][f'Position_{sym}'] * df.iloc[i][f'Close_{sym}'] * df.iloc[i][f'funding_rate_{sym}'] * contract_size * df.iloc[i]['funding_multiplier']
                
                df.iloc[i, df.columns.get_loc('gain')] = gain_total
                df.iloc[i, df.columns.get_loc('fee_cost')] = fee_total
                df.iloc[i, df.columns.get_loc('funding_pnl')] = funding_total
                df.iloc[i, df.columns.get_loc('gain_after_fee')] = gain_total - fee_total
                equity += gain_total - fee_total + funding_total
                df.iloc[i, df.columns.get_loc('equity')] = equity
            
            # Cumsum
            df['total_gain_after_fee'] = df['gain_after_fee'].cumsum()
            df['total_funding'] = df['funding_pnl'].cumsum()
            df['total_gain_after_fee_funding'] = df['total_gain_after_fee'] + df['total_funding']
        
        # Margin calculations
        notional_total = pd.Series(0.0, index=df.index)
        for sym in symbols:
            notional_total += np.abs(df[f'Position_{sym}']) * df[f'Close_{sym}'] * contract_size
 
        df['notional_total']      = notional_total
        df['initial_margin']     = df['notional_total'] / leverage
        df['maintenance_margin'] = df['initial_margin'] * maintenance_ratio
        df['liquidation_flag']   = df['equity'] < df['maintenance_margin']
 
        # =====================
        # 10. FINAL SETUP
        # =====================
        df = df.set_index('Datetime')
 
        # df2 = chỉ những bar có ít nhất 1 symbol thay đổi position
        change_mask = pd.Series(False, index=df.index)
        for sym in symbols:
            change_mask |= (df[f'input_pos_{sym}'] != 0)
        self.df2 = df[change_mask].copy()
 
        self.df = df
        
    # =====================
    # EXPORT TRADE LOG
    # =====================
    def export_trade_log(self, filename: str = 'trade_log.csv', datetime_as_index: bool = True) -> None:
        df = self.df.copy()
        returns = df['equity'].pct_change().fillna(0)
        cum_return = (df['equity'] - self.initial_capital) / self.initial_capital * 100
        log = pd.DataFrame({'returns': returns, 'cumulative_return': cum_return})
        for sym in self.symbols:
            log[f'position_{sym}'] = df[f'Position_{sym}']
            log[f'close_{sym}'] = df[f'Close_{sym}']
        log['funding_pnl'] = df['funding_pnl']
        log['fee_cumulative'] = df['fee_cost'].cumsum()
        if not datetime_as_index:
            log = log.reset_index()
        log.to_csv(filename, index=datetime_as_index)
        print(f"Exported to {filename}")
 
    # =====================
    # PLOTS & METRICS (Full)
    # =====================
    def _calculate_buy_hold(self) -> Tuple[pd.Series, float]:
        df = self.df.copy()
        avg_close = sum(df[f'Close_{sym}'] for sym in self.symbols) / len(self.symbols)
        initial = avg_close.iloc[0]
        bh = (avg_close - initial) / initial
        bh_daily = bh.resample("1D").last().dropna()
        diff = bh_daily.diff().fillna(0)
        sharpe = (diff.mean() / diff.std()) * np.sqrt(365) if diff.std() != 0 else 0
        return bh_daily * 100, sharpe
    
    def Number_of_trade(self) -> int:
        return len(self.df2)
        
    def Hitrate(self) -> Tuple[float, float]:
        """Calculate hitrate separately for long and short trades.
        Returns:
            Tuple[float, float]: (long_hitrate, short_hitrate) as percentages
        """
        df = self.df2.copy()
        
        # Identify long and short trades based on positions
        long_trades = 0
        short_trades = 0
        long_wins = 0
        short_wins = 0
        
        for sym in self.symbols:
            pos_col = f'Position_{sym}'
            mask = df[pos_col] != 0  # Only count when there's a position
            
            # Long trades
            long_mask = (df[pos_col] > 0) & mask
            long_trades += long_mask.sum()
            long_wins += ((df['gain_after_fee'] > 0) & long_mask).sum()
            
            # Short trades
            short_mask = (df[pos_col] < 0) & mask
            short_trades += short_mask.sum()
            short_wins += ((df['gain_after_fee'] > 0) & short_mask).sum()
        
        # Calculate hitrates (as percentages)
        long_hitrate = (long_wins / long_trades * 100) if long_trades > 0 else 0
        short_hitrate = (short_wins / short_trades * 100) if short_trades > 0 else 0
        
        return long_hitrate, short_hitrate
    
    def plot_dashboard(self, figsize: Tuple[int, int] = (15, 12)) -> None:
        df = self.df.copy()
        equity = df['equity']
        ret = (equity / equity.iloc[0] - 1) * 100
        peak = equity.cummax()
        dd = (peak - equity) / peak * 100
        
        daily_ret = ret.resample("1D").last().dropna()
        daily_dd = dd.resample("1D").last().dropna()
        bh_perc, bh_sharpe = self._calculate_buy_hold()
        sharpe = self.Sharp_after_fee_funding()
        
        fig, axs = plt.subplots(5, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1, 2, 1, 1]}, sharex=True)
        
        # 1. Return
        axs[0].plot(daily_ret.index, daily_ret, label=f"Strategy (Sharpe: {sharpe:.2f})", color="orange")
        # axs[0].plot(bh_perc.index, bh_perc, label=f"B&H (Sharpe: {bh_sharpe:.2f})", color="gray", linestyle="--")
        axs[0].set_ylabel("Return (%)")
        axs[0].legend(); axs[0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        axs[1].fill_between(daily_dd.index, daily_dd, 0, color='red', alpha=0.4)
        axs[1].set_ylabel("Drawdown (%)"); axs[1].grid(True, alpha=0.3)
        
        # 3. Positions
        max_p = max(df[f'Position_{sym}'].abs().max() for sym in self.symbols)
        colors = sns.color_palette("husl", len(self.symbols))
        for idx, sym in enumerate(self.symbols):
            pos = df[f'Position_{sym}'].resample("1D").last().dropna()
            axs[2].fill_between(pos.index, 0, pos, where=(pos > 0), color=colors[idx], alpha=0.6, label=f'{sym} Long')
            axs[2].fill_between(pos.index, 0, pos, where=(pos < 0), color=colors[idx], alpha=0.3, label=f'{sym} Short')
        axs[2].axhline(0, color='black', linewidth=0.8)
        axs[2].set_ylabel("Position"); axs[2].legend(); axs[2].grid(True, alpha=0.3)
        axs[2].set_ylim(-max_p-0.5, max_p+0.5)
        axs[2].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        
        # 4. Margin Ratio
        margin_ratio = (df['equity'] / df['initial_margin'].replace(0, np.inf)).resample("1D").last().dropna()
        axs[3].plot(margin_ratio.index, margin_ratio, color="blue")
        axs[3].axhline(1, color='green', linestyle='--', label="Safe")
        axs[3].axhline(self.maintenance_ratio, color='red', linestyle='--', label="Maintenance")
        axs[3].set_ylabel("Margin Ratio"); axs[3].legend(); axs[3].grid(True, alpha=0.3)
        
        # 5. PNL Contribution
        contrib = pd.DataFrame({sym: df[f'gain_{sym}'].resample("1D").sum() for sym in self.symbols})
        contrib.cumsum().plot(ax=axs[4], title="PNL Contribution per Symbol")
        axs[4].grid(True, alpha=0.3)
        
        for ax in axs: ax.set_xlim(daily_ret.index.min(), daily_ret.index.max())
        axs[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.tight_layout(); plt.show()

    def analyze(self, figsize: Tuple[int, int] = (15, 12)) -> None:
        self.metrics()
        print("\n" + "="*60 + "\n")
        self.plot_dashboard(figsize)
        print("\n" + "="*40 + " Monthly Returns " + "="*40 + "\n")
        self.plot_monthly_returns_heatmap()

    def metrics(self, plot: bool = True) -> None:
        df = self.df
        final_equity = df['equity'].iloc[-1]
        total_ret = (final_equity - self.initial_capital) / self.initial_capital * 100
        liq = df['liquidation_flag'].sum()
        insuff = df['insufficient_margin_flag'].sum()
        long_hr, short_hr = self.Hitrate()
        data = [
            ('Symbols', ', '.join(self.symbols)),
            ('Initial Capital', f"${self.initial_capital:,.0f}"),
            ('Final Equity', f"${final_equity:,.2f}"),
            ('Total Return %', f"{total_ret:.2f}%"),
            ('Liquidations', liq),
            ('Insufficient Margin', insuff),
            ('MDD %', f"{self.MDD()[1]:.2f}%"),
            ('Sharpe', f"{self.Sharp_after_fee_funding():.2f}"),
            ('Number of Trades', self.Number_of_trade()),
            ('Long Hitrate %', f"{long_hr:.2f}%"),
            ('Short Hitrate %', f"{short_hr:.2f}%"),
        ]
        for k, v in data:
            print(f"{k:>25}: {v}")

    def MDD(self) -> Tuple[float, float]:
        equity = self.df['equity']
        peak = equity.cummax()
        dd_abs = peak - equity
        dd_rel = (peak - equity) / peak * 100
        return dd_abs.max(), dd_rel.max()

    def Sharp_after_fee_funding(self) -> float:
        daily = self.df['equity'].resample("1D").last().ffill().dropna()
        ret = daily.pct_change().dropna()
        return (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0

    def plot_monthly_returns_heatmap(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        daily = self.df['equity'].resample("1D").last().ffill().dropna()
        monthly = daily.pct_change().resample("ME").sum() * 100
        years = sorted(monthly.index.year.unique())
        months = range(1, 13)
        heat = pd.DataFrame(index=years, columns=months, dtype=float).fillna(0)
        for idx, val in monthly.items():
            heat.loc[idx.year, idx.month] = val
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(heat, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
        ax.set_title('Monthly Returns Heatmap (%)')
        ax.set_xlabel('Month'); ax.set_ylabel('Year')
        ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        plt.tight_layout(); plt.show()