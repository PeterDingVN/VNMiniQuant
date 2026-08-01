<div align="center">

# <div align="center"><b>VNMiniQuant</b></div>

*A lightweight quantitative toolkit for avid quantitative researcher*

[![Python](https://img.shields.io/badge/Python-3.13.5-blue?style=flat-square)](https://python.org)
[![Market](https://img.shields.io/badge/Market-Vietnam%20%7C%20US%20%7C%20Crypto-red?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square)]()

</div>

---

## 📌 Introduction

**VNMiniQuant** is a quant research workspace designed to help individual investors/traders develop and backtest their own alphas.
Develop your own alpha and add a config file into designated folder, then run the exe.ipynb. Holistic backtest from financial to statistical would be ready!

---

## 🚀 Usage

### SET UP

```bash
# clone the repo
git clone https://github.com/PeterDingVN/VNMiniQuant.git

# Get into the main work dir
cd VNMiniQuant

# install dependencies
pip install -r requirements.txt
```


### QUICK START
#### Step 1: Build Alpha
- Develop your idea for an alpha. For example, when close > EMA10 go long, else go short.
- From the idea, access ```alpha_sample/MyAlpha.py``` and follow the instruction about an alpha's standard structure.
- Convert your idea into code following that structure.


#### Step 2: Configurate the config file
- Auto pick file with "cfg" or "config" in the name if multiple .json found in ```alpha_sample/```. Named otherwise, only one .json is accepted.
- Config files must have all the keys: ```"tv_username", "tv_password", "update_data", "data", "bt_cfg", "alpha_cfg"```:
  + tv_username and tv_password: used to log into TradingView account, which may provides more data than Guest Mode.
  + update_data: used to tell DataApi whether to scrape new data or reuse data scraped and stored in cache previously. However, auto scrape is still triggered if the data mentioned does not exist in cache.
  + bt_cfg: provide config for backtets engine. Please pay attention to fee_type when backtesting different assets (crypto fee differ from VN future fee). Sometimes, error about "not enough cash" is raised, if so, go increase the init_capital. Other than that, default setting works fine.
  + alpha_cfg: 
    - filename: provide exactly the name of .py file that contains the alpha
    - classname: provide exactly the name of the alpha class that contains core logic (please check out alpha file ```alpha_sample/MyAlpha.py``` for details)
    - alpha_type: currently, only ```ta``` is supported, so leave the default config as it is
    - params: the parameters for your alpha, check out the alpha file for more.
- For an example, check out sample ```MyAlphaCfg.json```.

#### Step 3: Run the alpha
- Go into Research_Space.ipynb and follow the process. Enjoy!


### OUTPUT

```
 =========== Financial Backtest Vietnam Future ==========

    Initial capital: 260,000,000.00
     Ending capital: -886,495,789.01
             Sharpe: -7.05
            Sortino: -9.45
             Calmar: -0.14
                MDD: 1,147,499,362.52 (441.35%); Time: 2018-01-02 09:00:00 -> 2023-12-06 14:10:00
       Total Profit: -1,146,495,789.01
   Margin per Trade: -4.03 bps
       Total Return: -371.56%
 Mean Annual Return: -63.09%
Comp. Annual Return: nan%
       Hitrate Long: 4.87%
      Hitrate Short: 8.20%
      Total Hitrate: 6.82%
        Longest Win: 7 days
       Longest Loss: 24 days
      Trade per Day: 3.55
        Long Trades: 2255
       Short Trades: 3017
```
![alt text](_img/image.png)



---

## ⚠️ Disclaimer

- **Data source**: Market data is fetched mainly via Binance, TradingView, Vietstock
- **Timeframe limitation**: Only **daily** OHLCV data is available through the public pipeline. Some additional data used internally (intraday, alternative datasets) comes from private sources and **cannot be published or redistributed**.
- **Not financial advice**: This tool is pure research and backtest environment. No alphas or trading strategies are provided within.


---

## 🔭 Future Roadmap

The following improvements are planned for future releases:
- Add PaperTrading module to allow real-life stress-test the alpha
- Add LiveTrading module and API connection instruction
- Add AlphaDashboard to monitor trades and portfolio performance while in paper trading and live.

