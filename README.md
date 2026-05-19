<div align="center">

# <div align="center"><b>VNMiniQuant</b></div>

*A lightweight quantitative toolkit for Vietnamese stock market research*

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)](https://python.org)
[![Data](https://img.shields.io/badge/Data-vnstock-green?style=flat-square)](https://github.com/thinh-vu/vnstock)
[![Market](https://img.shields.io/badge/Market-HOSE%20%7C%20HNX%20%7C%20UPCOM-red?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square)]()

</div>

---

## 📌 Introduction

**VNMiniQuant** is a personal quantitative research tool built to automate alpha validation step.

The core motivation is simple: individual investors in Vietnam rarely have access to institutional-grade backtesting infrastructure. 

This tool bridges that gap — allowing individuals to **backtest it on historical data, and validate whether it holds statistical merit** before risking real capital.


---

## 🚀 Usage


### Quick start

COPY CODE

```bash
# clone the repo
git clone https://github.com/PeterDingVN/VNMiniQuant.git

cd VNMiniQuant

# install dependencies
pip install -r requirements.txt

# Quick start to test the output
python -m core.exe
```

OUTPUT

```
Training Data up-to-date for VN30F1M is ready!
START W4W TRAINING
START STAT TEST
 
  Result can be optimistic from reality  
============= STRATEGY RESULT ============
Strategy validity pval: 1.0
Return per year: -1.124%
Sharpe: -0.839
MDD: -1.314%
```

After this you can start with your first strategy (alpha) by preparing a .py file that returns dataframe with 'time', 'close', 'position' (name must be precise)

Refer to ```strategy_sample\MyAlpha.py``` for reference
And look at ```config\settings.py``` for params setting idea


---

## ⚠️ Disclaimer

- **Data source**: Market data is fetched via [`vnstock`](https://github.com/thinh-vu/vnstock), an open-source library for Vietnamese stock data.
- **Timeframe limitation**: Only **daily** OHLCV data is available through the public pipeline. Some additional data used internally (intraday, alternative datasets) comes from private sources and **cannot be published or redistributed**.
- **Not financial advice**: This tool is built for research and educational purposes. Nothing produced by VNMiniQuant constitutes investment advice. Always do your own due diligence.

---

## 🔭 Future Roadmap

The following improvements are planned for future releases:

- **Statistical testing** — expanded suite of hypothesis tests (t-test, Sharpe significance, bootstrap permutation, etc.)
- **Financial metrics** — additional performance measures (Calmar ratio, Omega ratio, tail risk metrics, drawdown analysis)

