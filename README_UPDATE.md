# Update 29th August 2026
## 1. New function ```save_alpha()```
- Now after developing a new alpha, you can save its position by ```.save_alpha()```
- This function only save the alpha as ```csv``` file with date range equal to the range you used while training
- This function mainly serves as input for checking alpha correlation
- In the future, VNMiniQuant will support saving "alpha logic" instead so you can reload while training as well (for alpha combination purpose, for example)

## 2. New function ```check_alpha_corr()```
- This function is integrated into ```alpha.backtest()``` already
- Now instead of receiving only alpha PnL and Robustness test, you will get a table showing your current alpha with all of yours developed previously.
- In the first time, this will return "No data for corr check"
- So please use ```save_alpha()``` to initialize the target folder and save the alpha. Then after rerun the backtest, this corr tets will work.