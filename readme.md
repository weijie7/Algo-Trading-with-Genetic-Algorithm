# Algo-Trading-with-Genetic-Algorithm
Algo trading with strategy customization, genetic algorithm for hyper params optimizing, and backtesting.


1. run prepare_dataset.py to collect data. Customize if any indicators are missing.
// TODO: include args for timeframe and coin pairs selection
2. Modify strategy in trade_operation.py under Strategy's sub-class
3. Define your own fitness function under backtesting function in trade_operation.py
4. Depending on your number of variables defined in your strategy, customize the inputs accordingly in your StrategyOptimizer class in GA_Optimizer.py. Variables that you can adjust are:
- number of generation
- generation size
- number of genes (this is your number of variables defined in your strategy)
- mutation probability (probability for algo to select an individual for mutation)
- gene self-mutation prob (paobability for mutating each gene in selected individual )
- n_select_best : select top n best individual
- an option to insert good gene into the population pool (from your previous iteration)
- other backtest var i.e. stop loss target, stake amount (will impact your fees & profit), initial wallet amount, and fees/commission


## Current Strategy variables
### Buy signal
Rule 1 : ema1 CO ema2 (2 variables for periods for both)
Rule 2: rsi1 CO rsi2 (2 variables for periods for both)
Rule 3: momentum_rsi (data) > momentum_rsi_threshold (1 variables)
Rule 4: Heikin Ashi trend (data) > Heikin Ashi trend trend (data rolling 2d) (0 variables)
Rule 5: Heikin Ashi trend (data) > Heikin Ashi trend trend (data rolling 3d) (0 variables)
Rule 6: Heikin Ashi trend (data) > Heikin Ashi trend trend (data rolling 4d) (0 variables)
Rule 7: fastk (data) > fastk_threshold (1 variable)
Rule 8: fastd (data) > fastd_threshold (1 variable)
Rule 9: fastk (data) CO fastd (data)
Rule 10: closed_price (data) > ema3 (1 variable for period)

- Total 8 variables for buy signal
- Another 10 variables as boolean mask to activate / deactivate each rules

### Sell signal
Inverse of buy, with another unique 8+10 variables


### Misc variable
- take profit

### Total variables
Buy signal - 8 + 10
Sell signal - 8 + 10
tp - 1
total - 37