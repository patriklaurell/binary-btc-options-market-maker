# Avellaneda-Stoikov Market Making for Binary Options

## Overview

This implementation adapts the classic Avellaneda-Stoikov market making strategy for binary options on Polymarket. The strategy optimally sets bid and ask prices while managing inventory risk.

## Key Concepts

### The Avellaneda-Stoikov Model

The Avellaneda-Stoikov model addresses the fundamental challenge in market making: balancing profit-taking through spreads with inventory risk management. The model provides mathematically optimal bid/ask quotes based on:

1. **Reservation Price**: The price at which the market maker is indifferent between buying and selling
2. **Optimal Spread**: The spread that balances profit from trading with risk from holding inventory

### Mathematical Framework

#### Reservation Price
```
r = s - q × γ × σ² × (T - t)
```

Where:
- `r` = reservation price (indifference price)
- `s` = fair value (mid price)
- `q` = normalized inventory position (-1 to 1)
- `γ` = risk aversion parameter
- `σ²` = variance (for binary options: p(1-p))
- `T - t` = time to expiry

**Interpretation**:
- If you're long (`q > 0`), reservation price decreases to encourage selling
- If you're short (`q < 0`), reservation price increases to encourage buying
- This creates a natural mean-reversion force toward target inventory

#### Optimal Spread
```
δ = γ × σ² × (T - t) + (2/γ) × ln(1 + γ/k)
```

Where:
- `δ` = optimal half-spread
- `γ` = risk aversion parameter
- `σ²` = variance
- `T - t` = time to expiry
- `k` = order arrival rate

**Components**:
1. **Risk term** (`γ × σ² × (T - t)`): Compensates for inventory risk
2. **Arrival term** (`(2/γ) × ln(1 + γ/k)`): Accounts for adverse selection and trading intensity

#### Final Quotes
```
bid = r - δ/2
ask = r + δ/2
```

## Adaptation for Binary Options

Binary options have unique properties that require special handling:

### 1. Bounded Prices
- Binary options pay $0 or $1, so prices are probabilities in [0, 1]
- All quotes are clipped to valid range [0.001, 0.999]

### 2. Variance Calculation
For traditional assets: `σ² = volatility²`

For binary options: `σ² = p(1-p)` where `p` is fair value

This captures the payoff variance of a Bernoulli random variable.

### 3. Time Decay
Binary options have discrete expiry times (e.g., every 15 minutes). The model accounts for this in spread calculations - shorter time to expiry means less uncertainty and tighter spreads.

## Usage

### Basic Usage
```bash
# Run with market maker enabled
python polymarket_btc_binary_option_pricing.py --market-maker

# With custom parameters
python polymarket_btc_binary_option_pricing.py \
    --market-maker \
    --risk-aversion 0.2 \
    --order-arrival-rate 2.0 \
    --max-inventory 50.0
```

### Parameters

#### `--market-maker`
Enable Avellaneda-Stoikov market making strategy.

#### `--risk-aversion` (γ, default: 0.1)
Controls how aggressively the market maker manages inventory risk.
- **Lower values** (e.g., 0.05): Tighter spreads, more aggressive
- **Higher values** (e.g., 0.5): Wider spreads, more conservative
- Directly affects spread width and inventory adjustment intensity

#### `--order-arrival-rate` (k, default: 1.0)
Expected rate of order arrivals (orders per time unit).
- **Lower values** (e.g., 0.5): Assumes less liquidity, wider spreads
- **Higher values** (e.g., 5.0): Assumes more liquidity, tighter spreads
- Represents market thickness and adverse selection risk

#### `--max-inventory` (default: 100.0)
Maximum inventory size in contracts. Used to normalize inventory position.
- Sets the scale for position management
- Inventory is normalized to [-1, 1] range for calculations

## Example Output

```
Avellaneda-Stoikov Market Maker Quotes:

Inventory: 0.00 contracts (0.0% of max, target: 0.00)

Expiry Time       Fair Value      Bid         Ask         Spread      Res.Price
---------------   ------------    ----------  ----------  ----------  ----------
12:15(5m)         $0.5123         $0.4985     $0.5261     $0.0276     $0.5123
12:30(20m)        $0.5234         $0.4932     $0.5536     $0.0604     $0.5234
12:45(35m)        $0.5312         $0.4901     $0.5723     $0.0822     $0.5312
```

## Testing

Run the test suite to understand how the strategy responds to different conditions:

```bash
python test_avellaneda_stoikov.py
```

The test script demonstrates:
1. **Neutral inventory**: Symmetric quotes around fair value
2. **Long inventory**: Quotes skewed lower to encourage selling
3. **Short inventory**: Quotes skewed higher to encourage buying
4. **Volatility impact**: Higher volatility → wider spreads
5. **Time decay**: More time → wider spreads
6. **Risk aversion**: Higher risk aversion → wider spreads
7. **Order flow**: Higher arrival rates → tighter spreads
8. **Inventory dynamics**: How quotes adjust as position changes

## Parameter Tuning Guide

### Conservative Market Making
```bash
--risk-aversion 0.5 --order-arrival-rate 0.5 --max-inventory 20.0
```
- Wide spreads
- Strong inventory management
- Suitable for low-liquidity markets or risk-averse strategies

### Aggressive Market Making
```bash
--risk-aversion 0.05 --order-arrival-rate 5.0 --max-inventory 200.0
```
- Tight spreads
- Larger position tolerance
- Suitable for high-liquidity markets with mean-reverting prices

### Balanced (Default)
```bash
--risk-aversion 0.1 --order-arrival-rate 1.0 --max-inventory 100.0
```
- Moderate spreads
- Reasonable inventory management
- Good starting point for most markets

## Implementation Details

### Class: `AvellanedaStoikovMarketMaker`

#### Key Methods

**`calculate_quotes(fair_value, inventory, volatility, time_to_expiry)`**
- Computes optimal bid/ask quotes
- Returns dict with bid, ask, spread, reservation_price, etc.

**`update_inventory(trade_size, is_buy)`**
- Updates inventory after a trade
- Call this when orders are filled

**`get_inventory_metrics()`**
- Returns current inventory statistics
- Useful for monitoring position

### Integration with BitcoinPriceOracle

The market maker integrates seamlessly with the existing price oracle:

```python
oracle = BitcoinPriceOracle(
    enable_market_maker=True,
    risk_aversion=0.1,
    order_arrival_rate=1.0,
    max_inventory=100.0
)
oracle.run()
```

## Theoretical Background

The Avellaneda-Stoikov model is derived from stochastic optimal control theory. The market maker solves:

```
max E[X_T - α × q_T²]
```

Where:
- `X_T` = terminal wealth
- `q_T` = terminal inventory
- `α` = inventory penalty (related to risk aversion)

The model assumes:
1. Geometric Brownian motion for the underlying asset
2. Poisson arrival of market orders
3. Price impact from inventory
4. Exponential utility function (CARA)

### Key Insights

1. **Inventory management is automatic**: The reservation price naturally adjusts to push inventory toward target
2. **Risk-return tradeoff**: Tighter spreads mean more fills but less profit per trade
3. **Adverse selection**: The model accounts for informed traders through the order arrival rate
4. **Time consistency**: Strategy is dynamically optimal at each time step

## References

- Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book." Quantitative Finance, 8(3), 217-224.
- Guéant, O., Lehalle, C. A., & Fernandez-Tapia, J. (2013). "Dealing with the inventory risk: a solution to the market making problem." Mathematics and Financial Economics, 7(4), 477-507.

## Future Enhancements

Potential improvements to the implementation:

1. **Order execution simulation**: Model actual order fills based on Polymarket orderbook
2. **Multi-expiry optimization**: Simultaneously quote across multiple expiry times
3. **Adaptive parameters**: Dynamically adjust risk aversion based on market conditions
4. **PnL tracking**: Record trades and calculate realized profit/loss
5. **Polymarket API integration**: Automatically place and manage orders
6. **Backtesting framework**: Test strategy on historical data
