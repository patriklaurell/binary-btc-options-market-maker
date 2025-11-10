#!/usr/bin/env python3
"""
Polymarket BTC Binary Option Pricing
Fetches Bitcoin price from Binance API every second
Estimates real-time volatility using EWMA
Calculates binary option fair prices
"""

import requests
import time
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import argparse
from typing import Dict, Tuple, Optional


class EWMAVolatilityEstimator:
    """
    Exponentially Weighted Moving Average (EWMA) volatility estimator
    More robust and commonly used in practice for real-time volatility estimation
    """
    def __init__(self, lambda_ewma=0.94, initial_vol=0.0001):
        """
        Args:
            lambda_ewma: Decay factor for EWMA (0.94 is RiskMetrics standard for daily,
                        we use higher value like 0.98-0.99 for high-frequency)
            initial_vol: Initial volatility estimate (per second)
        """
        self.lambda_ewma = lambda_ewma
        self.variance = initial_vol ** 2  # Track variance instead of volatility

    def update(self, log_return):
        """
        Update volatility estimate with new log return

        Args:
            log_return: Log return since last update

        Returns:
            Updated volatility (standard deviation per time step)
        """
        # EWMA variance update: σ²_t = λ * σ²_{t-1} + (1 - λ) * r²_t
        self.variance = self.lambda_ewma * self.variance + (1 - self.lambda_ewma) * (log_return ** 2)

        # Return standard deviation
        return np.sqrt(self.variance)


class AvellanedaStoikovMarketMaker:
    """
    Avellaneda-Stoikov market making strategy adapted for binary options.

    The strategy optimally sets bid and ask spreads based on:
    - Inventory risk (position deviation from target)
    - Market volatility
    - Time to expiry
    - Risk aversion parameter

    Key formulas:
    - Reservation price: r = s - q * gamma * sigma^2 * (T - t)
    - Optimal spread: delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
    - Bid/Ask quotes: bid = r - delta/2, ask = r + delta/2

    Where:
    - s = fair value (mid price)
    - q = inventory position (normalized, -1 to 1)
    - gamma = risk aversion parameter
    - sigma = volatility
    - T - t = time to expiry
    - k = order arrival rate parameter
    """

    def __init__(
        self,
        risk_aversion: float = 0.1,
        order_arrival_rate: float = 1.0,
        target_inventory: float = 0.0,
        max_inventory: float = 100.0,
        min_spread: float = 0.001,
        max_spread: float = 0.1
    ):
        """
        Initialize Avellaneda-Stoikov market maker.

        Args:
            risk_aversion: Risk aversion parameter gamma (higher = wider spreads)
            order_arrival_rate: Expected order arrival rate k (higher = tighter spreads)
            target_inventory: Target inventory level (usually 0 for market neutral)
            max_inventory: Maximum inventory size for normalization
            min_spread: Minimum allowed spread
            max_spread: Maximum allowed spread
        """
        self.gamma = risk_aversion
        self.k = order_arrival_rate
        self.target_inventory = target_inventory
        self.max_inventory = max_inventory
        self.min_spread = min_spread
        self.max_spread = max_spread

        # Track inventory
        self.inventory = 0.0  # Current position in contracts

    def calculate_reservation_price(
        self,
        fair_value: float,
        inventory: float,
        volatility: float,
        time_to_expiry: float
    ) -> float:
        """
        Calculate reservation price (indifference price) based on inventory.

        The reservation price is the price at which the market maker is indifferent
        between buying and selling, adjusted for inventory risk.

        Args:
            fair_value: Fair value of the binary option
            inventory: Current inventory position (number of contracts)
            volatility: Market volatility (annualized)
            time_to_expiry: Time to expiry in years

        Returns:
            Reservation price
        """
        # Normalize inventory to [-1, 1] range
        q = (inventory - self.target_inventory) / self.max_inventory
        q = np.clip(q, -1, 1)

        # Inventory adjustment term
        # For binary options, we use the fair value variance as risk measure
        # Variance of binary option ≈ p(1-p) where p is fair value
        option_variance = fair_value * (1 - fair_value)

        # Adjust reservation price based on inventory
        # If long (q > 0), lower reservation price to encourage selling
        # If short (q < 0), raise reservation price to encourage buying
        inventory_adjustment = q * self.gamma * option_variance * time_to_expiry

        reservation_price = fair_value - inventory_adjustment

        # Ensure reservation price stays in valid probability range [0, 1]
        return np.clip(reservation_price, 0.001, 0.999)

    def calculate_optimal_spread(
        self,
        fair_value: float,
        volatility: float,
        time_to_expiry: float
    ) -> float:
        """
        Calculate optimal bid-ask spread using Avellaneda-Stoikov formula.

        Args:
            fair_value: Fair value of the binary option
            volatility: Market volatility (annualized)
            time_to_expiry: Time to expiry in years

        Returns:
            Optimal half-spread
        """
        # For binary options, use fair value variance as risk measure
        option_variance = fair_value * (1 - fair_value)

        # Avellaneda-Stoikov optimal spread formula
        # delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
        risk_term = self.gamma * option_variance * time_to_expiry
        arrival_term = (2 / self.gamma) * np.log(1 + self.gamma / self.k)

        half_spread = risk_term + arrival_term

        # Apply spread limits
        half_spread = np.clip(half_spread, self.min_spread / 2, self.max_spread / 2)

        return half_spread

    def calculate_quotes(
        self,
        fair_value: float,
        inventory: float,
        volatility: float,
        time_to_expiry: float
    ) -> Dict[str, float]:
        """
        Calculate optimal bid and ask quotes.

        Args:
            fair_value: Fair value of the binary option
            inventory: Current inventory position
            volatility: Market volatility (annualized)
            time_to_expiry: Time to expiry in years

        Returns:
            Dictionary with bid, ask, mid, spread, and reservation_price
        """
        # Calculate reservation price
        reservation_price = self.calculate_reservation_price(
            fair_value, inventory, volatility, time_to_expiry
        )

        # Calculate optimal spread
        half_spread = self.calculate_optimal_spread(
            fair_value, volatility, time_to_expiry
        )

        # Set bid and ask around reservation price
        bid = reservation_price - half_spread
        ask = reservation_price + half_spread

        # Ensure valid probability ranges
        bid = np.clip(bid, 0.001, 0.999)
        ask = np.clip(ask, 0.001, 0.999)

        # Ensure bid < ask
        if bid >= ask:
            mid = (bid + ask) / 2
            bid = mid - 0.001
            ask = mid + 0.001

        return {
            'bid': bid,
            'ask': ask,
            'mid': (bid + ask) / 2,
            'spread': ask - bid,
            'reservation_price': reservation_price,
            'half_spread': half_spread,
            'fair_value': fair_value
        }

    def update_inventory(self, trade_size: float, is_buy: bool):
        """
        Update inventory after a trade.

        Args:
            trade_size: Size of the trade in contracts
            is_buy: True if buying, False if selling
        """
        if is_buy:
            self.inventory += trade_size
        else:
            self.inventory -= trade_size

    def get_inventory_metrics(self) -> Dict[str, float]:
        """
        Get current inventory metrics.

        Returns:
            Dictionary with inventory statistics
        """
        normalized_inventory = (self.inventory - self.target_inventory) / self.max_inventory
        normalized_inventory = np.clip(normalized_inventory, -1, 1)

        return {
            'inventory': self.inventory,
            'target_inventory': self.target_inventory,
            'normalized_inventory': normalized_inventory,
            'inventory_pct': normalized_inventory * 100,
            'distance_from_target': abs(self.inventory - self.target_inventory)
        }


class BitcoinPriceOracle:
    def __init__(
        self,
        initial_strike=None,
        enable_market_maker=False,
        risk_aversion=0.1,
        order_arrival_rate=1.0,
        max_inventory=100.0
    ):
        # Use EWMA with high lambda (0.98) for second-by-second updates
        self.volatility_estimator = EWMAVolatilityEstimator(lambda_ewma=0.98, initial_vol=0.0001)
        self.last_price = None
        self.volatility = 0.50  # Initial volatility estimate (50% annualized)
        self.strike_price = initial_strike  # Strike price for current quarter
        self.current_quarter = None  # Track current quarter hour period

        # Market maker
        self.enable_market_maker = enable_market_maker
        self.market_maker = None
        if enable_market_maker:
            self.market_maker = AvellanedaStoikovMarketMaker(
                risk_aversion=risk_aversion,
                order_arrival_rate=order_arrival_rate,
                target_inventory=0.0,
                max_inventory=max_inventory,
                min_spread=0.001,
                max_spread=0.1
            )

    def fetch_binance_price(self):
        """Fetch Bitcoin price from Binance"""
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            params = {'symbol': 'BTCUSDT'}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data['lastPrice'])
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Binance error: {e}")
            return None

    def fetch_kraken_price(self):
        """Fetch Bitcoin price from Kraken"""
        try:
            url = "https://api.kraken.com/0/public/Ticker"
            params = {'pair': 'XBTUSD'}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            # Kraken returns last trade price in result['XXBTZUSD']['c'][0]
            price = float(data['result']['XXBTZUSD']['c'][0])
            return price
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Kraken error: {e}")
            return None

    def fetch_coinbase_price(self):
        """Fetch Bitcoin price from Coinbase"""
        try:
            url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            # Coinbase returns last trade price in 'price' field
            price = float(data['price'])
            return price
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Coinbase error: {e}")
            return None

    def fetch_bitstamp_price(self):
        """Fetch Bitcoin price from Bitstamp"""
        try:
            url = "https://www.bitstamp.net/api/v2/ticker/btcusd"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            # Bitstamp returns last trade price in 'last' field
            price = float(data['last'])
            return price
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Bitstamp error: {e}")
            return None

    def fetch_okx_price(self):
        """Fetch Bitcoin price from OKX"""
        try:
            url = "https://www.okx.com/api/v5/market/ticker"
            params = {'instId': 'BTC-USDT'}
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            # OKX returns last price in data[0]['last']
            price = float(data['data'][0]['last'])
            return price
        except (requests.exceptions.RequestException, KeyError, ValueError, IndexError) as e:
            print(f"OKX error: {e}")
            return None

    def fetch_price(self):
        """
        Fetch Bitcoin prices from Binance, Kraken, Coinbase, Bitstamp, and OKX
        Returns average price across all available venues
        """
        # Fetch from all five exchanges
        binance_price = self.fetch_binance_price()
        kraken_price = self.fetch_kraken_price()
        coinbase_price = self.fetch_coinbase_price()
        bitstamp_price = self.fetch_bitstamp_price()
        okx_price = self.fetch_okx_price()

        # Collect valid prices
        prices = []
        venue_info = []

        if binance_price:
            prices.append(binance_price)
            venue_info.append(f"Binance: ${binance_price:,.2f}")
        if kraken_price:
            prices.append(kraken_price)
            venue_info.append(f"Kraken: ${kraken_price:,.2f}")
        if coinbase_price:
            prices.append(coinbase_price)
            venue_info.append(f"Coinbase: ${coinbase_price:,.2f}")
        if bitstamp_price:
            prices.append(bitstamp_price)
            venue_info.append(f"Bitstamp: ${bitstamp_price:,.2f}")
        if okx_price:
            prices.append(okx_price)
            venue_info.append(f"OKX: ${okx_price:,.2f}")

        # If we have at least one price, calculate average
        if prices:
            avg_price = sum(prices) / len(prices)

            # Calculate spread (max - min)
            spread = max(prices) - min(prices) if len(prices) > 1 else 0

            return {
                'price': avg_price,
                'timestamp': datetime.now(),
                'venues': len(prices),
                'venue_prices': {
                    'binance': binance_price,
                    'kraken': kraken_price,
                    'coinbase': coinbase_price,
                    'bitstamp': bitstamp_price,
                    'okx': okx_price
                },
                'spread': spread,
                'venue_info': venue_info
            }
        else:
            print("Error: Unable to fetch price from any exchange")
            return None

    def update_volatility(self, current_price):
        """
        Update volatility estimate using EWMA
        Returns: current volatility estimate (annualized)
        """
        if self.last_price is None:
            self.last_price = current_price
            return self.volatility

        # Calculate log return
        log_return = np.log(current_price / self.last_price)
        self.last_price = current_price

        # Update EWMA volatility estimator
        instantaneous_vol = self.volatility_estimator.update(log_return)

        # Annualize volatility (1 second -> 1 year)
        # sqrt(seconds_per_year) = sqrt(365.25 * 24 * 3600) ≈ 5615.6
        self.volatility = instantaneous_vol * np.sqrt(365.25 * 24 * 3600)

        return self.volatility

    def calculate_binary_option_price(self, strike_price, time_to_expiry_seconds, current_price=None):
        """
        Calculate fair price of a binary option (pays $1 if price > strike at expiry)

        Args:
            strike_price: Strike price of the option
            time_to_expiry_seconds: Time to expiry in seconds
            current_price: Current Bitcoin price (uses last known if None)

        Returns:
            Fair price of binary option (probability of finishing in the money)
        """
        if current_price is None:
            current_price = self.last_price

        if current_price is None:
            return None

        # Time to expiry in years
        T = time_to_expiry_seconds / (365.25 * 24 * 3600)

        # Parameters
        S = current_price
        K = strike_price
        sigma = self.volatility
        r = 0.0  # Risk-free rate (assume 0 for crypto)

        # Under risk-neutral measure, binary option price is N(d2)
        # where d2 = (ln(S/K) + (r - 0.5*sigma^2)*T) / (sigma * sqrt(T))

        if T <= 0:
            # Option has expired
            return 1.0 if S > K else 0.0

        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        # Probability of finishing above strike
        prob = norm.cdf(d2)

        return prob


    def fetch_polymarket_orderbook(self, token_id):
        """
        Fetch orderbook for a Polymarket market

        Args:
            token_id: Token ID of the market (e.g., "21742633143463906290569050155826241533067272736897614950488156847949938836455")

        Returns:
            Dictionary containing orderbook data:
            {
                'market': Market condition ID,
                'asset_id': Asset/token ID,
                'timestamp': Unix timestamp in milliseconds,
                'bids': List of [price, size] pairs,
                'asks': List of [price, size] pairs,
                'best_bid': Best bid price,
                'best_ask': Best ask price,
                'spread': Bid-ask spread,
                'mid_price': Mid-market price
            }
            Returns None on error
        """
        try:
            url = "https://clob.polymarket.com/book"
            params = {'token_id': token_id}

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            # Parse bids and asks
            bids = [[float(bid['price']), float(bid['size'])] for bid in data.get('bids', [])]
            asks = [[float(ask['price']), float(ask['size'])] for ask in data.get('asks', [])]

            # Calculate best prices and spread
            best_bid = bids[0][0] if bids else None
            best_ask = asks[0][0] if asks else None
            spread = (best_ask - best_bid) if (best_bid and best_ask) else None
            mid_price = ((best_bid + best_ask) / 2) if (best_bid and best_ask) else None

            return {
                'market': data.get('market'),
                'asset_id': data.get('asset_id'),
                'timestamp': data.get('timestamp'),
                'bids': bids,
                'asks': asks,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'mid_price': mid_price,
                'min_order_size': data.get('min_order_size'),
                'tick_size': data.get('tick_size')
            }

        except requests.exceptions.RequestException as e:
            print(f"Error fetching Polymarket orderbook: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            print(f"Error parsing Polymarket orderbook: {e}")
            return None

    def get_current_quarter(self, current_time):
        """
        Get the current quarter hour period start time

        Args:
            current_time: Current datetime

        Returns:
            Datetime of current quarter hour start (rounded down)
        """
        minutes = current_time.minute
        quarter_minute = (minutes // 15) * 15
        return current_time.replace(minute=quarter_minute, second=0, microsecond=0)

    def update_strike_price(self, current_time, current_price):
        """
        Update strike price if we've entered a new quarter hour

        Args:
            current_time: Current datetime
            current_price: Current Bitcoin price

        Returns:
            True if strike was updated, False otherwise
        """
        current_quarter = self.get_current_quarter(current_time)

        if self.current_quarter is None:
            # First run - set initial strike
            self.current_quarter = current_quarter
            if self.strike_price is None:
                self.strike_price = current_price
            return True

        if current_quarter > self.current_quarter:
            # Entered new quarter hour
            self.current_quarter = current_quarter
            self.strike_price = current_price
            return True

        return False

    def get_next_quarter_hours(self, current_time, num_quarters=3):
        """
        Get the next N quarter hour timestamps

        Args:
            current_time: Current datetime
            num_quarters: Number of quarter hours to return

        Returns:
            List of (datetime, seconds_until) tuples
        """
        quarters = []

        # Round up to next quarter hour
        minutes = current_time.minute
        seconds = current_time.second
        microseconds = current_time.microsecond

        # Calculate minutes until next quarter
        next_quarter_minute = ((minutes // 15) + 1) * 15

        if next_quarter_minute >= 60:
            # Next quarter is in the next hour
            next_quarter = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            # Next quarter is in current hour
            next_quarter = current_time.replace(minute=next_quarter_minute, second=0, microsecond=0)

        for i in range(num_quarters):
            quarter_time = next_quarter + timedelta(minutes=15 * i)
            seconds_until = (quarter_time - current_time).total_seconds()
            quarters.append((quarter_time, seconds_until))

        return quarters

    def run(self, interval=1, show_options=True):
        """
        Run the oracle continuously, fetching price at specified interval

        Args:
            interval: Seconds between price fetches (default: 1)
            show_options: Display binary option prices (default: True)
        """
        print("Polymarket BTC Binary Option Pricing")
        print("Real-time volatility estimation using EWMA")
        print("Fetching prices from Binance, Kraken, Coinbase, Bitstamp, and OKX")
        print(f"Polling interval: {interval} second(s)")
        print("=" * 80)

        try:
            while True:
                price_data = self.fetch_price()

                if price_data:
                    timestamp = price_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    price = price_data['price']
                    venues = price_data['venues']
                    spread = price_data['spread']
                    venue_info = price_data['venue_info']

                    # Update strike price if new quarter hour started
                    strike_updated = self.update_strike_price(price_data['timestamp'], price)

                    # Update volatility estimate
                    vol = self.update_volatility(price)

                    print(f"\n[{timestamp}]")
                    print(f"  Average Price: ${price:,.2f} (from {venues} venue{'s' if venues > 1 else ''})")
                    print(f"  Venues: {', '.join(venue_info)}")
                    if venues > 1:
                        print(f"  Cross-Exchange Spread: ${spread:.2f}")
                    print(f"  Volatility (EWMA): {vol*100:.2f}% annualized")

                    if self.strike_price is not None:
                        print(f"  Strike Price: ${self.strike_price:,.2f} (Quarter: {self.current_quarter.strftime('%H:%M')})")
                        if strike_updated:
                            print(f"  >>> NEW QUARTER STARTED - Strike updated! <<<")

                    # Show binary option prices
                    if show_options and self.last_price is not None and self.strike_price is not None:
                        print(f"\n  Binary Option Prices (pays $1 if BTC > ${self.strike_price:,.2f} at expiry):\n")

                        # Get next 3 quarter hour expiries
                        quarters = self.get_next_quarter_hours(price_data['timestamp'], num_quarters=3)

                        # Format header with quarter hour times
                        header = f"  {'Expiry Time':<15}"
                        separator = f"  {'-'*15}"

                        for quarter_time, seconds_until in quarters:
                            time_label = quarter_time.strftime("%H:%M")
                            mins_until = int(seconds_until / 60)
                            header += f"\t{time_label}({mins_until}m)  "
                            separator += f"\t{'-'*12}"

                        print(header)
                        print(separator)

                        # Calculate probability for each expiry
                        row = f"  {'Fair Price (Long)':<15}"
                        for quarter_time, seconds_until in quarters:
                            prob = self.calculate_binary_option_price(self.strike_price, seconds_until, price)
                            if prob is not None:
                                row += f"\t${prob:.4f}     "
                            else:
                                row += f" N/A         "
                        print(row)

                        row = f"  {'Fair Price (Short)':<15}"
                        for quarter_time, seconds_until in quarters:
                            prob = self.calculate_binary_option_price(self.strike_price, seconds_until, price)
                            if prob is not None:
                                row += f"\t${1-prob:.4f}     "
                            else:
                                row += f" N/A         "
                        print(row)

                        # Display Avellaneda-Stoikov market maker quotes if enabled
                        if self.enable_market_maker and self.market_maker is not None:
                            print(f"\n  Avellaneda-Stoikov Market Maker Quotes:\n")

                            # Get inventory metrics
                            inv_metrics = self.market_maker.get_inventory_metrics()
                            print(f"  Inventory: {inv_metrics['inventory']:.2f} contracts "
                                  f"({inv_metrics['inventory_pct']:.1f}% of max, "
                                  f"target: {inv_metrics['target_inventory']:.2f})")

                            # Calculate and display quotes for each expiry
                            print(f"\n  {'Expiry Time':<15}\t{'Fair Value':<12}\t{'Bid':<10}\t{'Ask':<10}\t{'Spread':<10}\t{'Res.Price':<10}")
                            print(f"  {'-'*15}\t{'-'*12}\t{'-'*10}\t{'-'*10}\t{'-'*10}\t{'-'*10}")

                            for quarter_time, seconds_until in quarters:
                                # Calculate fair value
                                fair_value = self.calculate_binary_option_price(
                                    self.strike_price, seconds_until, price
                                )

                                if fair_value is not None:
                                    # Calculate time to expiry in years
                                    time_to_expiry_years = seconds_until / (365.25 * 24 * 3600)

                                    # Get market maker quotes
                                    quotes = self.market_maker.calculate_quotes(
                                        fair_value=fair_value,
                                        inventory=self.market_maker.inventory,
                                        volatility=vol,
                                        time_to_expiry=time_to_expiry_years
                                    )

                                    time_label = quarter_time.strftime("%H:%M")
                                    mins_until = int(seconds_until / 60)

                                    print(f"  {time_label}({mins_until}m)     "
                                          f"\t${fair_value:.4f}    "
                                          f"\t${quotes['bid']:.4f}  "
                                          f"\t${quotes['ask']:.4f}  "
                                          f"\t${quotes['spread']:.4f}  "
                                          f"\t${quotes['reservation_price']:.4f}")

                    print("-" * 80)

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nOracle stopped by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Bitcoin Price Oracle with Volatility Estimation and Binary Option Pricing'
    )
    parser.add_argument(
        '--strike',
        type=float,
        help='Initial strike price for the first quarter session. If not provided, uses current price.',
        default=None
    )
    parser.add_argument(
        '--interval',
        type=float,
        help='Seconds between price fetches (default: 1)',
        default=1
    )
    parser.add_argument(
        '--market-maker',
        action='store_true',
        help='Enable Avellaneda-Stoikov market making strategy'
    )
    parser.add_argument(
        '--risk-aversion',
        type=float,
        help='Risk aversion parameter gamma for market maker (higher = wider spreads, default: 0.1)',
        default=0.1
    )
    parser.add_argument(
        '--order-arrival-rate',
        type=float,
        help='Expected order arrival rate k for market maker (higher = tighter spreads, default: 1.0)',
        default=1.0
    )
    parser.add_argument(
        '--max-inventory',
        type=float,
        help='Maximum inventory size in contracts (default: 100.0)',
        default=100.0
    )

    args = parser.parse_args()

    oracle = BitcoinPriceOracle(
        initial_strike=args.strike,
        enable_market_maker=args.market_maker,
        risk_aversion=args.risk_aversion,
        order_arrival_rate=args.order_arrival_rate,
        max_inventory=args.max_inventory
    )
    oracle.run(interval=args.interval)
