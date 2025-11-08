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


class BitcoinPriceOracle:
    def __init__(self, api_url="https://api.binance.com/api/v3/ticker/24hr", initial_strike=None):
        self.api_url = api_url
        self.params = {
            'symbol': 'BTCUSDT'
        }
        # Use EWMA with high lambda (0.98) for second-by-second updates
        self.volatility_estimator = EWMAVolatilityEstimator(lambda_ewma=0.98, initial_vol=0.0001)
        self.last_price = None
        self.volatility = 0.50  # Initial volatility estimate (50% annualized)
        self.strike_price = initial_strike  # Strike price for current quarter
        self.current_quarter = None  # Track current quarter hour period

    def fetch_price(self):
        """Fetch current Bitcoin price from Binance API"""
        try:
            response = requests.get(self.api_url, params=self.params, timeout=5)
            response.raise_for_status()
            data = response.json()

            price = float(data['lastPrice'])
            change_24h = float(data['priceChangePercent'])

            return {
                'price': price,
                'change_24h': change_24h,
                'timestamp': datetime.now()
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching price: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Error parsing response: {e}")
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
        print(f"Fetching price every {interval} second(s)")
        print("=" * 80)

        try:
            while True:
                price_data = self.fetch_price()

                if price_data:
                    timestamp = price_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    price = price_data['price']
                    change = price_data['change_24h']

                    # Update strike price if new quarter hour started
                    strike_updated = self.update_strike_price(price_data['timestamp'], price)

                    # Update volatility estimate
                    vol = self.update_volatility(price)

                    change_symbol = "+" if change >= 0 else ""
                    print(f"\n[{timestamp}]")
                    print(f"  Price: ${price:,.2f} ({change_symbol}{change:.2f}% 24h)")
                    print(f"  Volatility (EWMA): {vol*100:.2f}% annualized")

                    if self.strike_price is not None:
                        print(f"  Strike Price: ${self.strike_price:,.2f} (Quarter: {self.current_quarter.strftime('%H:%M')})")
                        if strike_updated:
                            print(f"  >>> NEW QUARTER STARTED - Strike updated! <<<")

                    # Show binary option prices
                    if show_options and self.last_price is not None and self.strike_price is not None:
                        print(f"\n  Binary Option Prices (pays $1 if BTC > ${self.strike_price:,.2f} at expiry):")

                        # Get next 3 quarter hour expiries
                        quarters = self.get_next_quarter_hours(price_data['timestamp'], num_quarters=3)

                        # Format header with quarter hour times
                        header = f"  {'Expiry Time':<15}"
                        separator = f"  {'-'*15}"

                        for quarter_time, seconds_until in quarters:
                            time_label = quarter_time.strftime("%H:%M")
                            mins_until = int(seconds_until / 60)
                            header += f" {time_label}({mins_until}m)  "
                            separator += f" {'-'*12}"

                        print(header)
                        print(separator)

                        # Calculate probability for each expiry
                        row = f"  {'Fair Price':<15}"
                        for quarter_time, seconds_until in quarters:
                            prob = self.calculate_binary_option_price(self.strike_price, seconds_until, price)
                            if prob is not None:
                                row += f" ${prob:.4f}     "
                            else:
                                row += f" N/A         "

                        print(row)

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

    args = parser.parse_args()

    oracle = BitcoinPriceOracle(initial_strike=args.strike)
    oracle.run(interval=args.interval)
