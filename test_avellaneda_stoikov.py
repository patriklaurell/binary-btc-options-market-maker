#!/usr/bin/env python3
"""
Test script for Avellaneda-Stoikov market maker
Demonstrates how the strategy adjusts quotes based on inventory and market conditions
"""

import numpy as np
from polymarket_btc_binary_option_pricing import AvellanedaStoikovMarketMaker


def test_market_maker_basic():
    """Test basic market maker functionality"""
    print("=" * 80)
    print("Avellaneda-Stoikov Market Maker Test")
    print("=" * 80)

    # Initialize market maker
    mm = AvellanedaStoikovMarketMaker(
        risk_aversion=0.1,
        order_arrival_rate=1.0,
        target_inventory=0.0,
        max_inventory=100.0,
        min_spread=0.001,
        max_spread=0.1
    )

    # Test parameters
    fair_value = 0.5  # 50% probability
    volatility = 0.5  # 50% annualized volatility
    time_to_expiry = 15 * 60 / (365.25 * 24 * 3600)  # 15 minutes in years

    print("\nTest 1: Neutral Inventory (inventory = 0)")
    print("-" * 80)
    quotes = mm.calculate_quotes(fair_value, 0.0, volatility, time_to_expiry)
    print(f"Fair Value:        ${quotes['fair_value']:.4f}")
    print(f"Reservation Price: ${quotes['reservation_price']:.4f}")
    print(f"Bid:               ${quotes['bid']:.4f}")
    print(f"Ask:               ${quotes['ask']:.4f}")
    print(f"Spread:            ${quotes['spread']:.4f}")
    print(f"Half-spread:       ${quotes['half_spread']:.4f}")

    print("\n\nTest 2: Long Inventory (inventory = +50 contracts)")
    print("-" * 80)
    quotes = mm.calculate_quotes(fair_value, 50.0, volatility, time_to_expiry)
    print(f"Fair Value:        ${quotes['fair_value']:.4f}")
    print(f"Reservation Price: ${quotes['reservation_price']:.4f}")
    print(f"Bid:               ${quotes['bid']:.4f}")
    print(f"Ask:               ${quotes['ask']:.4f}")
    print(f"Spread:            ${quotes['spread']:.4f}")
    print("Note: Reservation price is LOWER to encourage selling")

    print("\n\nTest 3: Short Inventory (inventory = -50 contracts)")
    print("-" * 80)
    quotes = mm.calculate_quotes(fair_value, -50.0, volatility, time_to_expiry)
    print(f"Fair Value:        ${quotes['fair_value']:.4f}")
    print(f"Reservation Price: ${quotes['reservation_price']:.4f}")
    print(f"Bid:               ${quotes['bid']:.4f}")
    print(f"Ask:               ${quotes['ask']:.4f}")
    print(f"Spread:            ${quotes['spread']:.4f}")
    print("Note: Reservation price is HIGHER to encourage buying")

    print("\n\nTest 4: Impact of Volatility")
    print("-" * 80)
    for vol in [0.3, 0.5, 0.7]:
        quotes = mm.calculate_quotes(fair_value, 0.0, vol, time_to_expiry)
        print(f"Volatility: {vol*100:.0f}% | Spread: ${quotes['spread']:.4f}")
    print("Note: Higher volatility leads to WIDER spreads")

    print("\n\nTest 5: Impact of Time to Expiry")
    print("-" * 80)
    for mins in [5, 15, 30, 60]:
        tte = mins * 60 / (365.25 * 24 * 3600)
        quotes = mm.calculate_quotes(fair_value, 0.0, volatility, tte)
        print(f"Time to Expiry: {mins:2d}m | Spread: ${quotes['spread']:.4f}")
    print("Note: More time to expiry leads to WIDER spreads (more uncertainty)")

    print("\n\nTest 6: Impact of Risk Aversion")
    print("-" * 80)
    for gamma in [0.05, 0.1, 0.2, 0.5]:
        mm_temp = AvellanedaStoikovMarketMaker(
            risk_aversion=gamma,
            order_arrival_rate=1.0,
            max_inventory=100.0
        )
        quotes = mm_temp.calculate_quotes(fair_value, 0.0, volatility, time_to_expiry)
        print(f"Risk Aversion: {gamma:.2f} | Spread: ${quotes['spread']:.4f}")
    print("Note: Higher risk aversion leads to WIDER spreads")

    print("\n\nTest 7: Impact of Order Arrival Rate")
    print("-" * 80)
    for k in [0.5, 1.0, 2.0, 5.0]:
        mm_temp = AvellanedaStoikovMarketMaker(
            risk_aversion=0.1,
            order_arrival_rate=k,
            max_inventory=100.0
        )
        quotes = mm_temp.calculate_quotes(fair_value, 0.0, volatility, time_to_expiry)
        print(f"Order Arrival Rate: {k:.1f} | Spread: ${quotes['spread']:.4f}")
    print("Note: Higher order arrival rate leads to TIGHTER spreads (more liquidity)")

    print("\n\nTest 8: Inventory Dynamics")
    print("-" * 80)
    mm_inv = AvellanedaStoikovMarketMaker(
        risk_aversion=0.1,
        order_arrival_rate=1.0,
        target_inventory=0.0,
        max_inventory=100.0
    )

    print(f"Initial inventory: {mm_inv.inventory:.0f}")
    quotes = mm_inv.calculate_quotes(fair_value, mm_inv.inventory, volatility, time_to_expiry)
    print(f"Reservation price: ${quotes['reservation_price']:.4f}")

    # Simulate buying 30 contracts
    mm_inv.update_inventory(30, is_buy=True)
    print(f"\nAfter buying 30 contracts: {mm_inv.inventory:.0f}")
    quotes = mm_inv.calculate_quotes(fair_value, mm_inv.inventory, volatility, time_to_expiry)
    print(f"Reservation price: ${quotes['reservation_price']:.4f} (lower to encourage selling)")

    # Simulate selling 50 contracts
    mm_inv.update_inventory(50, is_buy=False)
    print(f"\nAfter selling 50 contracts: {mm_inv.inventory:.0f}")
    quotes = mm_inv.calculate_quotes(fair_value, mm_inv.inventory, volatility, time_to_expiry)
    print(f"Reservation price: ${quotes['reservation_price']:.4f} (higher to encourage buying)")

    # Get inventory metrics
    metrics = mm_inv.get_inventory_metrics()
    print(f"\nInventory metrics:")
    print(f"  Current: {metrics['inventory']:.2f}")
    print(f"  Target: {metrics['target_inventory']:.2f}")
    print(f"  Normalized: {metrics['normalized_inventory']:.3f}")
    print(f"  Percentage: {metrics['inventory_pct']:.1f}%")
    print(f"  Distance from target: {metrics['distance_from_target']:.2f}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_market_maker_basic()
