import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def monte_carlo_stock_simulation(
    initial_price,
    years,
    days,
    simulations,
    expected_return,
    volatility,
):
    dt = years / days

    prices = np.zeros((days + 1, simulations))
    prices[0] = initial_price

    for t in range(1, days + 1):
        random_shock = np.random.normal(0, 1, simulations)

        prices[t] = prices[t - 1] * np.exp(
            (expected_return - 0.5 * volatility**2) * dt
            + volatility * np.sqrt(dt) * random_shock
        )

    return prices


# Parameters
initial_stock_price = 100
years = 1
days = 252
simulations = 1000
expected_return = 0.10
volatility = 0.20

# Run simulation
stock_prices = monte_carlo_stock_simulation(
    initial_stock_price,
    years,
    days,
    simulations,
    expected_return,
    volatility,
)

# Plot sample paths
plt.figure(figsize=(10, 5))

for i in range(50):
    plt.plot(stock_prices[:, i])

plt.title("Monte Carlo Simulation of Stock Prices")
plt.xlabel("Trading Days")
plt.ylabel("Stock Price ($)")
plt.tight_layout()
plt.savefig("monte_carlo_stock_paths.png", dpi=200)
plt.close()

# Final price statistics
final_prices = stock_prices[-1]

print(f"Average final price: ${np.mean(final_prices):.2f}")
print(f"Median final price: ${np.median(final_prices):.2f}")
print(f"Best case: ${np.max(final_prices):.2f}")
print(f"Worst case: ${np.min(final_prices):.2f}")
