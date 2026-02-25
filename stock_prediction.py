import yfinance as yf
import matplotlib.pyplot as plt

# Download stock data (example: Apple)
data = yf.download("AAPL", start="2022-01-01", end="2024-01-01")

# Plot closing price
plt.plot(data['Close'])
plt.title("Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()