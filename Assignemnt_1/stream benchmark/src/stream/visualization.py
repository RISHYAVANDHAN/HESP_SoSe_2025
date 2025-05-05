import pandas as pd
import matplotlib.pyplot as plt

# Load results
data = pd.read_csv("bandwidth_results.csv")

# Plot bandwidth vs. buffer size
for version in data["Version"].unique():
    subset = data[data["Version"] == version]
    plt.plot(subset["BufferSize"], subset["Bandwidth(GB/s)"], label=version)

plt.xscale("log")
plt.xlabel("Buffer Size")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Sustained CPU Bandwidth")
plt.legend()
plt.grid()
plt.savefig("bandwidth_plot.png")
plt.show()