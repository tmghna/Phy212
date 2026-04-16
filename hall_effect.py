import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "data/Hall_effect.ods"
df = pd.read_excel(file_path, engine="odf")
# Convert everything to numeric (force errors → NaN)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN
df = df.dropna()

# ---- CHANGE THESE NAMES IF NEEDED ----
I1 = df["I1 (mA)"] * 1e-3
V1 = df["V1 (mV)"] * 1e-3

I2 = df["I2 (mA)"] * 1e-3
V2 = df["V2 (mV)"] * 1e-3

def linear_fit_with_uncertainty(x, y):
    n = len(x)

    # Fit
    coeffs = np.polyfit(x, y, 1)
    m, c = coeffs

    # Predicted values
    y_fit = m*x + c

    # Residuals
    residuals = y - y_fit
    s2 = np.sum(residuals**2) / (n - 2)

    # Variance calculations
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean)**2)

    delta_m = np.sqrt(s2 / Sxx)
    delta_c = np.sqrt(s2 * (1/n + x_mean**2 / Sxx))

    return m, c, delta_m, delta_c


m1, c1, dm1, dc1 = linear_fit_with_uncertainty(I1, V1)
m2, c2, dm2, dc2 = linear_fit_with_uncertainty(I2, V2)

# PLOTTING

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def plot_case(ax, x, y, m, c, dm, dc, title):

    ax.scatter(x, y, label="Data")

    x_fit = np.linspace(min(x), max(x), 100)

    # Main fit
    ax.plot(x_fit, m*x_fit + c, label=f"(m,c)=({m:.3e},{c:.3e})")

    # Variations
    ax.plot(x_fit, (m+dm)*x_fit + c, linestyle="--",
            label=f"(m+dm,c)=({(m+dm):.3e},{c:.3e})")

    ax.plot(x_fit, (m-dm)*x_fit + c, linestyle="--")

    ax.plot(x_fit, m*x_fit + (c+dc), linestyle=":",
            label=f"(m,c+dc)=({m:.3e},{(c+dc):.3e})")

    ax.plot(x_fit, m*x_fit + (c-dc), linestyle=":")

    ax.set_xlabel("Current (I)")
    ax.set_ylabel("Hall Voltage (V)")
    ax.set_title(title)
    ax.legend()

plot_case(axes[0], I1, V1, m1, c1, dm1, dc1, "B into sample")
plot_case(axes[1], I2, V2, m2, c2, dm2, dc2, "B out of sample")

plt.tight_layout()
plt.show()


print("=== B INTO SAMPLE ===")
print(f"slope = {m1} ± {dm1}")
print(f"intercept = {c1} ± {dc1}")

print("\n=== B OUT OF SAMPLE ===")
print(f"slope = {m2} ± {dm2}")
print(f"intercept = {c2} ± {dc2}")