import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def abrupts(data: np.array, start: int = 1):
    peaks,troughs = [], []
    for i in range(1,len(data)):
        if data[i,1]<data[i-1,1] and start==1:
            peaks.append(data[i-1])
            start = -start
        elif data[i,1]>data[i-1,1] and start==-1:
            troughs.append(data[i-1])
            start = -start
    return np.array(peaks), np.array(troughs)


existing_file = 'frank-hertz.xlsx'
require_cols = [0,2]
required_df = pd.read_excel(existing_file, usecols = require_cols)
X = np.array([data for data in required_df['Voltage']])
Y = np.array([data for data in required_df['Current']])*1e-2
points = np.array([*zip(X,Y)])

peaks, troughs = abrupts(points)
troughs = np.append(troughs, [points[-1]], axis=0)

peak_ranges = []
for i in range(len(peaks)-1):
    mask = (points[:, 0] >= peaks[i, 0]) & (points[:, 0] <= peaks[i+1, 0])
    peak_ranges.append(points[mask])

trough_ranges = []
for i in range(len(troughs)-1):
    mask = (points[:, 0] >= troughs[i, 0]) & (points[:, 0] <= troughs[i+1, 0])
    trough_ranges.append(points[mask])


fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_figheight(5)
fig.set_figwidth(10)

plt.suptitle("Voltage vs Current")

Troughs = np.array([])
for i in range(len(peak_ranges)):
    _X, _Y = zip(*peak_ranges[i])
    coefficients = np.polyfit(_X, _Y, 4)
    p = np.poly1d(coefficients)
    extended_X = np.linspace(_X[0], _X[-1], 100)
    extended_Y = p(extended_X)
    Troughs = np.append(Troughs, min(extended_Y))
    ax[0].plot(extended_X, extended_Y, color = 'b', label=f'Trough Fit {i+1}')

Peaks = np.array([])
for i in range(len(trough_ranges)):
    _X, _Y = zip(*trough_ranges[i])
    coefficients = np.polyfit(_X, _Y, 4)
    p = np.poly1d(coefficients)
    extended_X = np.linspace(_X[0], _X[-1], 100)
    extended_Y = p(extended_X)
    Peaks = np.append(Peaks, min(extended_Y))
    ax[1].plot(extended_X, extended_Y, color = 'r', ls='--', label=f'Peak Fit {i+1}')

ax[0].set_xlabel('Voltage(V)')
ax[0].set_ylabel('Current(A)')
ax[0].scatter(X, Y, label='Original Data', color='g')
for trough in Troughs:
    ax[0].axvline(trough, ls = '--')
ax[0].axvline()
ax[0].legend()
ax[1].set_xlabel('Voltage(V)')
ax[1].set_ylabel('Current(A)')
ax[1].scatter(X, Y, label='Original Data', color='g')
ax[1].legend()

plt.tight_layout()
plt.show()
