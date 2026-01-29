import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression

def get_superscript(n):
    normal_digits = "0123456789-"
    super_digits = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
    trans = str.maketrans(normal_digits, super_digits)
    return str(n).translate(trans)

existing_file = 'data/thermistor.xlsx'
require_cols = [0,2]
required_df = pd.read_excel(existing_file, usecols = require_cols)
X = 1.0 / (np.array([data for data in required_df['Temperature']])+273.15)*1e3
Y = np.log(np.array([data for data in required_df['Resistance']]))
# points = np.array([*zip(X,Y)])

fig = plt.figure(figsize=(6,4))
plt.title('10K Thermistor')
plt.xlabel(f'1/T (x 10{get_superscript(-3)})')
plt.ylabel('log(R)')
plt.scatter(X, Y, color = 'red', label = 'raw data')

x=X.reshape((-1,1))
y=Y
model = LinearRegression()
model.fit(x, y) # or model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print(f"Coefficient of determination (R^2): {r_sq}")
print(f"Intercept (b0): {model.intercept_}")
print(f"Slope (b1): {model.coef_}")
y_pred = model.predict(x)
print(f"Predicted responses:\n{y_pred}")

# You can also predict on new data
y_new_pred = model.predict(x)
plt.plot(x, y_new_pred, color='b', label=f'best fit slope = {model.coef_}')

plt.legend()
plt.show()