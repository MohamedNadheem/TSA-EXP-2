# TSA-EXP-2
# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
### Name : Mohamed Nadheem N
### Date: 26.08.2025
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load CSV dataset
df = pd.read_csv(r"C:\Users\admin\Downloads\asiacup.csv")

# Use only Year and Avg Bat Strike Rate
yearly_scores = df.groupby("Year")["Avg Bat Strike Rate"].mean().reset_index()
yearly_scores.columns = ["year", "avg_strike_rate"]   # rename for consistency

# X, y prepare
X = yearly_scores["year"].values.reshape(-1, 1)
y = yearly_scores["avg_strike_rate"].values
X_norm = X - X.min()   # normalize year (avoid large numbers in poly fit)

# --- Linear Regression ---
linear_model = LinearRegression()
linear_model.fit(X_norm, y)
y_pred_linear = linear_model.predict(X_norm)

# --- Polynomial Regression (Degree 3) ---
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_norm)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

X_range = np.linspace(X_norm.min(), X_norm.max(), 300).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_pred_poly_smooth = poly_model.predict(X_range_poly)

```

A - LINEAR TREND ESTIMATION
```
plt.figure(figsize=(10,5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred_linear, color="red", label="Linear Trend")
plt.title("Average Batting Strike Rate Trend by Year (Linear Regression)")
plt.xlabel("Year")
plt.ylabel("Average Batting Strike Rate")
plt.legend()
plt.grid(True)
plt.show()

```

B- POLYNOMIAL TREND ESTIMATION
```
plt.figure(figsize=(10,5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_range + X.min(), y_pred_poly_smooth, color="green", linestyle="--", label="Polynomial Trend (degree=3)")
plt.title("Average Batting Strike Rate Trend by Year (Polynomial Regression)")
plt.xlabel("Year")
plt.ylabel("Average Batting Strike Rate")
plt.legend()
plt.grid(True)
plt.show()

```

### OUTPUT
A - LINEAR TREND ESTIMATION

<img width="1163" height="587" alt="image" src="https://github.com/user-attachments/assets/362b1f71-4240-4c45-b647-d92accaa6db7" />


B- POLYNOMIAL TREND ESTIMATION

<img width="1190" height="577" alt="image" src="https://github.com/user-attachments/assets/b6c19a86-4219-48c2-ba93-63f11b1b8990" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
