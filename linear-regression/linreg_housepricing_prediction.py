"""
House Price Predictor with Linear Regression
--------------------------------------------
Predicts the price of a house based on its area, number of bedrooms and age.
Shows how Linear Regression can rediscover the hidden pricing rule
and predict prices outside the training range (extrapolation).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# ==========================================
# STEP 1: CREATE SYNTHETIC DATA
# 300 houses with different areas, number of bedrooms and ages
# ==========================================
rng = np.random.default_rng(42)  # random generator with a fixed seed (reproducible)
n_houses = 300

area_m2 = rng.integers(40, 251, n_houses)   # 40 to 250 m²
bedrooms = rng.integers(1, 6, n_houses)     # 1 to 5 bedrooms
house_age = rng.integers(0, 31, n_houses)   # 0 to 30 years

# ==========================================
# STEP 2: CREATE THE TARGET (hidden pricing rule)
# ==========================================
TRUE_INTERCEPT = 100_000
TRUE_COEFS = {"area_m2": 3_500, "bedrooms": 30_000, "house_age": -4_000}

base_price = (
    TRUE_INTERCEPT
    + area_m2 * TRUE_COEFS["area_m2"]       # each m² adds R$ 3,500
    + bedrooms * TRUE_COEFS["bedrooms"]     # each bedroom adds R$ 30,000
    + house_age * TRUE_COEFS["house_age"]   # each year of age removes R$ 4,000
)

# ==========================================
# STEP 3: ADD RANDOM NOISE
# Simulates things the data doesn't capture (location, finishings, negotiation...)
# ==========================================
NOISE_STD = 30_000
price = base_price + rng.normal(loc=0, scale=NOISE_STD, size=n_houses)

# ==========================================
# STEP 4: BUILD THE DATAFRAME AND SPLIT THE DATA
# ==========================================
features = list(TRUE_COEFS)
data = pd.DataFrame({
    "area_m2": area_m2,
    "bedrooms": bedrooms,
    "house_age": house_age,
    "price": price.round(2),
})
print(data.head(), "\n")

X = data[features]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# STEP 5: TRAIN THE LINEAR REGRESSION
# It looks for the straight-line formula that best fits the data:
# price = intercept + coef1 × area + coef2 × bedrooms + coef3 × age
# ==========================================
model = LinearRegression()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ==========================================
# STEP 6: READ THE FORMULA THE MODEL LEARNED
# ==========================================
def format_currency(value):
    sign = "-" if value < 0 else ""
    return f"{sign}R$ {abs(value):,.2f}"

print("--- Formula learned by the model (true rule in brackets) ---")
print(f"{'intercept':>10}: {format_currency(model.intercept_):>15}   [{format_currency(TRUE_INTERCEPT)}]")
for feature, coef in zip(features, model.coef_):
    print(f"{feature:>10}: {format_currency(coef):>15}   [{format_currency(TRUE_COEFS[feature])}]")

equation = f"price = {model.intercept_:,.0f}" + "".join(
    f" {'+' if c >= 0 else '-'} {abs(c):,.0f} × {f}" for f, c in zip(features, model.coef_)
)
print(f"\n📐 {equation}")

# ==========================================
# STEP 7: EVALUATE
# R²   -> how much of the price variation the model explains (1.0 = perfect)
# MAE  -> average error in R$, every error counts proportionally
# RMSE -> also in R$, but squares the errors first, so big mistakes weigh more
# ==========================================
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
baseline_mae = mean_absolute_error(y_test, baseline.predict(X_test))

print("\n--- Evaluation ---")
print(f"📈 Cross-validation R²: {cv_r2.mean():.3f} (± {cv_r2.std():.3f})")
print(f"🎯 Test R²: {r2_score(y_test, predictions):.3f}")
print(f"MAE:  {format_currency(mean_absolute_error(y_test, predictions))}")
print(f"RMSE: {format_currency(root_mean_squared_error(y_test, predictions))}")
print(f"😴 MAE of the lazy baseline (always the mean): {format_currency(baseline_mae)}")
print(f"🧱 Best possible RMSE (because of the noise): ~{format_currency(NOISE_STD)}")

# ==========================================
# STEP 8: PREDICT NEW HOUSES (inside and outside the training range)
# ==========================================
# XGBoost is trained only to compare how each model handles unseen ranges
xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

new_houses = pd.DataFrame(
    [[120, 3, 8],    # normal house, inside the training range
     [400, 4, 5]],   # huge house, OUTSIDE the training range (max was 250 m²)
    columns=features,
)
true_prices = TRUE_INTERCEPT + new_houses @ pd.Series(TRUE_COEFS)

print("\n--- New houses ---")
for i, house in new_houses.iterrows():
    print(f"🏠 {house['area_m2']} m², {house['bedrooms']} bedrooms, {house['house_age']} years old")
    print(f"   True price (hidden rule): {format_currency(true_prices[i])}")
    print(f"   Linear Regression:        {format_currency(model.predict(house.to_frame().T)[0])}")
    print(f"   XGBoost:                  {format_currency(xgb.predict(house.to_frame().T)[0])}")

# ==========================================
# STEP 9: PLOTS
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
ax_pred, ax_res, ax_coef, ax_ext = axes.ravel()
k_formatter = lambda v, _: f"{v / 1000:,.0f}k"

# 1) Actual vs predicted: the closer to the dashed line, the better
ax_pred.scatter(y_test, predictions, alpha=0.7, color="#dd8452")
limits = [y_test.min() * 0.95, y_test.max() * 1.05]
ax_pred.plot(limits, limits, "k--", label="Perfect prediction")
ax_pred.set(xlim=limits, ylim=limits, title="Actual vs Predicted Price (test set)",
            xlabel="Actual price (R$)", ylabel="Predicted price (R$)")
ax_pred.xaxis.set_major_formatter(k_formatter)
ax_pred.yaxis.set_major_formatter(k_formatter)
ax_pred.legend()
ax_pred.grid(True, linestyle=":", alpha=0.6)

# 2) Residuals: errors should be random, with no pattern
residuals = y_test - predictions
ax_res.scatter(predictions, residuals, alpha=0.7, color="#dd8452")
ax_res.axhline(0, color="k", linestyle="--")
ax_res.set(title="Residuals (errors) vs Predicted Price",
           xlabel="Predicted price (R$)", ylabel="Error: actual − predicted (R$)")
ax_res.xaxis.set_major_formatter(k_formatter)
ax_res.yaxis.set_major_formatter(k_formatter)
ax_res.grid(True, linestyle=":", alpha=0.6)

# 3) Learned coefficients vs the true rule
positions = np.arange(len(features))
ax_coef.bar(positions - 0.2, list(TRUE_COEFS.values()), width=0.4, color="#8c8c8c", label="True rule")
ax_coef.bar(positions + 0.2, model.coef_, width=0.4, color="#dd8452", label="Learned by the model")
ax_coef.axhline(0, color="k", linewidth=0.8)
ax_coef.set_xticks(positions, ["R$ per m²", "R$ per bedroom", "R$ per year of age"])
ax_coef.set(title="Coefficients: Learned vs True", ylabel="Effect on price (R$)")
ax_coef.yaxis.set_major_formatter(k_formatter)
ax_coef.legend()
ax_coef.grid(True, axis="y", linestyle=":", alpha=0.6)

# 4) Extrapolation: price vs area for a 3-bedroom, 10-year-old house
areas = np.arange(20, 451)
grid = pd.DataFrame({"area_m2": areas, "bedrooms": 3, "house_age": 10})
ax_ext.axvspan(40, 250, color="#d8f3dc", alpha=0.6, label="Training range (40 to 250 m²)")
ax_ext.plot(areas, TRUE_INTERCEPT + grid @ pd.Series(TRUE_COEFS), "k--", label="True rule")
ax_ext.plot(areas, model.predict(grid), color="#dd8452", linewidth=2, label="Linear Regression")
ax_ext.plot(areas, xgb.predict(grid), color="#4c72b0", linewidth=2, label="XGBoost")
ax_ext.set(title="Extrapolation: 3 bedrooms, 10 years old",
           xlabel="Area (m²)", ylabel="Predicted price (R$)")
ax_ext.yaxis.set_major_formatter(k_formatter)
ax_ext.legend(loc="upper left")
ax_ext.grid(True, linestyle=":", alpha=0.6)

fig.suptitle("Linear Regression: House Price Prediction", fontsize=15)
fig.tight_layout()
fig.savefig("linear_regression_results.png", dpi=150)  # image ready for the README
print("\n📊 Plot saved as 'linear_regression_results.png'")
plt.show()
