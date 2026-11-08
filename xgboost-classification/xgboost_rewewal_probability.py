"""
Streaming Subscription Renewal Predictor with XGBoost
-----------------------------------------------------
Predicts whether a customer will renew their streaming subscription
based on how long they have been a customer, monthly usage and complaints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# ==========================================
# STEP 1: CREATE SYNTHETIC DATA
# Each customer has: months as a customer, monthly usage hours and number of complaints
# ==========================================
rng = np.random.default_rng(42)  # random generator with a fixed seed (reproducible)
n_customers = 300

months_as_customer = rng.integers(1, 61, n_customers)   # 1 to 60 months
monthly_usage_hours = rng.integers(1, 51, n_customers)  # 1 to 50 hours
complaints = rng.integers(0, 8, n_customers)            # 0 to 7 complaints

# ==========================================
# STEP 2: CREATE THE TARGET
# The customer renews (1) if they use the service a lot AND complain little; otherwise 0
# Note: months_as_customer does NOT affect the rule, so the model should learn to ignore it
# ==========================================
will_renew = ((monthly_usage_hours > 15) & (complaints < 3)).astype(int)

# ==========================================
# STEP 3: ADD RANDOM NOISE
# Flip the answer for ~10% of customers, simulating unpredictable real-life behavior
# ==========================================
noise = rng.random(n_customers) < 0.10
will_renew[noise] = 1 - will_renew[noise]

# ==========================================
# STEP 4: BUILD THE DATAFRAME
# ==========================================
data = pd.DataFrame({
    "months_as_customer": months_as_customer,
    "monthly_usage_hours": monthly_usage_hours,
    "complaints": complaints,
    "will_renew": will_renew,
})

print(f"Customers who renew: {data['will_renew'].mean():.1%}")
print(f"Customers with flipped labels (noise): {noise.mean():.1%}")

# ==========================================
# STEP 5: SPLIT FEATURES AND TARGET
# ==========================================
features = ["months_as_customer", "monthly_usage_hours", "complaints"]
X = data[features]
y = data["will_renew"]

# ==========================================
# STEP 6: SPLIT TRAIN AND TEST SETS
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ==========================================
# STEP 7: CREATE THE MODEL
# ==========================================
model = XGBClassifier(
    n_estimators=100,      # number of trees, each one fixing the mistakes of the previous ones
    max_depth=3,           # max depth of each tree (how many questions in a row)
    learning_rate=0.1,     # how much each new tree is allowed to correct
    random_state=42,
    eval_metric="logloss",
)

# ==========================================
# STEP 8: CROSS-VALIDATION (on the training set)
# A more reliable estimate than a single 60-customer test set
# ==========================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
print(f"\n📈 Cross-validation accuracy: {cv_scores.mean():.2%} (± {cv_scores.std():.2%})")

# ==========================================
# STEP 9: TRAIN AND PREDICT
# ==========================================
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Baseline: a "lazy" model that always predicts the most common class
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))

print(f"🎯 Test accuracy (XGBoost): {accuracy:.2%}")
print(f"😴 Test accuracy (lazy baseline): {baseline_accuracy:.2%}")
print(f"🧱 Best possible accuracy (because of the noise): ~{1 - noise.mean():.0%}")

print("\n--- Detailed report ---")
print(classification_report(y_test, predictions, target_names=["Cancels", "Renews"]))

# ==========================================
# STEP 10: PREDICT A NEW CUSTOMER
# ==========================================
new_customer = pd.DataFrame(
    [[24, 30, 1]],  # 24 months, 30 hours/month, 1 complaint
    columns=features,
)
renewal_probability = model.predict_proba(new_customer)[0, 1]

print("--- New customer ---")
print(new_customer.to_string(index=False))
print(f"Renewal probability: {renewal_probability:.1%}")
if renewal_probability >= 0.5:
    print("✅ Likely to RENEW!")
else:
    print("⚠️ Likely to CANCEL! Time to send a special offer.")

# ==========================================
# STEP 11: PLOTS (confusion matrix + feature importance)
# ==========================================
fig, (ax_cm, ax_fi) = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=["Cancels", "Renews"],
    cmap="Blues", colorbar=False, ax=ax_cm,
)
ax_cm.set_title("Confusion Matrix (test set)")

importance = pd.Series(model.feature_importances_, index=features).sort_values()
ax_fi.barh(importance.index, importance.values, color="#4c72b0")
ax_fi.set_title("Feature Importance")
ax_fi.set_xlabel("Importance")
for i, value in enumerate(importance.values):
    ax_fi.text(value + 0.01, i, f"{value:.2f}", va="center")
ax_fi.set_xlim(0, importance.max() * 1.15)

fig.suptitle("XGBoost: Streaming Subscription Renewal", fontsize=14)
fig.tight_layout()
fig.savefig("xgboost_results.png", dpi=150)  # image ready for the README
print("\n📊 Plot saved as 'xgboost_results.png'")
plt.show()
