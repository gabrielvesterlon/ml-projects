"""
Fraud Detector with SVM: Dinosaurs vs Robots
--------------------------------------------
Classifies purchases as legitimate (dinosaurs) or fraudulent (robots)
based on the purchase amount and the time of day.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ==========================================
# STEP 1: CREATE THE DATA (Dinosaurs vs Robots)
# Columns: [Purchase amount (BRL), Time of day (0h to 24h)]
# ==========================================

# 🟢 Legitimate purchases (Dinosaurs): lower amounts, business hours
dinosaur_data = np.array([
    [30, 9], [120, 12], [15, 14], [200, 15], [45, 10], [300, 11], [60, 16], [110, 13]
], dtype=float)

# 🤖 Fraudulent purchases (Robots): high amounts, late at night
robot_data = np.array([
    [3500, 4], [8000, 1], [4200, 3], [6000, 0], [4000, 5], [7000, 2], [3100, 4], [5500, 1]
], dtype=float)

X = np.vstack((dinosaur_data, robot_data))
y = np.concatenate((
    np.zeros(len(dinosaur_data), dtype=int),  # 0 = Dinosaur
    np.ones(len(robot_data), dtype=int),      # 1 = Robot
))

# ==========================================
# STEP 2: PREPROCESSING + MODEL
# ==========================================
def encode_cyclic_hour(X):
    """Replace the hour with (sine, cosine) so that 23h ends up close to 0h."""
    amount = X[:, 0]
    angle = 2 * np.pi * X[:, 1] / 24
    return np.column_stack((amount, np.sin(angle), np.cos(angle)))

svm_model = make_pipeline(
    FunctionTransformer(encode_cyclic_hour),
    StandardScaler(),               # puts all features on the same scale
    SVC(kernel="linear", C=1.0),    # C controls how strict the judge is
)

# Cross-validation: an honest estimate of performance (tiny dataset!)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
scores = cross_val_score(svm_model, X, y, cv=cv)
print(f"📈 Cross-validation accuracy: {scores.mean():.2%} (± {scores.std():.2%})")

svm_model.fit(X, y)
svc = svm_model.named_steps["svc"]
print("🎉 The SVM Robot Judge was trained successfully!")
print(f"Number of support vectors: {len(svc.support_vectors_)}")

# ==========================================
# STEP 3: REAL-TIME TEST (New suspicious purchase)
# ==========================================
def format_currency(amount):
    return f"R$ {amount:,.2f}"

def format_time(hour):
    h, m = int(hour), int(round((hour % 1) * 60))
    return f"{h:02d}:{m:02d}"

new_purchase = np.array([[2500, 23.5]])  # R$ 2,500 at 23:30
prediction = svm_model.predict(new_purchase)[0]
score = svm_model.decision_function(new_purchase)[0]

print("\n--- Transaction Alert ---")
print(f"Purchase attempt: {format_currency(new_purchase[0, 0])} at {format_time(new_purchase[0, 1])}")
print(f"Distance to the boundary (score): {score:+.2f}")

if prediction == 1:
    print("🚨 HACKER DETECTED (ROBOT)! Card blocked for security!")
else:
    print("🟢 PURCHASE APPROVED (DINOSAUR)! Have a nice day!")

# ==========================================
# STEP 4: PLOT THE DECISION BOUNDARY
# ==========================================
xx, yy = np.meshgrid(np.linspace(-500, 9000, 400), np.linspace(0, 24, 400))
grid = np.column_stack((xx.ravel(), yy.ravel()))
Z = svm_model.decision_function(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 6))

# Colored regions and lines: boundary (0) and margins (-1 and +1)
ax.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=["#d8f3dc", "#dbe7ff"], alpha=0.6)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=[":", "--", ":"])

ax.scatter(dinosaur_data[:, 0], dinosaur_data[:, 1], c="green", s=100, label="Dinosaurs (legitimate)")
ax.scatter(robot_data[:, 0], robot_data[:, 1], c="blue", s=100, label="Robots (fraud)")
ax.scatter(*new_purchase[0], c="red", marker="x", s=200, linewidths=3, label="New suspicious purchase")

# Highlight the support vectors (the points that define the margin)
sv = X[svc.support_]
ax.scatter(sv[:, 0], sv[:, 1], s=250, facecolors="none", edgecolors="k", linewidths=1.5,
           label="Support vectors")

ax.plot([], [], "k--", label="Decision boundary")
ax.plot([], [], "k:", label="Margins")

ax.set_title("SVM Fraud Detector: Dinosaurs vs Robots")
ax.set_xlabel("Purchase amount (BRL)")
ax.set_ylabel("Time of day (hours)")
ax.set_xlim(-500, 9000)
ax.set_ylim(0, 24)
ax.legend(loc="center right")
ax.grid(True, linestyle=":", alpha=0.6)

fig.tight_layout()
fig.savefig("svm_decision_boundary.png", dpi=150)  # image ready for the README
print("\n📊 Plot saved as 'svm_decision_boundary.png'")
plt.show()
