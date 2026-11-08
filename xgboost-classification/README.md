# 📺 Streaming Subscription Renewal Predictor with XGBoost

A study project that uses **XGBoost** to predict whether a customer will **renew** or **cancel** their streaming subscription, based on how long they have been a customer, their monthly usage and the number of complaints.


## 🧠 What it does
- Generates a synthetic dataset of 300 customers with a hidden rule: customers renew if they use the service more than 15 hours/month **and** have fewer than 3 complaints
- Adds ~10% random label noise to simulate unpredictable real-life behavior
- Includes a feature (`months_as_customer`) that has **no effect** on the target, to check whether the model learns to ignore it
- Trains an `XGBClassifier` and evaluates it with stratified cross-validation and a held-out test set
- Compares the model against a "lazy" baseline that always predicts the most common class
- Shows a classification report, a confusion matrix and the feature importance
- Predicts the renewal probability for a new customer


## 📊 Results
| Metric | Value |
|--------|-------|
| Cross-validation accuracy | 85.42% (± 3.49%) |
| Test accuracy (XGBoost) | 86.67% |
| Test accuracy (lazy baseline) | 70.00% |
| Best possible accuracy (noise ceiling) | ~90% |

## 💡 Key takeaways
- **Accuracy alone can be misleading:** since only ~30% of customers renew, a model that always says "cancels" already gets 70%. The real gain of XGBoost is the jump from 70% to ~87%.
- **The model is close to the ceiling:** with 10% of labels flipped at random, no model can go much beyond ~90%.
- **The model found the right features:** `complaints` and `monthly_usage_hours` dominate the importance, while `months_as_customer` (which plays no role in the rule) gets the lowest score.

## ⚠️ Limitations
The data is synthetic and follows a simple rule, and the test set is small (60 customers), so results can vary with a different random seed.

## 🚀 Next steps
- Tune the hyperparameters with `GridSearchCV` or `RandomizedSearchCV`
- Try a real churn dataset (e.g. Telco Customer Churn)
- Explain individual predictions with SHAP values
