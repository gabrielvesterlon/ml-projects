# 🦖 vs 🤖 Fraud Detector with SVM
 
A study project that uses a **Support Vector Machine (SVM)** to classify purchases as **legitimate (dinosaurs)** or **fraudulent (robots)**, based on the purchase amount and the time of day.

 
## 🧠 What it does
- Encodes the time of day cyclically (sine and cosine), so that 23h is close to 0h
- Scales the features with `StandardScaler`
- Trains a linear SVM inside a scikit-learn `Pipeline`
- Evaluates the model with stratified cross-validation
- Plots the decision boundary, the margins and the support vectors
## ▶️ How to run
From the root of the repository:
```bash
pip install -r requirements.txt
python svm/fraud_detector_svm.py
```
 
## 📊 Example output
```
📈 Cross-validation accuracy: 100.00% (± 0.00%)
🎉 The SVM Robot Judge was trained successfully!
Number of support vectors: 3
 
--- Transaction Alert ---
Purchase attempt: R$ 2,500.00 at 23:30
Distance to the boundary (score): +1.03
🚨 HACKER DETECTED (ROBOT)! Card blocked for security!
```
 
## ⚠️ Limitations
The dataset is tiny (16 samples), hand-crafted and perfectly separable, so the 100% accuracy does not reflect a real-world scenario.
