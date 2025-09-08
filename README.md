# Telecom Customer Churn Prediction

This project demonstrates step-by-step **data preprocessing, model training, and evaluation** using Python and Scikit-Learn on the Telco Customer Churn dataset.

---

## 📌 Steps Performed

### 1. Load Dataset
```python
import pandas as pd
import numpy as np

file_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-apache-spark/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_url)
print(df.head())
```
### 2. Handle Missing Values & Clean Data
```python
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan).astype(float)
df.dropna(subset=['TotalCharges'], inplace=True)
df.drop('customerID', axis=1, inplace=True)
```
### 3. Encode Target Variable
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['Churn'])
X = df.drop('Churn', axis=1)
```
### 4. Preprocess Features (Scaling + One-Hot Encoding)
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)
```
### 5. Split Data into Train/Test
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 6. Train Logistic Regression Model
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear'))
])

lr_pipeline.fit(X_train, y_train)
```
### 7. Train Random Forest Model
```python
from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
```
### 8. Evaluate Models
```python
from sklearn.metrics import classification_report, accuracy_score

y_pred_lr = lr_pipeline.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

y_pred_rf = rf_pipeline.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
```
### 9. Feature Importance (Random Forest)
```python
import seaborn as sns
import matplotlib.pyplot as plt

feature_names = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()
importances = rf_pipeline.named_steps['classifier'].feature_importances_

feature_importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances_df.head(15))
plt.title('Top 15 Feature Importances from Random Forest')
plt.show()
```
### 10. Save Model
```python
import joblib

joblib.dump(rf_pipeline, "telecom_churn_model.pkl")
print("Model saved as telecom_churn_model.pkl")
```
---
### ✅ Tools Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn
- Joblib
