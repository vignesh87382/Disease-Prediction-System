import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

np.random.seed(42)

data_size = 1000

fever = np.random.randint(0, 2, data_size)
cough = np.random.randint(0, 2, data_size)
fatigue = np.random.randint(0, 2, data_size)
headache = np.random.randint(0, 2, data_size)

disease = (
    (fever & cough) |
    (fatigue & headache)
).astype(int)

data = pd.DataFrame({
    "fever": fever,
    "cough": cough,
    "fatigue": fatigue,
    "headache": headache,
    "disease": disease
})

X = data.drop("disease", axis=1)
y = data["disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

pickle.dump(model, open("disease_model.pkl", "wb"))

print("Model saved successfully.")
