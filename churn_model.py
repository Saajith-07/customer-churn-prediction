import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("churn_data.csv")

# Encode categorical columns
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["ContractType"] = le.fit_transform(df["ContractType"])
df["Churn"] = le.fit_transform(df["Churn"])

# Features and target
X = df.drop(["CustomerID", "Churn"], axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)