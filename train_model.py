import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Sample house price dataset
data = {
    "area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    "bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "bathrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    "age": [20, 15, 10, 8, 5, 7, 4, 3, 2, 1],
    "price": [25, 30, 40, 48, 55, 62, 72, 82, 95, 110]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms", "bathrooms", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

joblib.dump(model, "house_price_model.pkl")
print("Saved model as house_price_model.pkl")
