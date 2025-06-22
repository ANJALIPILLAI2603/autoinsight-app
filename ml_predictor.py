from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import pandas as pd
def train_ml_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert all columns to numeric if possible
    X = pd.get_dummies(X)

    # Match X and y rows
    X, y = X.align(y, join='inner', axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if y.dtype == 'object' or len(y.unique()) <= 10:
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if isinstance(model, RandomForestClassifier):
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results = {
            "type": "classification",
            "accuracy": round(acc * 100, 2),
            "confusion_matrix": cm.tolist()
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        results = {
            "type": "regression",
            "mse": round(mse, 2)
        }

    return results, model, X.columns.tolist()
