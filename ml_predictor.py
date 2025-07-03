import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

def train_ml_model(df, target_col):
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    raw_inputs = X.columns.tolist()
    cat_inputs = {col: X[col].unique().tolist() for col in categorical_features}

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    model_type = RandomForestClassifier() if y.dtype == "object" or y.nunique() <= 10 else RandomForestRegressor()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_type)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Metrics
    if isinstance(model_type, RandomForestClassifier):
        metrics = {
            "type": "classification",
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
    else:
        metrics = {
            "type": "regression",
            "mae": round(mean_absolute_error(y_test, y_pred), 2),
            "rmse": round(mean_squared_error(y_test, y_pred, squared=False), 2)
        }

    # Final encoded column names (for prediction UI input reindexing)
    encoded_num = numeric_features
    if categorical_features:
        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe.fit(X_train[categorical_features])
        encoded_cat = ohe.get_feature_names_out(categorical_features).tolist()
    else:
        encoded_cat = []

    final_features = encoded_num + encoded_cat

    return metrics, pipeline, final_features, cat_inputs, raw_inputs
