import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Load dataset
df = pd.read_csv("House Price Prediction Dataset.csv")

print("\n==============================")
print("HOUSE PRICE PREDICTION")
print("==============================")

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# Remove duplicate rows
df = df.drop_duplicates()

# Remove ID column
if "Id" in df.columns:
    df = df.drop(columns=["Id"])


# ==============================
# EDA
# ==============================

print("\nStatistical Summary:")
print(df.describe())

print("\nColumns:")
print(df.columns.tolist())


# Price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["Price"], kde=True)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.savefig("outputs/price_distribution.png")
plt.close()


# Area vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Area", y="Price")
plt.title("Area vs House Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("outputs/area_vs_price.png")
plt.close()


# Location vs Price
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Location", y="Price")
plt.title("Location vs House Price")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/location_vs_price.png")
plt.close()


# Condition vs Price
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Condition", y="Price")
plt.title("Condition vs House Price")
plt.tight_layout()
plt.savefig("outputs/condition_vs_price.png")
plt.close()


# Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()


# ==============================
# FEATURES AND TARGET
# ==============================

X = df.drop(columns=["Price"])
y = df["Price"]


# Numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ==============================
# PREPROCESSING
# ==============================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==============================
# ML MODELS
# ==============================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        )
}


# ==============================
# TRAIN MODELS
# ==============================

results = []
trained_models = {}

print("\n==============================")
print("MODEL TRAINING")
print("==============================")


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print("\n", name)
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))


# ==============================
# MODEL COMPARISON
# ==============================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(results_df.to_string(index=False))


# Save results
results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)


# Model comparison graph
plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="R2 Score",
    y="Model"
)

plt.title("Model Performance Comparison")
plt.xlabel("R2 Score")
plt.ylabel("Model")
plt.xlim(0, 1)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png"
)

plt.close()


# ==============================
# MODEL IMPROVEMENT
# ==============================

print("\n==============================")
print("MODEL IMPROVEMENT")
print("==============================")


rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


param_grid = {

    "model__n_estimators": [100, 200],

    "model__max_depth": [
        None,
        10,
        20
    ],

    "model__min_samples_split": [
        2,
        5
    ]
}


grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)


grid_search.fit(
    X_train,
    y_train
)


tuned_model = grid_search.best_estimator_

tuned_predictions = tuned_model.predict(
    X_test
)


tuned_mae = mean_absolute_error(
    y_test,
    tuned_predictions
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tuned_predictions
    )
)

tuned_r2 = r2_score(
    y_test,
    tuned_predictions
)


print("\nTuned Random Forest")

print("Best Parameters:")
print(grid_search.best_params_)

print("MAE:", round(tuned_mae, 2))
print("RMSE:", round(tuned_rmse, 2))
print("R2 Score:", round(tuned_r2, 4))


# ==============================
# SELECT FINAL MODEL
# ==============================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_predictions = best_model.predict(
    X_test
)

best_r2 = r2_score(
    y_test,
    best_predictions
)


if tuned_r2 >= best_r2:

    final_model = tuned_model
    final_model_name = "Tuned Random Forest"
    final_mae = tuned_mae
    final_rmse = tuned_rmse
    final_r2 = tuned_r2

else:

    final_model = best_model
    final_model_name = best_model_name

    final_mae = mean_absolute_error(
        y_test,
        best_predictions
    )

    final_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            best_predictions
        )
    )

    final_r2 = best_r2


# ==============================
# SAVE FINAL MODEL
# ==============================

model_package = {

    "model": final_model,

    "features": X.columns.tolist(),

    "model_name": final_model_name
}


joblib.dump(
    model_package,
    "models/house_price_model.pkl"
)


# ==============================
# FINAL OUTPUT
# ==============================

print("\n==============================")
print("FINAL MODEL")
print("==============================")

print("Model:", final_model_name)
print("MAE:", round(final_mae, 2))
print("RMSE:", round(final_rmse, 2))
print("R2 Score:", round(final_r2, 4))

print("\nModel saved successfully!")
print("models/house_price_model.pkl")

print("\nAll graphs saved in outputs folder.")

print("\nPROJECT COMPLETED!")