import time
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    task,
):

    if task == "Classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(
    n_neighbors=min(5, max(1, len(X_train) - 1))
),
            "Support Vector Machine": SVC(),
        }

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
        }

    results = []

    for name, model in models.items():

        start = time.time()

        model.fit(X_train, y_train)

        end = time.time()

        predictions = model.predict(X_test)

        if task == "Classification":

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "F1 Score": f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "Training Time": round(end - start, 4),
            })

        else:

            mse = mean_squared_error(y_test, predictions)

            results.append({
                "Model": name,
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mse,
                "RMSE": mse ** 0.5,
                "R² Score": r2_score(y_test, predictions),
                "Training Time": round(end - start, 4),
            })

    df = pd.DataFrame(results)

    if task == "Classification":

        df = df.sort_values(
            by="Accuracy",
            ascending=False,
        )

    else:

        df = df.sort_values(
            by="R² Score",
            ascending=False,
        )

    df = df.reset_index(drop=True)

    return df