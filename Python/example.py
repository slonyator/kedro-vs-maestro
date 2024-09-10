"""
This module trains and evaluates a Random Forest classifier on the iris dataset.
"""

import logging
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Data Loading started")
    iris = load_iris()
    df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
    logging.info("Data Loading completed")

    logging.info("Data Preprocessing started")
    df["target"] = iris["target"]

    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logging.info("Data Preprocessing completed")

    logging.info("Model Training started")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model Training completed")

    logging.info("Model Evaluation started")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logging.info("Model Evaluation completed")

    logging.info("Accuracy: %s", accuracy)
    logging.info("Classification Report:\n%s", report)
