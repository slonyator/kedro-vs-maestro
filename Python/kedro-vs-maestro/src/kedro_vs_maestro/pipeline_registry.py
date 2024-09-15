"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, node
from .nodes import load_data, preprocess_data, train_model, evaluate_model


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = Pipeline(
        [
            node(load_data, None, "raw_data"),
            node(
                preprocess_data,
                "raw_data",
                ["X_train", "X_test", "y_train", "y_test"],
            ),
            node(train_model, ["X_train", "y_train"], "model"),
            node(evaluate_model, ["model", "X_test", "y_test"], None),
        ]
    )
    return pipelines
