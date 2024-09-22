# Kedro vs Maestro

In this repository, we provide a basic comparison between `Kedro`, a `Python` framework for creating reproducible, maintainable, and modular data science code, and `maestro`, a new framework for `R` that orchestrates pipelines.

## Background

I truly love R. It’s the language that began my data science journey. Moreover, I believe there are tasks, particularly in "classical Machine Learning," where `R` has, in my opinion, an edge over `Python`. That being said, multiple statements can be true at once. For me, these are:

- I prefer `tidyverse` & `tidymodels` over `pandas` & `sklearn`, because of their readability, consistency, and control over the workflow. It’s unmatched. However, for very large datasets, additional packages may be necessary to improve speed.
- There’s a great saying: `Python` is the second-best language for every problem, and its ecosystem for productionizing code is unparalleled. Tools like `Kedro` and `Airflow` are just two examples from the vast array available.

But here’s the thing: A few weeks ago, [maestro](https://github.com/whipson/maestro) was released on CRAN. It’s a framework for `R` to orchestrate pipelines. I was intrigued and wanted to see how it compares to `Kedro`. While `tidyverse` & `tidymodels` may be excellent, if orchestrating and deploying pipelines is a challenge, I’d prefer a solution that, although not perfect, enables me to accomplish both with ease.

## Comparison

### Installation & Setup

#### Kedro

We assume you have an empty Git repository, along with `Python` and a working environment where `Kedro` is installed. To set up a new project, I ran the following command:

```bash
kedro new --name=kedro-vs-maestro --tools=none --example=n
```

I used this command because I wanted the setup to be as minimal as possible. The `--tools=none` flag ensures that no additional tools are installed, and the `--example=n` flag prevents an example pipeline from being created.

Here's what the project structure looks like if you use the `tree` command:

```bash
.
└── kedro-vs-maestro
    ├── README.md
    ├── conf
    │   ├── README.md
    │   ├── base
    │   │   ├── catalog.yml
    │   │   └── parameters.yml
    │   └── local
    │       └── credentials.yml
    ├── notebooks
    ├── pyproject.toml
    ├── requirements.txt
    └── src
        └── kedro_vs_maestro
            ├── __init__.py
            ├── __main__.py
            ├── pipeline_registry.py
            ├── pipelines
            │   └── __init__.py
            └── settings.py
```

I don't know what you guys think, but when I first saw it, it was a bit overwhelming. I understand that it's a framework designed for more complex projects, but I was hoping for a bit more simplicity. While I'm sure it works well for larger projects, for smaller ones, it seems like an overkill. There are quite a lot of files and directories and you have to do a bit of reading to understand what's going on, as it's not obvious from the start.


#### Maestro

The setup and installation for maestro are easier. You install the `maestro` package from CRAN and run the `create_maestro("./", overwrite = TRUE)` command in your R console. And that's it. The project structure is as follows (using the `tree` command again):

```bash
.
├── orchestrator.R
└── pipelines
    └── my_pipe.R
```

Well that's a lot simpler, isn't it? The `orchestrator.R` file is where you define your pipeline, and the `pipelines` directory is where you store your pipeline scripts. The `my_pipe.R` file is an example pipeline script that you can use to get started. We will examine the contents of these in the next sections.

### Writing a simple pipeline

#### Kedro

We are going to start with a simple pipeline that loads the Iris dataset, preprocesses it, trains a Random Forest model, and evaluates it. Here's the code:

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging


def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
    df["target"] = iris["target"]
    return df


def preprocess_data(df):
    X = df.drop(columns="target")
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logging.info("Accuracy: %s", accuracy)
    logging.info("Classification Report:\n%s", report)
```

you have to put it into a file called `nodes.py` which should be located in `src/kedro_vs_maestro/`. 
As we defined so far only the functions, we have to stack them together in a pipeline. This is done in the `pipeline_registry.py` file, which you have to modify as follows:


```python
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
```

Here we stack together all the functions we defined in the `nodes.py` file. The `node` function is used to define a node in the pipeline. The first argument is the function that should be executed, the second argument is the input of the function, and the third argument is the output of the function.

One more file and we are good to go. We have our datasets in the catalog.yml file, which is located in the `conf/base/` directory. Here's how it looks like:

```yaml
# Here you can define all your data sets by using simple YAML syntax.
#
# Documentation for this file format can be found in "The Data Catalog"
# Link: https://docs.kedro.org/en/stable/data/data_catalog.html
# Update the dataset names in `kedro-vs-maestro/conf/base/catalog.yml`
raw_data:
  type: MemoryDataset

X_train:
  type: MemoryDataset

X_test:
  type: MemoryDataset

y_train:
  type: MemoryDataset

y_test:
  type: MemoryDataset

model:
  type: MemoryDataset
No newline at end of file
```

In our case the `catalog.yml` does not look very spectacular, as we only have memory datasets. In case you store any data or models you would have to specify the path to the file in a `filepath` attribute which you can set for each dataset or model.

Now we are good to go to execute the pipeline. This is simply done by navigating to the `kedro-vs-maestro` directory (or however you named your `kedro` project) and running the following command on the terminal:

```bash
kedro run
```

which gives you the following output:

```bash
[09/21/24 21:49:51] INFO     Using                                                                                            __init__.py:249
                             '/Users/michael/Library/Caches/pypoetry/virtualenvs/kedro-vs-maestro-KMFHHjfA-py3.11/lib/python3
                             .11/site-packages/kedro/framework/project/rich_logging.yml' as logging configuration.
[09/21/24 21:49:51] INFO     Kedro project kedro-vs-maestro                                                                    session.py:327
[09/21/24 21:49:52] INFO     Kedro is sending anonymous usage data with the sole purpose of improving the product. No personal  plugin.py:233
                             data or IP addresses are stored on our side. If you want to opt out, set the
                             `KEDRO_DISABLE_TELEMETRY` or `DO_NOT_TRACK` environment variables, or create a `.telemetry` file
                             in the current working directory with the contents `consent: false`. Read more at
                             https://docs.kedro.org/en/stable/configuration/telemetry.html
                    INFO     Using synchronous mode for loading and saving data. Use the --async flag for potential   sequential_runner.py:67
                             performance gains.
                             https://docs.kedro.org/en/stable/nodes_and_pipelines/run_a_pipeline.html#load-and-save-a
                             synchronously
                    INFO     Running node: load_data(None) -> [raw_data]                                                          node.py:364
                    INFO     Saving data to raw_data (MemoryDataset)...                                                   data_catalog.py:581
                    INFO     Completed 1 out of 4 tasks                                                               sequential_runner.py:93
                    INFO     Loading data from raw_data (MemoryDataset)...                                                data_catalog.py:539
                    INFO     Running node: preprocess_data([raw_data]) -> [X_train;X_test;y_train;y_test]                         node.py:364
                    INFO     Saving data to X_train (MemoryDataset)...                                                    data_catalog.py:581
                    INFO     Saving data to X_test (MemoryDataset)...                                                     data_catalog.py:581
                    INFO     Saving data to y_train (MemoryDataset)...                                                    data_catalog.py:581
                    INFO     Saving data to y_test (MemoryDataset)...                                                     data_catalog.py:581
                    INFO     Completed 2 out of 4 tasks                                                               sequential_runner.py:93
                    INFO     Loading data from X_train (MemoryDataset)...                                                 data_catalog.py:539
                    INFO     Loading data from y_train (MemoryDataset)...                                                 data_catalog.py:539
                    INFO     Running node: train_model([X_train;y_train]) -> [model]                                              node.py:364
                    INFO     Saving data to model (MemoryDataset)...                                                      data_catalog.py:581
                    INFO     Completed 3 out of 4 tasks                                                               sequential_runner.py:93
                    INFO     Loading data from model (MemoryDataset)...                                                   data_catalog.py:539
                    INFO     Loading data from X_test (MemoryDataset)...                                                  data_catalog.py:539
                    INFO     Loading data from y_test (MemoryDataset)...                                                  data_catalog.py:539
                    INFO     Running node: evaluate_model([model;X_test;y_test]) -> None                                          node.py:364
                    INFO     Completed 4 out of 4 tasks                                                               sequential_runner.py:93
                    INFO     Pipeline execution completed successfully.                                                         runner.py:123
```

we not going to digest the logs in detail, but you can see that the pipeline terminated successfuly and you can also see the logs of the individual nodes. Via the CLI you can set a whole lot of parameters, in case you want to run the pipeline in a asynchronous way, or you want to run only a specific pipeline, or you want to run only a specific node. You can find more information [here](https://docs.kedro.org/en/stable/nodes_and_pipelines/run_a_pipeline.html).

#### Maestro

For `maestro` like the setup the whole thing is easier. You put your `R` functions just in a file (I called it `iris_model` but the name does not really make a difference) and put it in the `pipelines` directory. Here's the code:

```R
#' Iris Model Pipeline
#' @maestroFrequency 1 day
#' @maestroStartTime 2024-03-25 12:30:00
iris_model_pipeline <- function() {
  load_data <- function() {
    log_info("Data Loading started")
    data(iris)
    log_info("Data Loading completed")
    return(iris)
  }

  preprocess_data <- function(iris_data) {
    log_info("Data Preprocessing started")
    set.seed(42)
    iris_split <- initial_split(iris_data, prop = 0.8, strata = Species)
    iris_train <- training(iris_split)
    iris_test <- testing(iris_split)
    log_info("Data Preprocessing completed")
    return(list(train = iris_train, test = iris_test))
  }

  train_model <- function(iris_train) {
    log_info("Model Training started")
    rf_spec <- rand_forest(mtry = 2, trees = 100)  |>
      set_engine("randomForest") |>
      set_mode("classification")

    rf_workflow <- workflow() |>
      add_formula(Species ~ .) |>
      add_model(rf_spec)

    rf_fit <- rf_workflow |> fit(data = iris_train)
    log_info("Model Training completed")
    return(rf_fit)
  }

  evaluate_model <- function(rf_fit, iris_test) {
    log_info("Model Evaluation started")
    iris_predictions <- predict(rf_fit, iris_test) |> bind_cols(iris_test)

    accuracy <- iris_predictions |>
      metrics(truth = Species, estimate = .pred_class) |>
      filter(.metric == "accuracy")

    conf_matrix <- iris_predictions |>
      conf_mat(truth = Species, estimate = .pred_class)

    log_info("Model Evaluation completed")
    log_info("Accuracy: %s", accuracy$.estimate)
    log_info("Confusion Matrix:\n%s", conf_matrix)
  }

  iris_data <- load_data()
  split_data <- preprocess_data(iris_data)
  rf_fit <- train_model(split_data$train)
  evaluate_model(rf_fit, split_data$test)
}
```

The only thing which is different to a normal `R` script where you store your functions are the roxygen tags which define the frequency and the start time of the pipeline. The `@maestroFrequency` tag defines how often the pipeline should be executed and the `@maestroStartTime` tag defines when the pipeline should be executed for the first time.

To execute the pipeline we have to write some code in the `orchestrator.R` file:

```R
library(maestro)
library(logger)
library(tidyverse)
library(tidymodels)

log_info("Building schedule")

schedule_table <- build_schedule(pipeline_dir = "pipelines")

log_info("Running scheduled pipelines")

output <- run_schedule(
  schedule_table, 
  orch_frequency = "1 day"
)

log_info("Pipeline run completed")
```
Pretty straightforward. You execute it by running the `orchestrator.R` file in your R console or via the terminal with the command (assuming you are in the correct directory):

```bash
Rscript orchestrator.R
```

which gives you the following output:

```bash
- The project is out-of-sync -- use `renv::status()` for details.
── Attaching core tidyverse packages ───────────────────────────────────────────────────────────────────────────────────── tidyverse 2.0.0 ──
✔ dplyr     1.1.4     ✔ readr     2.1.5
✔ forcats   1.0.0     ✔ stringr   1.5.1
✔ ggplot2   3.5.1     ✔ tibble    3.2.1
✔ lubridate 1.9.3     ✔ tidyr     1.3.1
✔ purrr     1.0.2
── Conflicts ─────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
✖ dplyr::filter() masks stats::filter()
✖ dplyr::lag()    masks stats::lag()
ℹ Use the conflicted package to force all conflicts to become errors
── Attaching packages ─────────────────────────────────────────────────────────────────────────────────────────────────── tidymodels 1.2.0 ──
✔ broom        1.0.6     ✔ rsample      1.2.1
✔ dials        1.3.0     ✔ tune         1.2.1
✔ infer        1.0.7     ✔ workflows    1.1.4
✔ modeldata    1.4.0     ✔ workflowsets 1.1.0
✔ parsnip      1.2.1     ✔ yardstick    1.3.1
✔ recipes      1.1.0
── Conflicts ────────────────────────────────────────────────────────────────────────────────────────────────────── tidymodels_conflicts() ──
✖ scales::discard() masks purrr::discard()
✖ dplyr::filter()   masks stats::filter()
✖ recipes::fixed()  masks stringr::fixed()
✖ dplyr::lag()      masks stats::lag()
✖ yardstick::spec() masks readr::spec()
✖ recipes::step()   masks stats::step()
• Search for functions across packages at https://www.tidymodels.org/find/
INFO [2024-09-21 22:37:36] Building schedule
ℹ 1 script successfully parsed
INFO [2024-09-21 22:37:36] Running scheduled pipelines

── Running pipelines ▶
INFO [2024-09-21 22:37:36] Data Loading started
INFO [2024-09-21 22:37:36] Data Loading completed
INFO [2024-09-21 22:37:36] Data Preprocessing started
INFO [2024-09-21 22:37:36] Data Preprocessing completed
INFO [2024-09-21 22:37:36] Model Training started
INFO [2024-09-21 22:37:36] Model Training completed
INFO [2024-09-21 22:37:36] Model Evaluation started
INFO [2024-09-21 22:37:36] Model Evaluation completed
INFO [2024-09-21 22:37:36] Accuracy: %s0.9
INFO [2024-09-21 22:37:36] Confusion Matrix:
%slist(table = c(10, 0, 0, 0, 9, 1, 0, 2, 8))
✔ pipelines/iris_model.R iris_model_pipeline [73ms]

── Pipeline execution completed ■ | 0.08 sec elapsed
✔ 1 success | → 0 skipped | ! 0 warnings | ✖ 0 errors | ◼ 1 total
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

── Next scheduled pipelines ❯
Pipe name | Next scheduled run
• iris_model_pipeline | 2024-09-23
INFO [2024-09-21 22:37:36] Pipeline run completed
```

Simple, but here is the big, big BUT, which is kind of hidden under the surface. **You cannot build a DAG with `maestro`**. You can only define a pipeline which is executed sequentially. This is a big drawback in my opinion, as you cannot define complex pipelines with dependencies between the nodes. How to schedule a the execution of one "node" after the previous one is finished? It's just not possible with `maestro`.