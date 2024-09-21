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