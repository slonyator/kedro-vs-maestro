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

