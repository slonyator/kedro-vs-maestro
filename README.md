# Kedro vs Maestro

In this repository, we provide a basic comparison between `Kedro`, a `Python` framework for creating reproducible, maintainable, and modular data science code, and `maestro`, a new framework for `R` that orchestrates pipelines.

# Background

I truly love R. It’s the language that began my data science journey. Moreover, I believe there are tasks, particularly in "classical Machine Learning," where `R` has, in my opinion, an edge over `Python`. That being said, multiple statements can be true at once. For me, these are:

- I prefer `tidyverse` & `tidymodels` over `pandas` & `sklearn`, because of their readability, consistency, and control over the workflow. It’s unmatched. However, for very large datasets, additional packages may be necessary to improve speed.
- There’s a great saying: `Python` is the second-best language for every problem, and its ecosystem for productionizing code is unparalleled. Tools like `Kedro` and `Airflow` are just two examples from the vast array available.

But here’s the thing: A few weeks ago, [maestro](https://github.com/whipson/maestro) was released on CRAN. It’s a framework for `R` to orchestrate pipelines. I was intrigued and wanted to see how it compares to `Kedro`. While `tidyverse` & `tidymodels` may be excellent, if orchestrating and deploying pipelines is a challenge, I’d prefer a solution that, although not perfect, enables me to accomplish both with ease.
