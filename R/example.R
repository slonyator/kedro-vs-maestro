library(tidyverse)
library(tidymodels)
library(logger)

log_info("Data Loading started")
data(iris)
log_info("Data Loading completed")

log_info("Data Preprocessing started")
set.seed(42)
iris_split <- initial_split(iris, prop = 0.8, strata = Species)
iris_train <- training(iris_split)
iris_test <- testing(iris_split)
log_info("Data Preprocessing completed")

log_info("Model Training started")
rf_spec <- rand_forest(mtry = 2, trees = 100)  |> 
  set_engine("randomForest") |> 
  set_mode("classification")

rf_workflow <- workflow() |> 
  add_formula(Species ~ .) |> 
  add_model(rf_spec)

rf_fit <- rf_workflow |> 
  fit(data = iris_train)

log_info("Model Training completed")

log_info("Model Evaluation started")
iris_predictions <- predict(rf_fit, iris_test)  |> 
  bind_cols(iris_test)

accuracy <- iris_predictions  |> 
  metrics(truth = Species, estimate = .pred_class)  |> 
  filter(.metric == "accuracy")

conf_matrix <- iris_predictions  |> 
  conf_mat(truth = Species, estimate = .pred_class)

log_info("Model Evaluation completed")

log_info("Accuracy: %s", accuracy$.estimate)
log_info("Confusion Matrix:\n%s", conf_matrix)
