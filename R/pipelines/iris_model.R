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
