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
