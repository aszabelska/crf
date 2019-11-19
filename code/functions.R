runmodels <- function(training_df, validation_df, dv){
  
  ################################################################################
  #### Linear Regression #########################################################
  ################################################################################
  lm_model <- lm(data = training_df, formula = paste(dv, "~.")) # the period indicates to use all other vars in the dataframe
  
  # Typically in psych we would look at these results and call it a day:
  # summary(lm_model)
  
  # But here we'll test these predictions "out-of-sample". We'll take the 
  # model generated on the training data, and make predictions for each observation
  # in the 'validation' dataset that the algorythm hasn't seen yet. This is
  # prediction, instead of explanation.
  lm_prediction_train <- predict(lm_model) 
  lm_prediction_validation <- predict(lm_model, newdata = validation_df) 
  
  # Evaluate in a common metric
  lm_performance_train <- cor(training_df[[dv]], lm_prediction_train)^2
  lm_performance_validation <- cor(validation_df[[dv]], lm_prediction_validation)^2

  
  ################################################################################
  #### Random Forest #############################################################
  ################################################################################
  
  # Next we'll fit a basic machine learning model - Random Forests. 
  rf_model <- caret::train(
    form = as.formula(paste(dv, "~.")), 
    tuneLength = 1, 
    data = training_df, method = "ranger", 
    trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
  )
  
  # Extract predictions
  rf_prediction_train <- predict(rf_model) #oob
  rf_prediction_validation <- predict(rf_model, newdata = validation_df)
  
  # Evaluate
  rf_performance_train <- cor(training_df[[dv]], rf_prediction_train)^2
  rf_performance_validation <- cor(validation_df[[dv]], rf_prediction_validation)^2
  
  ################################################################################
  #### Conditional Random Forest #################################################
  ################################################################################
  
  # Next we try a conditional random forest model, which is RF but with 
  # some added properties.
  crf_model <- cforest(as.formula(paste(dv, "~ .")), 
                       data = training_df, 
                       controls = cforest_control(teststat = "quad", 
                                                  testtype = "Univ", 
                                                  mincriterion = 0.95, 
                                                  ntree = 500,
                                                  mtry = 12,
                                                  replace = FALSE,
                                                  fraction = 0.632))
  
  # With CRF we can view the 'variable importance', or how much each variable
  # weighs into the prediction
  crf_variable_importance <- data.frame(varimp(crf_model))
  # keep row names as a column
  crf_variable_importance <- tibble::rownames_to_column(crf_variable_importance, "var")
  # Sort from strongest to weakest
  crf_variable_importance <- dplyr::arrange(crf_variable_importance, desc(varimp.crf_model.))
  
  # Extract predictions
  crf_prediction_train <- predict(crf_model) #oob
  crf_prediction_validation <- predict(crf_model, newdata = validation_df)
  
  # Evaluate
  crf_performance_train <- as.numeric(cor(training_df[[dv]], crf_prediction_train)^2)
  crf_performance_validation <- as.numeric(cor(validation_df[[dv]], crf_prediction_validation)^2)
  
  # Compile results from all three models in a single DF
  compare_results <- data.frame(lm_performance_train,
                                lm_performance_validation,
                                rf_performance_train,
                                rf_performance_validation,
                                crf_performance_train,
                                crf_performance_validation)
  
  names(compare_results) <- c("Linear Regression R^2 - Training",
                              "Linear Regression R^2 - Validation",
                              "Random Forest R^2 - Training",
                              "Random Forest R^2 - Validation",
                              "Conditional RF R^2 - Training",
                              "Conditional RF R^2 - Validation")
  
  loop_result <- rbind(loop_result, compare_results)
  return(loop_result)
  }

# example use: training dataframe = train, validation dataframe = validation, dependent variable = subjwell (subjective wellbeing)
# runmodels(train, validation, "subjwell")


