### Corresponding coder: Rick Klein raklein22@gmail.com 
library(randomForest) # for crf
library(party) # for crf
library(tidyverse) # convenience tools for data handling
library(ranger) # for rf
library(caret) # for rf

# setwd() if not using the .rproj file

# load the ml2 slate 1 data. Using the not-deidentified datasets
data <- readRDS("./data/ML2_RawData_S1.rds")

# Initialize a results data frame
loop_result <- data.frame()

# Set a seed for all randomization, so that results are reproducible. 
# Note: R's default randomization method changed in mid 2019. I'm forcing it 
# to use the old method so this code runs the same in both old and new versions
RNGkind(sample.kind = "Rounding")
set.seed(1)

################################################################################
#### Data preparation ##########################################################
################################################################################

# Select only numeric variables
data <- select_if(data, is.numeric)

# Code the Instructional Manipulation Check/attention check as pass/fail
# so we don't lose it in the next step.
# Participants pass the check if they check all four boxes.
data <- data %>% 
  rowwise() %>% 
  mutate(imc_sum = sum(c(IMC1_1, IMC1_2, IMC1_3, IMC1_4), na.rm = TRUE))

data$imc_pass <- 0
data$imc_pass[data$imc_sum == 4] <- 1

# Major problem is that CRF doesn't play well with missing data (afaik)
# Next few steps try to counteract this.

# Drop any variable with more than X percent missing values
# Many of these are site-specific variables (custom consent pages, etc.)
# with data only from a few sites.
data <- data[, -which(colMeans(is.na(data)) > 0.15)]

# Still some junk variables remaining, deleting them:
data <- data %>%
  select(-Status, 
         -Finished, 
         -ml2int, 
         -ml2int.t_1, 
         -ml2int.t_2, 
         -ml2int.t_3, 
         -ml2int.t_4,
         -van.p1.1,
         -LocationAccuracy,
         -disg2.1,
         -disg1.1,
         -baudv.1,
         -grah1.1,
         -IMC1_1)

# drop rows with missing data in any column
data <- data[complete.cases(data),]

#############
# PCA
#############
# To reduce the # of individual items you can consider doing a PCA first

data_pca <- prcomp(x = subset(data, select = -c(subjwell)), center = TRUE, scale. = TRUE, tol = 0.5)

data_pca_df <- data.frame(subjwell = data$subjwell, data_pca$x)

data <- data_pca_df

# UNCOMMENT IF: You want to loop over several seeds (also uncomment the } at the end of the script)
# for(i in 1:20){
# set.seed(i)

################################################################################
#### Data split ################################################################
################################################################################

# split into training, validation, and test datasets. (50/25/25 split)
# the 'training' dataset is used to build the models. Their predictive accuracy
# out-of-sample is assessed using the 'validation' dataset, after which
# we can tune the model for better prediction. The 'test' dataset is reserved
# as a final unbiased test of performance of the final model (one time, in theory).
rows <- sample(nrow(data)) #creates randomized vector same length as data
data_randomized <- data[rows,] #randomizes df to index from 'rows'
split <- floor(nrow(data)*.50) #creates index to split the file at 50% of data, rounded
split2 <- floor(nrow(data)*.75) #creates index to split the file at 75% of data, rounded
train <- data_randomized[1:split,] #first 1/2 to train
validation <- data_randomized[(split+1):split2,] # next 1/4 to validation
test <- data_randomized[(split2+1):nrow(data),] #remaining 1/4 to test

################################################################################
#### Linear Regression #########################################################
################################################################################

# Running a multiple linear regression as we normally might in the course of 
# research. Predicting the subjective wellbeing DV from all other 
# variables in the dataset.
lm_model <- (lm(data = train, subjwell ~ .)) # the period indicates to use all other vars in the dataframe

# Typically in psych we would look at these results and call it a day:
# summary(lm_model)

# But here we'll test these predictions "out-of-sample". We'll take the 
# model generated on the training data, and make predictions for each observation
# in the 'validation' dataset that the algorythm hasn't seen yet. This is
# prediction, instead of explanation.
lm_prediction_train <- predict(lm_model) 
lm_prediction_validation <- predict(lm_model, newdata = validation) 

# Evaluate in a common metric
lm_performance_train <- cor(train$subjwell, lm_prediction_train)^2
lm_performance_validation <- cor(validation$subjwell, lm_prediction_validation)^2


################################################################################
#### Random Forest #############################################################
################################################################################

# Next we'll fit a basic machine learning model - Random Forests. 
rf_model <- train(
  subjwell ~ ., 
  tuneLength = 1, 
  data = train, method = "ranger", 
  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE)
)

# Extract predictions
rf_prediction_train <- predict(rf_model) #oob
rf_prediction_validation <- predict(rf_model, newdata = validation)

# Evaluate
rf_performance_train <- cor(train$subjwell, rf_prediction_train)^2
rf_performance_validation <- cor(validation$subjwell, rf_prediction_validation)^2

################################################################################
#### Conditional Random Forest #################################################
################################################################################

# Next we try a conditional random forest model, which is RF but with 
# some added properties.
crf_model <- cforest(subjwell ~ ., 
                        data = train, 
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
crf_prediction_validation <- predict(crf_model, newdata = validation)

# Evaluate
crf_performance_train <- as.numeric(cor(train$subjwell, crf_prediction_train)^2)
crf_performance_validation <- as.numeric(cor(validation$subjwell, crf_prediction_validation)^2)

# Compare results from all three models

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
# }

# Write results to .csv if you want
# write.csv(loop_result, './results/loop_result.csv')
