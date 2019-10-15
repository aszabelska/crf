### Corresponding coder: Rick Klein raklein22@gmail.com 
library(randomForest)
library(party)
library(tidyverse)

# setwd() if not using the .rproj file

# set a seed for all randomization. Note: R's default randomization method
# changed in mid 2019, may not produce identical results in old versions.
set.seed(1)

# load the ml2 slate 1 data. Using the not-deidentified datasets
data <- readRDS("./data/ML2_RawData_S1.rds")

# # dropping some irrelevant columns. CRF has problems handling missing data.
# data <- data %>%
#   select(StartDate:education, Source.Global:IDiffOrderN)
# 
# # For the moment I'm going restrictively select only a few vars
# data <- data %>%
#   select(subjwell, tipi_1:tipi_10, mood)

# Select all numeric variables
data <- select_if(data, is.numeric)

# code the IMC pass/fail so we don't lose it in the next step.
data <- data %>% 
  rowwise() %>% 
  mutate(imc_sum = sum(c(IMC1_1, IMC1_2, IMC1_3, IMC1_4), na.rm = TRUE))

data$imc_pass <- 0
data$imc_pass[data$imc_sum == 4] <- 1

# Let's drop any variable with more than X percent missing values
# These are typically site-specific variables (custom consent pages, etc.)
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
         -van.p1.1)

# drop rows with missing data in any column
data <- data[complete.cases(data),]

# split into training and test (50/50 split)
rows <- sample(nrow(data)) #creates randomized vector same length as data
data_randomized <- data[rows,] #randomizes df to index from 'rows'
split <- round(nrow(data)*.50) #creates index to split the file into 2/3 1/3, rounded
train <- data_randomized[1:split,] #first 2/3 to train
test <- data_randomized[(split+1):nrow(data),] #remaining 1/3 to test

# fit the conditional forest
forest_model <- cforest(subjwell ~ ., 
                        data = train, 
                        controls = cforest_control(teststat = "quad", 
                                                   testtype = "Univ", 
                                                   mincriterion = 0.95, 
                                                   ntree = 500,
                                                   mtry = 5,
                                                   replace = FALSE,
                                                   fraction = 0.632))

# Extract predictions
model_prediction_train <- predict(forest_model) #oob
model_prediction_test <- predict(forest_model, newdata = test)

# Evaluate
cor(train$subjwell, model_prediction_train)^2
cor(test$subjwell, model_prediction_test)^2

# compare with a regular regression
lm_model <- (lm(data = train, subjwell ~ .)) # the period indicates to use all other vars in the dataframe

# Typically in psych we would look at these results and call it a day:
# summary(lm_model)

# but here we'll again test these predictions in data the 'algorythm' hasn't seen yet
model_prediction_lm_train <- predict(lm_model) 
model_prediction_lm_test <- predict(lm_model, newdata = test) 

# evaluate in a common metric like before
cor(train$subjwell, model_prediction_lm_train)^2
cor(test$subjwell, model_prediction_lm_test)^2



