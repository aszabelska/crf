### Corresponding coder: Rick Klein raklein22@gmail.com 
# Testing/comparing how well linear regression, random forest, and 
# conditional random forests predict variables in the Many Labs 2 dataset.

# This is the primary file testing multiple variables/conditions.
# ml2_example.R shows a single example to more easily understand the calls

library(randomForest) # for crf
library(party) # for crf
library(tidyverse) # convenience tools for data handling
library(ranger) # for rf
library(caret) # for rf

# read in a function file containing "runmodels" that runs lm, rf, and crf with
# some basic parameters to easily run on different variables
source("./code/functions.R")

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

# data_pca <- prcomp(x = subset(data, select = -c(subjwell)), center = TRUE, scale. = TRUE, tol = 0.5)
# 
# data_pca_df <- data.frame(subjwell = data$subjwell, data_pca$x)
# 
# data <- data_pca_df

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

# Use the "runmodels" function, read in from ./code/functions.R
# We specify the dataset and variable within that dataset we want to use 
# as a depdendent variable. The function fits linear regression, random forest,
# and conditional random forest models predicting this variable from all other
# variables in the dataset. It them compares the variance explained (1) in 
# the training data, and (2) in a validation dataset (e.g., prediction)

#examples: 
result_subjwell <- runmodels(train, validation, "subjwell")
result_Q_TotalDuration <- runmodels(train, validation, "Q_TotalDuration")
result_ross.s1.1_1_TEXT <- runmodels(train, validation, "ross.s1.1_1_TEXT")
result_sise <- runmodels(train, validation, "sise")
result_mood <- runmodels(train, validation, "mood")
result_politics <- runmodels(train, validation, "politics")

# }
