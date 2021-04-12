library(h2o)
h2o.init()
h2o.removeAll()
# set working directory
setwd("~/Desktop/ML Course/Coursera/PML/Project")

# Load the dataset
dataset = read.csv('pml-training.csv')
df.hex <- as.h2o(dataset)
head(df.hex)

# the classe variable should be a factor variable
df.hex['classe'] <- as.factor(df.hex['classe'])

# Split the data into train, test and validation samples
parts<-h2o.splitFrame(df.hex, c(0.6,0.2), seed = -1)
train <- parts[[1]]
valid <- parts[[2]]
test <- parts[[3]]
x<-colnames(train, do.NULL = FALSE)[8:159]
y<-colnames(train, do.NULL = FALSE)[160]

table(dataset$classe)
table(as.data.frame(train$classe))
table(as.data.frame(test$classe))
table(as.data.frame(valid$classe))


m<-h2o.randomForest(x, y, train, nfolds=7)
summary(m)
h2o.varimp(m)
h2o.confusionMatrix(m,test)
# error 0.0087
h2o.performance(m,test)


g1<- h2o.grid("gbm", grid_id = "GBM_stew",
              search_criteria = list(
                strategy = "RandomDiscrete",
                max_models = 20
              ),
              hyper_params = list(
                max_depth = c(5,10,15),
                min_rows = c(2,5,10),
                sample_rate = c(0.6,0.8,1.0),
                col_sample_rate = c(0.5,0.8,1.0),
                col_sample_rate_per_tree = c(0.8,1.0),
                #learn_rate = c(0.1),
                seed = 12345
              ),
              x = x, y = y, training_frame = train, validation_frame = valid,
              
              stopping_tolerance = 0.001,
              stopping_rounds = 3,
              score_tree_interval = 10,
              ntrees = 400
              )


gbm_gridperf1 <- h2o.getGrid(grid_id = "GBM_stew",
                             sort_by = "accuracy",
                             decreasing = TRUE)

print(gbm_gridperf1)

gbm_gridperf2 <- h2o.getGrid(grid_id = "GBM_stew",
                               sort_by = "logloss",
                               decreasing = FALSE)
print(gbm_gridperf2)

gbm_gridperf3 <- h2o.getGrid(grid_id = "GBM_stew",
                             sort_by = "MSE",
                             decreasing = FALSE)

print(gbm_gridperf3)

gbm_gridperf4 <- h2o.getGrid(grid_id = "GBM_stew",
                             sort_by = "rmse",
                             decreasing = FALSE)

print(gbm_gridperf4)

m_gbm <- h2o.getModel(g1@model_ids[[1]])


h2o.varimp(m_gbm)
h2o.confusionMatrix(m_gbm,test)
h2o.performance(m_gbm,test)
# error is 0.004

h2o.varimp_plot(m_gbm)
h2o.pd_plot(m_gbm, column = 'roll_belt', newdata = test)

h2o.partialPlot(object = m_gbm, data = test, cols = "roll_belt", 
                targets=c("A", "B", "C", "D", "E"))




dataset_test = read.csv('pml-testing.csv')
df_test.hex <- as.h2o(dataset_test)
head(df_test.hex)
dim(df_test.hex)

# Predict using the GBM model and the testing dataset
pred <- h2o.predict(object = m_gbm, newdata = df_test.hex[8:159])
dim(pred)

h2o.exportFile(pred, "pred.csv")

