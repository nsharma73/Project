library(h2o)
h2o.init()
# setwd("~/Desktop/ML Course/Coursera/PML/Project")
# df <- h2o.importFile("~/Desktop/ML Course/Coursera/PML/Project/pml-training.csv")
# df <- h2o.importFile("~/Desktop/ML Course/Coursera/PML/Project/raw_data.csv")
dataset = read.csv('pml-training.csv')
df.hex <- as.h2o(dataset)
head(df.hex)
df.hex['classe'] <- as.factor(df.hex['classe'])
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


m<-h2o.randomForest(x, y, train, nfolds=10)
summary(m)
exm <- h2o.explain(m, test)
exm
h2o.varimp(m)

perf <- h2o.performance(m, train)
perf_test <- h2o.performance(m, test)
perf_valid <- h2o.performance(m, valid)
h2o.confusionMatrix(perf)
h2o.confusionMatrix(perf_test)
h2o.confusionMatrix(perf_valid)


aml <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  max_models = 20,
                  seed = 1)

# View the AutoML Leaderboard
lb <- aml@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)

