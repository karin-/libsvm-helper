require(randomForest)

setwd("~/Dropbox/Documents/Pomona/2014-2015\ Spring/cs145/yahoofinance")

scores <- c()
data <- read.csv("input.csv")
data[,1] <- as.factor(data[,1])
data[,-1] <- as.matrix(data[,-1])

sampledRows <- sample(1:nrow(data), 100)
sampledCols <- c(1, sample(2:ncol(data), 1000))
smallData <- data[sampledRows, sampledCols]

rf <- randomForest(smallData[,-1], smallData[,1])

testRows <- sample(1:nrow(data), 100)
testCols <- sampledCols[-1]
newData <- data[testRows, testCols]
preds <- predict(rf, newData)
accuracy = mean(preds == data[testRows,1])

scores <- c(scores, accuracy)

# not too bad, I'm getting ~ 80% accuracy
cat(mean(scores), "+=", sd(scores) / sqrt(length(scores)))

sCols <- sample(2:ncol(data), 2000)
sData <- data[,sCols]
sLabels <- data[,1]
tRows <- sample(1:nrow(data), 200)
tData <- data[tRows,sCols]
tLabels <- data[tRows,1]
rf <- randomForest(sData, y = sLabels, xtest = tData, ytest = tLabels)
rf
