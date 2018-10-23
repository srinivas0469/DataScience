library(e1071) 
data(HouseVotes84, package = "mlbench")
str(HouseVotes84)
#Set training data set and test data set
# I set the first 75% of 435 observatiosn as training, the rest is test
hv_train<-HouseVotes84[1:326,-1]
hv_test<-HouseVotes84[327:435,-1]

# Save labels
hv_train_labels <- HouseVotes84[1:326, ]$Class 
hv_test_labels<- HouseVotes84[327:435, ]$Class

# Train the model
hv_classifier <- naiveBayes(hv_train, hv_train_labels)

# Predict
hv_test_pred <- predict(hv_classifier, hv_test)
head(hv_test_pred)

table(hv_test_pred,hv_test_labels)

accuracy = mean(hv_test_pred == hv_test_labels)
accuracy


hv_classifier2 <- naiveBayes(hv_train, hv_train_labels, laplace = 1)
hv_test_pred2 <- predict(hv_classifier2, hv_test)

table(hv_test_pred2,hv_test_labels)

accuracy = mean(hv_test_pred2 == hv_test_labels)
accuracy
