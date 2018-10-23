library("recommenderlab")
m <- matrix(sample(c(as.numeric(0:5), NA), 50,replace=TRUE, prob=c(rep(.4/6,6),.6)), ncol=10, dimnames = list(user=paste("u", 1:5, sep=''),item = paste("i", 1:10, sep='')))
r <- as(m, "realRatingMatrix")
r 
getRatingMatrix(r)          
# The realRatingMatrix can be coerced back into a matrix which is identical to the original
# matrix.
identical(as(r, "matrix"),m)
as(r, "list")
head(as(r, "data.frame"))

# An important operation for rating matrices is to normalize the entries to, e.g., centering to
# remove rating bias by subtracting the row mean from all ratings in the row. This is can be
# easily done using normalize().
r_m <- normalize(r)
r_m
getRatingMatrix(r_m)

# Normalization can be reversed using denormalize().
denormalize(r_m)

# Small portions of rating matrices can be visually inspected using image().
image(r, main = "Raw Ratings")
image(r_m, main = "Normalized Ratings")

# Binarization of data
# A matrix with real valued ratings can be transformed into a 0-1 matrix with binarize() and
# a user specied threshold (min_ratings) on the raw or normalized ratings. In the following
# 18 Developing and Testing Recommendation Algorithms
# only items with a rating of 4 or higher will become a positive rating in the new binary rating
# matrix.

r_b <- binarize(r, minRating=4)
r_b
as(r_b, "matrix")

# The data set contains ratings for 100 jokes on a
# scale from ????10 to +10. All users in the data set have rated 36 or more jokes.

data(Jester5k)
Jester5k

# Jester5k contains 362106 ratings. For the following examples we use only a subset of the data
# containing a sample of 1000 users 
set.seed(1234)
r <- sample(Jester5k, 1000)
r
##User1 rated # jokes
rowCounts(r[1,])
as(r[1,], "list")
# User's average rating
rowMeans(r[1,])

hist(getRatings(r), breaks=100)
hist(getRatings(normalize(r)), breaks=100)
hist(getRatings(normalize(r, method="Z-score")), breaks=100)
hist(rowCounts(r), breaks=50)

# Average rating on jokes
hist(colMeans(r), breaks=20)
# 
# Next, we create a recommender which generates recommendations solely on the popularity
# of items (the number of users who have the item in their profile).

r <- Recommender(Jester5k[1:1000], method = "POPULAR")
names(getModel(r))
recom <- predict(r, Jester5k[1001:1002], n=5)
recom
as(recom, "list")
# 
# Since the top-N lists are ordered, we can extract sublists of the best items in the top-N. For
# example, we can get the best 3 recommendations for each list using bestN().
recom3 <- bestN(recom, n = 3)
recom3
as(recom3, "list")

# Many recommender algorithms can also predict ratings. This is also implemented using
# predict() with the parameter type set to "ratings".

recom <- predict(r, Jester5k[1001:1002], type="ratings")
recom
as(recom, "matrix")[,1:10]
# The prediction contains NA
# for the items rated by the active users.

# Alternatively, we can also request the complete rating matrix which includes the original
# ratings by the user.

recom <- predict(r, Jester5k[1001:1002], type="ratingMatrix")
recom
as(recom, "matrix")[,1:10]


# Evaluation of predicted ratings
# Here we create an evaluation scheme which splits the rst 1000 users in Jester5k
# into a training set (90%) and a test set (10%). For the test set 15 items will be given to the
# recommender algorithm and the other items will be held out for computing the error.
e <- evaluationScheme(Jester5k[1:1000], method="split", train=0.9, given=15, goodRating=5)
e
# 
# User Based Collaborative Filtering
r1 <- Recommender(getData(e, "train"), "UBCF")
r1
# Item Based Collaborative Filtering
r2 <- Recommender(getData(e, "train"), "IBCF")
r2
# we compute predicted ratings for the known part of the test data 
# (15 items for each user) using the two algorithms.
p1 <- predict(r1, getData(e, "known"), type="ratings")
p1
p2 <- predict(r2, getData(e, "known"), type="ratings")
p2

# calculate the error between the prediction and the unknown part of the test
# data.
error <- rbind(UBCF = calcPredictionAccuracy(p1, getData(e, "unknown")), IBCF = calcPredictionAccuracy(p2, getData(e, "unknown")))
error
# user-based collaborative filtering produces a smaller prediction error.

# Evaluation of a top-N recommender algorithm
# we create a 4-fold cross validation scheme with the the Given-3 protocol,
# i.e., for the test users all but three randomly selected items are withheld for evaluation.
scheme <- evaluationScheme(Jester5k[1:1000], method="cross", k=4, given=3, goodRating=5)
scheme
results <- evaluate(scheme, method="POPULAR", type = "topNList",n=c(1,3,5,10,15,20))
results

# The result is an object of class EvaluationResult which contains several confusion matrices.
# For the first run we have 6 confusion matrices represented by rows, one for each of the six
# different top-N lists we used for evaluation. n is the number of recommendations per list.
getConfusionMatrix(results)[[1]]
avg(results)
# ROC Curve
plot(results, annotate=TRUE)
# Precision-recall plot
plot(results, "prec/rec", annotate=TRUE)

set.seed(2016)
scheme <- evaluationScheme(Jester5k[1:1000], method="split", train = .9,k=1, given=-5, goodRating=5)
scheme

algorithms <- list("random items" = list(name="RANDOM", param=NULL),
                   "popular items" = list(name="POPULAR", param=NULL),
                   "user-based CF" = list(name="UBCF", param=list(nn=50)),
                   "item-based CF" = list(name="IBCF", param=list(k=50)))
                   )
## run algorithms

results <- evaluate(scheme, algorithms, type = "topNList",
                    n=c(1, 3, 5, 10, 15, 20))
results
names(results)
plot(results, annotate=c(1,3), legend="bottomright")
plot(results, "prec/rec", annotate=3, legend="topleft")
# 
# we evaluate not top-N recommendations, but how well the algorithms can predict
# ratings.

## run algorithms
results <- evaluate(scheme, algorithms, type = "ratings")
results
plot(results, ylim = c(0,100))

# Using a 0-1 data set

# For comparison we will check how the algorithms compare given less information.

Jester_binary <- binarize(Jester5k, minRating=5)
Jester_binary <- Jester_binary[rowCounts(Jester_binary)>20]
Jester_binary
scheme_binary <- evaluationScheme(Jester_binary[1:1000],method="split", train=.9, k=1, given=3)
scheme_binary

results_binary <- evaluate(scheme_binary, algorithms, type = "topNList", n=c(1,3,5,10,15,20))
plot(results_binary, annotate=c(1,3), legend="topright")
