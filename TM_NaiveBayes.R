# Load the data into R.
setwd("F:\\Datascience")
sms_raw <- read.csv("sms_spam.csv", header = T, sep = ",", stringsAsFactors = FALSE)

str(sms_raw)

# Change the data type
sms_raw$type <- factor(sms_raw$type)
str(sms_raw$type)
table(sms_raw$type)
prop.table(table(sms_raw$type))

# Text Mining
# For Naive Bayes to run effectively, the test data needs to be transformed. 

install.packages("tm")
library(tm)

# create a volitile coprus that contains the "text" vector from our data frame.
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
print(sms_corpus)

# Check out the first few messages in the new corpus, which is basically a 
# list that can be manipulated with list operations.
inspect(sms_corpus[1:3])

# Use "as.character" function to see what a message looks like.
as.character(sms_corpus[[3]])

# In order to standardize the messages, the data set must be tranformed to all lower case letters.
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

# Look at third message again to see if our data was transformed.
as.character(sms_corpus[[3]])
as.character((sms_corpus_clean[[3]]))

# remove the numbers using the "removeNumbers" function.
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

# Remome words that appear often but don't contribute to our objective. These words include "to", "and", "but" and "or".
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

# Remove punctuation as well using the "removePunctuation" function.
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
as.character((sms_corpus_clean[[3]]))


# Perform "stemming" to the text data to strip the suffix from words like "jumping", so the words "jumping" "jumps" and "jumped" 
# are all transformed into "jump".

install.packages("SnowballC")
library(SnowballC)
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

# final step in text mining is to remove white space from the document.

sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
as.character(sms_corpus_clean[[3]])

# Perform tokenization using the "DocumentTermMatrix" function. 
# This creates a matrix in which the rows indicat documents 
# and the columns indicate words.

sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm
inspect(sms_dtm[500:505, 500:505])
findFreqTerms(sms_dtm, 100)
help(findFreqTerms)
d <- as.matrix(sms_dtm)
d[1,1:5]
v <- sort(colSums(d), decreasing = T)
head(v,14)
v[1:14]
words <- names(v)
df <- data.frame(word=words, freq=v)
head(df)
install.packages("wordcloud")
library(wordcloud)
wordcloud(df$word,df$freq,min.freq = 50)
wordcloud(df$word,df$freq,max.words = 50)
# termfrequency (tf) = frequency a term in a document / frequency of the most occurring term in the document

freq = data.frame(sort(colSums(as.matrix(sms_dtm)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))

# inverse document frequency (idf) = log(total number of documents / number of documents containing the term)
# Term frequency-Inverse document frequency (tf-idf) = tf*idf
sms_dtm_tfidf <- DocumentTermMatrix(sms_corpus_clean, control = list(weighting = weightTfIdf))
sms_dtm_tfidf
inspect(sms_dtm_tfidf[1,1:20])
freq = data.frame(sort(colSums(as.matrix(sms_dtm_tfidf)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))


# Data Preparation
# Split our data into training and testing sets

sms_dtm_train <- sms_dtm[1:4180, ]
sms_dtm_test <- sms_dtm[4181:5559, ]

# Save vectors labeling rows in the training and testing vectors
sms_train_labels <- sms_raw[1:4180, ]$type
sms_test_labels <- sms_raw[4181:5559,]$type


# Make sure that the proportion of spam is similar in the training and testing data set.
prop.table(table(sms_train_labels))

prop.table(table(sms_test_labels))

# Visualization
# Create a wordcloud of the frequency of the words in the dataset

install.packages("wordcloud")
library(wordcloud)

wordcloud(sms_corpus_clean, max.words = 50, random.order = FALSE)

spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")
wordcloud(spam$text, max.words = 50, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 50, scale = c(3, 0.5))

# Preparation for Naive Bayes
# Remove words from the matrix that appear less than 5 times.
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)

# Limit our Document Term Matrix to only include words in the sms_freq_vector. 
# We want all the rows, but we want to limit the column to these words in the 
# frequency vector.

sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

# The naive bayes classifier works with categorical reatures, 
# so we need to convert the matrix to "yes" and "no" categorical variables. 
# To do this we'll build a convert_counts function and apply it to our data.
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

# Replace values greater than 0 with yes, and values not greater than 0 with no. 
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

# Train Model on the Data.
# Use the e1071 package to impliment the Naive Bayes algorithm on the data, 
# and predict whether a message is likely to be spam or ham.

install.packages("e1071")

library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

# Predict and Evaluate the Model.
sms_test_pred <- predict(sms_classifier, sms_test)

# Confusion Matrix
table(Pred = sms_test_pred, Actual = sms_test_labels)

accuracy = mean(sms_test_pred == sms_test_labels); accuracy

# Show the 5 most frequent words in the sms data:
sack <- TermDocumentMatrix(sms_corpus_clean)
pack <- as.matrix(sack)
snack <- sort(rowSums(pack), decreasing = TRUE)
hack <- data.frame(word = names(snack), freq=snack)
head(hack, 5)

# Visualize the most frequent words from each class:
wordcloud(spam$text, max.words = 10, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 10, scale = c(3, 0.5))
install.packages("RWeka")
library(RWeka)

# creating n-gram clouds
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram = TermDocumentMatrix(sms_corpus_clean,control = list(tokenize = BigramTokenizer))
freq = sort(rowSums(as.matrix(tdm.bigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
wordcloud(freq.df$word,freq.df$freq,max.words=10)
