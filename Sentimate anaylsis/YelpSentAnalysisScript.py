### James Grandin
### Yelp Reviews Sentiment Analysis
### MATH 5030
### MAC os





###import libraries and functions
import pandas as pd  # used to read in files of mixed data types. Integer, string, boolean"
import re  # regular expressions to find/replace strings in file
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
###Read in Data and split into training and test data

fileName = "YelpReviewFormatted.csv"
SentimentValue = "SentimentValue"
ReviewText = "ReviewText"

stop_words = set(stopwords.words('english'))


df = pd.read_csv(fileName, header=0, quoting=2)

X_train, X_test, y_train, y_test = train_test_split(df[ReviewText], df[SentimentValue], test_size=.2)

print(X_train, X_test, y_train, y_test)

###Defining Functions

def clean_my_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", "", text)             # quick removal of HTML tags
    text = re.sub(r"http\S+", " ", text)          # removes all URLs
    text = re.sub("[^a-zA-Z#@]", " ", text)       # strip out all non-alpha chars
    text = text.strip().lower()                   # convert all text to lowercase
    text = re.sub("n t ", " not ", text)          # change orphan n't into not
    text = re.sub(r'\W*\b\w{1,2}\b', "", text)    # removes all words 1-2 chars long


    tokenizer = nltk.tokenize.TreebankWordTokenizer()  # tokenizes text using
                                                       # smart divisions
    tokens = tokenizer.tokenize(text)      # store results in tokens

    unstopped = []                         # holds the cleaned data string
    for word in tokens:
        if word not in stop_words:         # removes stopwords
            unstopped.append(word)         # adds word to unstopped string
    stemmer = nltk.stem.WordNetLemmatizer()   # consolidates word forms
    cleanText = " ".join(stemmer.lemmatize(token) for token in unstopped)
                # joins final clean tokens into a string
    return cleanText

def clean_my_data(dataList):
    print("Cleaning all of the data")
    i = 0
    for textEntry in dataList:              # reads line of text under
                                                    # review category
        cleanElement = clean_my_text(textEntry)     # cleans line of text
        dataList[i] = cleanElement   # stores cleaned text
        i = i + 1
        if (i%500 == 0):
            print("Cleaning review number", i, "out of", len(dataList))
    print("Finished cleaning all of the data\n")
    return dataList


def train_logistic_regression(features, label):
    print ("Training the logistic regression model...")
    from sklearn.linear_model import LogisticRegression
    ml_model = LogisticRegression(C = 100, random_state = 0, solver = 'liblinear')
    ml_model.fit(features, label)
    print ('Finished training the model\n')
    return ml_model


def create_bag_of_words(X):
    from sklearn.feature_extraction.text import CountVectorizer
    # use scikit-learn for vectorization

    print('Generating bag of words...')

    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 ngram_range=(1, 2), \
                                 max_features=10000)
    # generates vectorization for ngrams of up to 2 words in length
    # this will greatly increase feature size, but gives more accurate
    # sentiment analysis since some word combinations have large
    # impact on sentiment ie: ("not good", "very fast")

    data_features = vectorizer.fit_transform(X)
    # vectorizes sparse matrix using calculated mean and variance
    data_features = data_features.toarray()
    # convert to a NumPy array for efficient matrix operations
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(data_features)
    # use tf-idf to weight features - places highest sentiment value on
    # low-frequency ngrams that are not too uncommon
    return vectorizer, tfidf_features, tfidf

###Convert Cleaned Data to TF-IDF Matrices

print("Operating on training data...\n")
trainReviews = X_train.tolist()
cleanReviewData = clean_my_data(trainReviews)            # cleans the training data
vectorizer, train_tfidf_features, tfidf  = (create_bag_of_words(cleanReviewData))
    # stores the sparse matrix of the tf-idf weighted features

###Train Logistical Regression Model with Training Data
ml_model = train_logistic_regression(train_tfidf_features, y_train)
            # holds the trained model


#####PART 2

###Clean Test Data

print("Operating on test data...\n")
testReviews = X_test.tolist()
cleanTestData = clean_my_data(testReviews)
            # cleans the test data for accuracy evaluation


#####Convert Test Data to TF-IDF Matrices

test_data_features = vectorizer.transform(cleanTestData)
            # vectorizes the test data using the mean and variance from the training data
test_data_features = test_data_features.toarray()

test_tfidf_features = tfidf.transform(test_data_features)
            #tfidf transform of the vectorized text data using the previous mean and variance
test_tfidf_features = test_tfidf_features.toarray()
####Apply Model to Make Predictions

predicted_y = ml_model.predict(test_tfidf_features)
    # uses the trained logistic regression model to assign sentiment to each
    # test data example

###Test the accuracy of model

correctly_identified_y = predicted_y == y_test
accuracy = np.mean(correctly_identified_y) * 100
print ('The accuracy of the model in predicting Yelp sentiment is %.0f%%' %accuracy)
    # compares the predicted sentiment (predicted_y) vs the actual
    # sentiment stored in y_test



