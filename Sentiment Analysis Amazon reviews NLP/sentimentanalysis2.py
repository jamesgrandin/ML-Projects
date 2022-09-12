import numpy as np  # increases efficiency of matrix operations
import pandas as pd  # reads in data files of mixed data types
import re  # regular expressions to find/replace strings
import nltk  # natural language toolkit
import os
nltk.download('wordnet')
# out non-sentiment filler words
from sklearn.model_selection import train_test_split

path = os.path.expanduser("~/Desktop/Machine Learning Class/Sentiment Analysis/venv/stop_words.txt")
stop_words = open(path,'r')
 # make the stopword list a set
# to increase speed of comparisons

df = pd.read_csv("trainingData5000.txt", header=0, delimiter="\t", quoting=3)
# read the training data stored in "trainingDataXXXX.txt"
# note: data files are tab delimited

X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2)

""" clean_my_text(): cleans the data with several replacements/deletions,
    tokenizes the text, and removes stopwords
    input: string data
    output: cleaned string data ready for sentiment analysis
"""


def clean_my_text(text):
    text = re.sub(r"<.*?>", "", text)  # quick removal of HTML tags
    text = re.sub("[^a-zA-Z]", " ", text)  # strip out all non-alpha chars
    text = text.strip().lower()  # convert all text to lowercase
    text = re.sub(" s ", " ", text)  # remove isolated s chars that
    # result from cleaning possessives

    tokenizer = nltk.tokenize.TreebankWordTokenizer()  # tokenizes text using
    # smart divisions
    tokens = tokenizer.tokenize(text)  # store results in tokens

    unstopped = []  # holds the cleaned data string
    for word in tokens:
        if word not in stop_words:  # removes stopwords
            unstopped.append(word)  # adds word to unstopped string
    stemmer = nltk.stem.WordNetLemmatizer()  # consolidates word forms
    cleanText = " ".join(stemmer.lemmatize(token) for token in unstopped)
    # joins final clean tokens into a string
    return cleanText


""" clean_my_data() calls clean_my_text for each line of text in a dataset
    category  
    input: data file containing raw text  
    output: data file containing cleaned text entries
"""


def clean_my_data(dataList):
    print("Cleaning all of the data")
    i = 0
    for textEntry in dataList:  # reads line of text under
        # review category
        cleanElement = clean_my_text(textEntry)  # cleans line of text
        dataList[i] = cleanElement  # stores cleaned text
        i = i + 1
        if (i % 50 == 0):
            print("Cleaning review number", i, "out of", len(dataList))
    print("Finished cleaning all of the data\n")
    return dataList


""" create_bag_of_words() generates the bag of words used to evaluate sentiment
    input: cleaned dataset
    output: tf-idf weighted sparse matrix
"""


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


""" train_logistic_regression() uses logistic regression model to
    evaluate sentiment
    options: C sets how strong regularization will be: large C = small amount
    input: tf-idf matrix and the sentiment attached to the training example
    output: the trained logistic regression model
"""


def train_logistic_regression(features, label):
    print("Training the logistic regression model...")
    from sklearn.linear_model import LogisticRegression
    ml_model = LogisticRegression(C=100, random_state=0, solver='liblinear')
    ml_model.fit(features, label)
    print('Finished training the model\n')
    return ml_model


print("Operating on training data...\n")
trainReviews = X_train.tolist()
cleanReviewData = clean_my_data(trainReviews)  # cleans the training data
vectorizer, train_tfidf_features, tfidf = (create_bag_of_words(cleanReviewData))
# stores the sparse matrix of the tf-idf weighted features

ml_model = train_logistic_regression(train_tfidf_features, y_train)
# holds the trained model

print("Operating on test data...\n")
testReviews = X_test.tolist()
cleanTestData = clean_my_data(testReviews)
# cleans the test data for accuracy evaluation

test_data_features = vectorizer.transform(cleanTestData)
# vectorizes the test data using the mean and variance from the training data
test_data_features = test_data_features.toarray()

test_tfidf_features = tfidf.transform(test_data_features)
# tfidf transform of the vectorized text data using the previous mean and variance
test_tfidf_features = test_tfidf_features.toarray()

predicted_y = ml_model.predict(test_tfidf_features)
# uses the trained logistic regression model to assign sentiment to each
# test data example

correctly_identified_y = predicted_y == y_test
accuracy = np.mean(correctly_identified_y) * 100
print('The accuracy of the model in predicting movie review sentiment is %.0f%%' % accuracy)
# compares the predicted sentiment (predicted_y) vs the actual
# sentiment stored in y_test

