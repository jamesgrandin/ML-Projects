import nltk
import random  # used to generate random responses
import string  # used to remove punctuation
import pandas as pd
import requests #
from bs4 import BeautifulSoup # used to parse html to output summary and financials
import textwrap3 as tw # used to make text fit the command window better
import webbrowser


### Initialize ###

with open("DowCompanies.txt", 'r') as myFile1:
    data1 = myFile1.read()

input_tokens = data1.split("@")

botID = "Warren BuffBot: "

initGreeting = ("Hello.  I am a Warren BuffBot.  I can give you a company summary and fundamental/technical analysis information for any company in the Dow Jones Index."
                   " Just type in the name or symbol for the company you want information on, followed by the desired prompt."
                   " For example: Type MSFT followed by \'Financials\' or \'Summary\' to get the desired information for Microsoft."
                   " If you want a list of ticker symbols for Dow Jones companies, type \'Tickers\'."
                   " Which Dow Jones company would you like to learn more about?")

initialGreeting = tw.fill(initGreeting,70) #makes the maximum character length per line displayed in console 70 so that the text is easier to read

confusedResponse = "I'm not sure what you are looking for, please try again."


greetings = ["hello", "hi", "greetings", "sup", "whats up", "hey", "howdy"]
greetingResponses = ([" Hi, let me know what company and info you want and i'll get right on it.",
                      " Hello. If you need a list of companies to choose from, type \'Tickers\' in the prompt.",])


thanks = ["thanks", "thank you", "cool", "awesome"]
welcomeResponse = "You are welcome. Goodbye!"


goodbyes = ["bye", "goodbye", "later", "lates", "cya", "cyas", "peace"]
goodbyeResponse = "Remember, intelligent investors are greedy when others are fearful. Take care and stay safe!"


list_of_tickers = pd.read_csv('tickersymbols.csv', index_col=0)

lemmer = nltk.stem.WordNetLemmatizer()  # used to consolidate different word forms

### Define Functions ###

# returns cleaned list of consolidated tokens
def lemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


# different method for removing non-alphanumeric characters
def lemNormalize(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# checks to see if the input text matches one of the greeting_inputs.  If so,
# return one of the random greeting_responses.
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greetings:
            return random.choice(greetingResponses)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def response(user_response):
    bot_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=lemNormalize)
    tfidf = TfidfVec.fit_transform(input_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    with open(outputFilename, 'r') as myFile2:
        data2 = myFile2.read()
    out_text = data2.split("@")

    if (req_tfidf == 0):
        bot_response = botID + confusedResponse
        flag2 = 2
        return bot_response, flag2
    else:
        bot_response = bot_response + out_text[idx]
        flag2 =1

        return bot_response, flag2


def summary_scraper(sumurl): #parses the input html and scrapes relevent body text, prints it to the console
    def getdata(url):
        r = requests.get(url)
        return r.text

    htmldata = getdata(sumurl)
    soup = BeautifulSoup(htmldata, 'html.parser')
    summary = soup.find('p',"TextLabel__text-label___3oCVw TextLabel__black___2FN-Z TextLabel__serif___3lOpX Profile-body-2Aarn").get_text()
    return print(tw.fill(summary, 70))


### Start of Program ###

flag = True
print("\n\n" + botID + initialGreeting)



while (flag == True):

    user_response = input(">>> ")

    if "financials" in user_response.lower():

        outputFilename = "financials.txt"
        input_tokens.append(user_response)
        print(botID, end="")
        a = response(user_response)
        input_tokens.remove(user_response)

        if a[1] == 1:
            webbrowser.get('firefox') #opens web browser for the technical/fundamental analysis info
            webbrowser.open_new(a[0].strip())
        else:
            print(a[0])

    elif "summary" in user_response.lower():
        outputFilename = "summary.txt"
        input_tokens.append(user_response)
        print(botID, end="")
        a = response(user_response)
        input_tokens.remove(user_response)

        if a[1] == 1:
            summary_scraper(a[0].strip())
        else:
            print(a[0])

    elif "tickers" in user_response.lower():
        print(list_of_tickers)

    else:

        user_response = user_response.lower()

        if user_response not in goodbyes:

            if user_response in thanks:
                flag = False
                print(botID + welcomeResponse)

            elif greeting(user_response) != None:
                print(botID + greeting(user_response))

            else:
                print(botID + confusedResponse)

        else:
            flag = False
            print(botID + goodbyeResponse)
