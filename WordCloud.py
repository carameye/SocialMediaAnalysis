# Import packages
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')

# Import tweets
twitterDf = pd.read_csv('tweets_vaccine_5000.csv')

# Part B: Preliminary Analysis
tweetText = twitterDf['tweet_text']

# 1. What are the ten most popular words with and without stop words?
# Code inspired from: https://stackoverflow.com/questions/40206249/count-of-most-popular-words-in-a-pandas-dataframe
#       - With stopwords:
withStops = nltk.FreqDist(tweetText)
print(withStops)


#       - Without stop words
stopwords = nltk.corpus.stopwords.words('english')

def notStopWord(word):
    return word not in stopwords

# Filter out words that are not stop word
withoutStops = filter(notStopWord, withStops)

# 2. What are the ten most popular hashtags (#hashtag)?

# 3. What are the ten most frequently mentioned usernames (@username)?

# 4. Who is the vocal user on the keyword? In other words, who is the most frequently tweeting person (tweet author) on the keyword in the collected data? For this question, please use user_screen_name column.






