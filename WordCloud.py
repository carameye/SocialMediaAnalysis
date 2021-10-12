# Import packages
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from wordcloud import WordCloud
from textblob import TextBlob
from nltk.translate.meteor_score import wordnetsyn_match

nltk.download('stopwords')
# Get the English stopwords
stopwords = nltk.corpus.stopwords.words('english')
customStops = ['rt', 'https']

# Define ustom comparator function to filter out stop words (from nltk and from custom list)
# and to filter out garbage chars
def notStopWord(word):
    return len(word) > 1 and word not in stopwords and not any([word.startswith(c) for c in customStops])

def wordsToString(words):
    return ' '.join(words)

def getPolarities(p):
    return p['pol']

# Import tweets from csv file
twitterDf = pd.read_csv('tweets_vaccine_5000.csv', index_col=0)

# Part B: Preliminary Analysis
# 0. Data Cleaning/Processing Stage
    # Subset out tweet data
tweetText = list(twitterDf['tweet_text'])
    # Split out words from tweets
withStops = []
withoutStops = []
list_for_polarity = []
tweetPolSubj = []
list_for_subjectivity = []

for tweet in tweetText:
    # Split tweet into words
    splittedTweet = tweet.lower().split()
    # Append onto withStops
    withStops += splittedTweet
    # Filter out stops
    stopFilteredTweet = list(filter(notStopWord, splittedTweet))
    # Append stopped tweets to withoutStop array
    withoutStops += stopFilteredTweet
    # Polarity
    pol = TextBlob(tweet)
    currPol = pol.sentiment.polarity
    list_for_polarity.append(currPol)

    # Subjectivity
    sub=TextBlob(tweet)
    currSubjectivity = sub.sentiment.subjectivity
    list_for_subjectivity.append(currSubjectivity)

    # Append current sentimentality, polarity and tweet text
    currTweetPolSubj = {'tweet_text': tweet,
                        'pol': currPol,
                        'subj': currSubjectivity}
    tweetPolSubj.append(currTweetPolSubj)

# 1. What are the ten most popular words with and without stop words?
    # Create chart of word frequencies (with stops)
withStopsFreqs = nltk.FreqDist(withStops)
withStopsFreqs.plot(10)

    # Word Frequency with Counter
withStopsCounts = Counter(withStops)
print(withStopsCounts.most_common(10))
    # Create chart of word frequencies (without stops)
withoutStopsFreqs = nltk.FreqDist(withoutStops)
withoutStopsFreqs.plot(10)
    # Print withoutstops frequencies
withoutStopsCounts = Counter(withoutStops)
print(withoutStopsCounts.most_common(10))

# 2. What are the ten most popular hashtags (#hashtag)?
hashtagsList = []
    # Extract hashtags only
for word in withStops:
    if '#' in word:
        hashtagsList.append(word)
    # Create chart of hashtage frequencies
hashtagsFreqs = nltk.FreqDist(hashtagsList)
hashtagsFreqs.plot(10)
    # Print hashtag frequencies
hashtagCounts = Counter(hashtagsList)
print(hashtagCounts.most_common(10))

# 3. What are the ten most frequently mentioned usernames (@username)?
usernamesList = []
for word in withStops:
    if '@' in word:
        usernamesList.append(word)
usernamesFreq = nltk.FreqDist(usernamesList)
print(usernamesFreq)
    # Print usernames frequencies
usernameCounts = Counter(usernamesList)
print(usernameCounts.most_common(10))

# 4. Who is the vocal user on the keyword? In other words,
#    who is the most frequently tweeting person (tweet author) on the keyword in the collected data?
#    For this question, please use user_screen_name column.

MostVocal = list(twitterDf['user_screen_name'])
user_counter = Counter(MostVocal)
print(user_counter.most_common(1))

# A string of all the tweets (without stopwords) to be used in the wordcloud
tweetsString = ' '.join(withoutStops)

# Make wordcloud from the tweetsString
wordcloud = WordCloud(width=800, height=400).generate(tweetsString)
wordcloudFileName = 'BA515_Team24_wordcloud'

# Display the generated wordcloud image:
plt.figure(figsize=(12,6)) # set up figure size
plt.imshow(wordcloud) # show the wordcloud image
plt.axis('off') # turn off axis
plt.savefig(wordcloudFileName + '.png') # save as PNG file
plt.savefig(wordcloudFileName + '.pdf') # save as PDF file
plt.show() # show wordcloud

# Part 4
# 1. What are the average polarity and subjectivity scores?

np.mean(list_for_polarity)
np.mean(list_for_subjectivity)

# 2. Visualize the polarity and subjectivity score distributions using histograms,
#    where X-axis is the score and Y-axis is the tweet count in the score bin.
#    In total, there should be 2 histograms for this task.

plt.hist(list_for_polarity, bins = 10)
plt.grid(True)
plt.xlabel('Polarity Score')
plt.ylabel('Tweet Count')
plt.savefig('polarity.pdf')
plt.show()

plt.hist(list_for_subjectivity, bins = 10)
plt.grid(True)
plt.xlabel('Subjectivity Score')
plt.ylabel('Tweet Count')
plt.savefig('subjectivity.pdf')
plt.show()

# 3. Based on the polarity scores, what are the most positive and negative tweets on the keyword?
#    Why is the author happy/angry on the topic? If there are multiple tweets, please pick 2-3 tweets among them.

# number of pos and neg tweets we are picking--pick the top 2 tweets if we have more than one, otherwise pick 1
numTweets = len(tweetText)
tweetsPicked = 2 if numTweets > 1 else 1

tweetPolSubj.sort(key=getPolarities)
if numTweets > 0:
    print(f'Most negative tweets {tweetPolSubj!r}')