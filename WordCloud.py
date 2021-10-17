# Import packages
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from wordcloud import WordCloud
from textblob import TextBlob

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

# ***********************************************************************************
# Part B: Preliminary Analysis
# ***********************************************************************************

# Subset out tweet data
tweetText = list(twitterDf['tweet_text'])

# Split out words from tweets
withStops = []
withoutStops = []
tweetsWithPolarity = []
list_for_polarity = []
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

    # Get the polarity and subjectivity scores for the tweet
    tweetSentiment = TextBlob(tweet).sentiment
    list_for_polarity.append(tweetSentiment.polarity)
    list_for_subjectivity.append(tweetSentiment.subjectivity)

    # Associate each tweet text with its polarity in a dict, append to list
    currTweetPolSubj = {'tweet_text': tweet,
                        'pol': tweetSentiment.polarity}

    tweetsWithPolarity.append(currTweetPolSubj)

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
# who is the most frequently tweeting person (tweet author) on the keyword in the collected data?
# For this question, please use user_screen_name column.

MostVocal = list(twitterDf['user_screen_name'])
user_counter = Counter(MostVocal)
print(user_counter.most_common(1))

# Subset retweets only
retweetsOnly = twitterDf.loc[twitterDf['is_retweet'] == True, :].copy()

# 5. Who is the most influential user? A user’s influence score is the sum of “source_user_followers_count”, “source_user_friends_count”,
# “source_user_listed_count”, “source_user_favourites_count”.
retweetsOnly.loc[:,'user_influence_score'] = retweetsOnly['source_user_followers_count'] + retweetsOnly['source_user_friends_count'] + retweetsOnly['source_user_listed_count'] + retweetsOnly['source_user_favourites_count']
maxUserInfluenceScore = retweetsOnly['user_influence_score'].max()

# Get user with max influence score
maxUser = retweetsOnly.loc[retweetsOnly['user_influence_score'] == maxUserInfluenceScore, :]
print('The Twitter user with maximum influence is: ' + str(maxUser['user_screen_name'].iloc[0]))

# 6. Which is the most influential retweet? A tweet’s influence score is the sum of “source_tweet_quote_count”, “source_tweet_reply_count”, 
# “source_tweet_retweet_count”, “source_tweet_favorite_count”
retweetsOnly.loc[:,'retweet_influence_score'] = retweetsOnly['source_tweet_quote_count'] + retweetsOnly['source_tweet_reply_count'] + retweetsOnly['source_tweet_retweet_count'] + retweetsOnly['source_tweet_favorite_count']
maxRetweetScore = retweetsOnly['retweet_influence_score'].max()

# Get tweet with max retweet score
maxRetweet = retweetsOnly.loc[retweetsOnly['retweet_influence_score'] == maxRetweetScore,:]
print('The retweet with maximum influence is: "' + str(maxRetweet['tweet_text'].iloc[0]) + '"')

# ***********************************************************************************
# Part C: Word Cloud
# ***********************************************************************************
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

# ***********************************************************************************
# Part D: Sentiment Analysis
# ***********************************************************************************
# 1. What are the average polarity and subjectivity scores?

print(f'Average polarity score: {np.mean(list_for_polarity)!r}')
print(f'Average subjectivity score: {np.mean(list_for_subjectivity)!r}')

# 2. Visualize the polarity and subjectivity score distributions using histograms,
# where x-axis is the score and y-axis is the tweet count in the score bin.
# In total, there should be 2 histograms for this task.

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

# sort the tweets based on polarity
tweetsWithPolarity.sort(key=getPolarities)
numTweets = len(tweetText)

if numTweets > 0:
    # helper function: if there's only one most positve or negatie tweet, pick that one.
    # Otherwise, there are multiple most positve or negative tweets--pick the 2 tweets among them.
    def tweetsToPick(startingIndex):
        mostPosOrNegTweet = tweetsWithPolarity[startingIndex]

        # if startingIndex == 0, then get the next most positive tweet at index 1
        # otherwise starting index is -1 and get the next most negative tweet at index -2
        nextTweet = tweetsWithPolarity[1 if startingIndex == 0 else -2]

        return 2 if  numTweets > 1 and mostPosOrNegTweet['pol'] == nextTweet['pol'] else 1

    # display the most negative tweet(s)
    negTweetsPicked = tweetsToPick(0)
    print(f'Most negative tweets: {tweetsWithPolarity[:negTweetsPicked]!r}')

    # display the most positive tweet(s)
    posTweetsPicked = tweetsToPick(-1)
    print(f'Most positive tweets: {tweetsWithPolarity[-posTweetsPicked:]!r}')
