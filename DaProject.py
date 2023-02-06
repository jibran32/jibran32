import snscrape.modules.twitter as tweetscraper
import pandas as pd
import re
import emoji
import nltk
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from transformers import pipeline
from textblob import TextBlob
nltk.download('words')
words = set(nltk.corpus.words.words())
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=False)
def Emotion_analysis(tweets):
    prediction=[]
    prediction += classifier(tweets)
    return prediction[0]['label']
def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.EMOJI_DATA) #Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
         if w.lower() in words or not w.isalpha())
    return tweet
def Sentiment_Analysis(tweet):
    res = TextBlob(tweet)
    if res.sentiment.polarity > 0:
        return 'Positive'
    elif res.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

a = input("which topic you are interested in ?")
print(f"{a} Twitter Topic Analysis")
query = f"{a} lang:en"
i=1
limit = 900
tweets=[]
for tweet in tweetscraper.TwitterSearchScraper(query).get_items():
    if i == limit:
        break
    else:
        print(i)
        i+=1
        tweets.append([tweet.content,tweet.retweetCount,tweet.replyCount,tweet.likeCount,tweet.quoteCount])
tweets_df = pd.DataFrame(tweets, columns=['Content', 'retweetCount', 'replyCount' ,'likeCount', 'quoteCount'])
tweets_df['Content'] = tweets_df['Content'].map(lambda x: cleaner(x))
tweets_df['SentimentScore'] = tweets_df['Content'].map(lambda x: Sentiment_Analysis(x))
tweets_df['Emotions'] = tweets_df['Content'].map(lambda y: Emotion_analysis(y))
comment_words = ''
stopwords = set(STOPWORDS)      
str = " "
for i in tweets_df['Content']:
    str+=i
    tokens = i.split()
    for j in range(len(tokens)):
        tokens[j] = tokens[j].lower()
    comment_words += " ".join(tokens)+" "
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
sentimentsofdata = ['neutral','positive','negative']
CoN=0
CoPv=0
CoNe=0
for i in tweets_df['SentimentScore']:
    if i == "Neutral":
        CoN+=1
    elif i == "Positive":
        CoPv +=1
    else:
        CoNe+=1
yvalues = [CoN,CoPv,CoNe]
fig = plt.figure(figsize = (10, 5))

plt.bar(
    sentimentsofdata,
         yvalues, 
         color ='maroon',
        width = 0.4)
plt.show()
CoAng=0
CoS=0
CoJ=0
CoL=0
CoF=0
CoSur=0
for i in tweets_df['Emotions']:
    if i == "anger":
        CoAng+=1
    elif i == "sadness":
        CoS +=1
    elif i == "joy":
        CoJ+=1
    elif i == "love":
        CoL+=1
    elif i == "fear":
        CoF+=1
    else:
        CoSur+=1
EmotionOfData = ['Anger','Sadness','Joy','Love','Fear','Surprise']
yvalues = [CoAng,CoS,CoJ,CoL,CoF,CoSur]
fig = plt.figure(figsize = (10, 5))

plt.bar(
    EmotionOfData,
         yvalues, 
         color ='green',
        width = 0.4)
plt.show()
tweets_df.to_csv("Extracted_Analyzed_tweets.csv");