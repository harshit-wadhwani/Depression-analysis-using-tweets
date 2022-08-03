import streamlit as st
import snscrape.modules.twitter as sntwitter
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import keras
import numpy as np
import re
import stylecloud
import matplotlib.pyplot as plt
from PIL import Image


st.set_page_config(page_title='Twitter Sentiment Analysis', layout='wide', page_icon=":bird:")


@st.experimental_singleton
def retrive():
    path = "trained_model.h5"
    model = tf.keras.models.load_model(path, custom_objects={'KerasLayer':hub.KerasLayer})
    return model


model = retrive()



def predict(username):
    query = "(from:" + username + ")"
    tweets =[]
    limit = 100
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date , tweet.user.username , tweet.content])
    print("tweets extracted")        
    data = pd.DataFrame(tweets, columns=['date', 'username', 'content'])
    global unclean_tweets
    unclean_tweets = data['content'].values.tolist()
    stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
    def clean_tweet(tweet):
        if type(tweet) == np.float:
            return ""
        temp = tweet.lower()
        temp = re.sub("'", "", temp) # to avoid removing contractions in english
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        temp = re.sub("#[A-Za-z0-9_]+","", temp)
        temp = re.sub(r'http\S+', '', temp)
        temp = re.sub('[()!?]', ' ', temp)
        temp = re.sub('\[.*?\]',' ', temp)
        temp = re.sub("[^a-z0-9]"," ", temp)
        temp = temp.split()
        temp = [w for w in temp if not w in stopwords]
        temp = " ".join(word for word in temp)
        return temp
    results = [clean_tweet(tw) for tw in unclean_tweets]
    prediction = model.predict(results)
    avg = sum(prediction)/len(prediction)
    fl = avg[0]*100
    return round(fl, 2)




def word_cloud(username):
    query = "(from:" + username + ")"
    tweets =[]
    limit = 100
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date , tweet.user.username , tweet.content])     
    data = pd.DataFrame(tweets, columns=['date', 'username', 'content'])
    data['content'].to_csv("tweets.csv", index = False)
    stylecloud.gen_stylecloud(file_path='tweets.csv',
                          icon_name='fab fa-twitter',
                          palette='colorbrewer.qualitative.Paired_3', #https://jiffyclub.github.io/palettable/
                          background_color='white',
                          gradient='horizontal',
                          stopwords = True,
                          custom_stopwords=['RT','THE','IS','WITH','ON','THIS','HTTPS','CO','TO','AND','OF','IT','MY','FOR','IN','a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can',          'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', "haven't", 'having', 'he',                          'her', 'here',           'hers', 'herself', 'him', 'himself', 'his', 'how',          'i',                               'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she',                    "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll",           'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',            'these', 'they',                                            'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'we',                                    'were', 'weren', "weren't", 'what',           'when',           'where',            'which', 'while', 'who',          'whom', 'why', 'will',          'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
    )




st.header("Twitter Sentiment Analysis")
st.subheader("Enter you username : ")

username = st.text_input("username", max_chars=100)
if st.button("Predict 🔥"):
    url = "https://twitter.com/" + username + "/"

    st.subheader("@" + username + " is " +str(predict(username)) + "% depressed")
    st.success('Prediction Done !!! :tada:')
    word_cloud(username)
    image = Image.open('stylecloud.png')
    col1, col2 = st.columns(2)
    col1.header("Word Cloud")
    col1.image(image)
    tweets = pd.read_csv("tweets.csv")
    col2.header("Tweets")  
    col2.markdown("Last 100 tweets of "+f"[{username}]({url})" )
    col2.dataframe(tweets)