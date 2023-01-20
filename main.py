pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib


import tweepy
import panda as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words= set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import CountVectorizer
from sklearn.linear_model import logisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusin_matrix, confusinMatrixDisplay


df=pd.read_csv("vaccination_tweets.csv")
df.head()
