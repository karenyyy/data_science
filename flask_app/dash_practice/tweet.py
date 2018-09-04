import sqlite3
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
analyzer = SentimentIntensityAnalyzer()

conn = sqlite3.connect('tweet.db')
c = conn.cursor()
#
c.execute('CREATE TABLE IF NOT EXISTS sentiment(unix real, tweets text, sentiment real)')

c.execute('CREATE TABLE IF NOT EXISTS raw_tweets(user_id varchar,user_name varchar,tweet_time varchar,location varchar,text text)')

lines = c.execute('SELECT text FROM raw_tweets')
#
# c.execute('ALTER TABLE raw_tweets ADD polarity double')
# c.execute('ALTER TABLE raw_tweets ADD subjectivity double')

sens = []
for line in lines:
    s = TextBlob(line[0]).sentiment
    print('INSERT INTO raw_tweets (polarity) VALUES ({})'.format(s[0]))
    c.execute('INSERT INTO raw_tweets (polarity) VALUES ({})'.format(s[0]))
    print('INSERT INTO raw_tweets (subjectivity) VALUES ({})'.format(s[1]))
    sens.append(s)
    print(line, s)



raw = pd.read_sql_query("SELECT * from raw_tweets", conn)

complete = pd.concat([raw, pd.Series(sens, name='sentiment')], axis=1)
print(complete)

