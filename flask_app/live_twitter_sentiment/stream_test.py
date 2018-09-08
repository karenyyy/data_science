from tweepy import Stream, OAuthHandler
from tweepy.streaming import StreamListener
import json
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from unidecode import unidecode

ckey = "tBvgBjQu5YYCqnnjKNqBsbTPd"
csecret = "MSyUCjkatYUbVPnr8P1mfd5FL2OwZiOmAl0lGtqmxY1M4zKgrm"
atoken = "1006940361640603648-7qZJNOBF7RlHs9xdfGj7OlYYZvUcDZ"
asecret = "bJmuvITm2GmMZIn7IuywONij14pQa6PvuUkGojdKdEQTJ"


analyzer = SentimentIntensityAnalyzer()

conn = sqlite3.connect(database='twitter.db')
c = conn.cursor()


def create_table():
    try:
        c.execute("create table if not exists sentiment(unix real, tweet text, sentiment real)")
        c.execute("create index fast_unix on sentiment(unix)")
        c.execute("create index fast_tweet on sentiment(tweet)")
        c.execute("create index fast_sentiment on sentiment(sentiment)")
        conn.commit()
    except Exception as e:
        print(str(e))

create_table()


class listener(StreamListener):
    def on_data(self, raw_data):
        print(raw_data)
        return True

    def on_error(self, status_code):
        print(status_code)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitter_stream = Stream(auth, listener())

output_file = 'nlp_key_words_result.csv'
with open(file=output_file, mode='w') as f:
    f.write(twitter_stream.filter(track=["nlp"]))


