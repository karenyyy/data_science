from tweepy import Stream, OAuthHandler
from tweepy.streaming import StreamListener
import json
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from unidecode import unidecode

ckey = "kW6FPW7k8IO3EnPXpWAjnuXKo"
csecret = "ZamoT9QGgCm0PyEJPeZ81F60tYDzBnDOUMbuxv5ExOQP720Rfp"
atoken = "1006940361640603648-j6AIjLx2gRiV1S6GKlK6M58J2U4lsi"
asecret = "WdJUoc5VLXCsjR4x4nA8lKKjPTbOQXU2KHP6q1FO8m2Wm"


analyzer = SentimentIntensityAnalyzer()

conn = sqlite3.connect(database='twitter.db')
c = conn.cursor()


def create_table():
    try:
        c.execute("drop table sentiment")
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
        try:
            data = json.loads(raw_data)
            tweet = unidecode(data['text'])
            time_ms = data['timestamp_ms']
            vs = analyzer.polarity_scores(tweet)
            sentiment = vs['compound']
            c.execute("insert into sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)",
                       (time_ms, tweet, sentiment))
            conn.commit()
        except KeyError as e:
            print(str(e))

    def on_error(self, status_code):
        print(status_code)


while True:
    try:
        print("yes")
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        twitterStream = Stream(auth, listener())
        result = twitterStream.filter(track=["a", "e", "i", "o", "u"])
        print(result)
    except Exception as e:
        print(str(e))


