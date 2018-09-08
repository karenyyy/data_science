from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from unidecode import unidecode
import time
from threading import Lock, Timer
import pandas as pd
from config import stop_words
import regex as re
from collections import Counter
import string
import pickle
import itertools
from textblob import TextBlob

analyzer = SentimentIntensityAnalyzer()

ckey = "X6FPrJfKXym2hAQSoWnRbJbH8"
csecret = "S1W5ORaZODdadPm1BcVRU2YLoKqWE0Tynd3JE9SFv709Ld4j38"
atoken = "1006940361640603648-WgncDwkYwcFZhg09hXJUOfbKm0PPrK"
asecret = "Sr162Rjc6dPbEkoz57oseYVJNLQU0XKtWudqUbZPeV1Zh"

conn = sqlite3.connect('twitter.db', isolation_level=None, check_same_thread=False)
c = conn.cursor()


def create_table():
    try:
        c.execute("PRAGMA journal_mode=wal")
        c.execute("PRAGMA wal_checkpoint=TRUNCATE")
        c.execute(
            "CREATE TABLE IF NOT EXISTS sentiment(id INTEGER PRIMARY KEY AUTOINCREMENT, unix INTEGER, tweet TEXT, sentiment REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS misc(key TEXT PRIMARY KEY, value TEXT)")
        c.execute("CREATE INDEX id_unix ON sentiment (id DESC, unix DESC)")
        c.execute(
            "CREATE VIRTUAL TABLE sentiment_fts USING fts5(tweet, content=sentiment, content_rowid=id, prefix=1, prefix=2, prefix=3)")
        c.execute("""
            CREATE TRIGGER sentiment_insert AFTER INSERT ON sentiment BEGIN
                INSERT INTO sentiment_fts(rowid, tweet) VALUES (new.id, new.tweet);
            END
        """)
    except Exception as e:
        print(str(e))


create_table()
lock = Lock()


class listener(StreamListener):
    data = []
    lock = None

    def __init__(self, lock):
        self.lock = lock
        self.save_in_database()
        super().__init__()

    def save_in_database(self):
        Timer(1, self.save_in_database).start()

        with self.lock:
            # here self.data as a iterator
            if len(self.data):
                c.execute('BEGIN TRANSACTION')
                try:
                    c.executemany("INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)", self.data)
                except:
                    pass
                c.execute('COMMIT')
                # flush cache
                self.data = []

    def on_data(self, data):
        try:
            data = json.loads(data) # data in json format
            if 'truncated' not in data:
                return True
            if data['truncated']:
                tweet = unidecode(data['extended_tweet']['full_text'])
            else:
                tweet = unidecode(data['text'])
            time_ms = data['timestamp_ms']
            print(tweet)
            vs = analyzer.polarity_scores(tweet)
            sentiment = vs['compound']

            with self.lock:
                self.data.append((time_ms, tweet, sentiment))

        except KeyError as e:

            print(str(e))
        return True

    def on_error(self, status):
        print(status)


stop_words.append('')
blacklist_counter = Counter(dict(zip(stop_words, [1000000] * len(stop_words))))

punctuation = [str(i) for i in string.punctuation]
split_regex = re.compile("[ \n" + re.escape("".join(punctuation)) + ']')


def map_nouns(col):
    return [word[0] for word in TextBlob(col).tags if word[1] == u'NNP']


def generate_trending():
    try:

        df = pd.read_sql("SELECT * FROM sentiment ORDER BY id DESC, unix DESC LIMIT 10000", conn)
        df['nouns'] = list(map(map_nouns, df['tweet']))

        tokens = split_regex.split(' '.join(list(itertools.chain.from_iterable(df['nouns'].values.tolist()))).lower())

        trending = (Counter(tokens) - blacklist_counter).most_common(10)

        trending_with_sentiment = {}
        for term, count in trending:
            df = pd.read_sql(
                "SELECT sentiment.* FROM  sentiment_fts fts LEFT JOIN sentiment ON fts.rowid = sentiment.id WHERE fts.sentiment_fts MATCH ? ORDER BY fts.rowid DESC LIMIT 1000",
                conn, params=(term,))
            trending_with_sentiment[term] = [df['sentiment'].mean(), count]

        with lock:
            c.execute('BEGIN TRANSACTION')
            try:
                c.execute("REPLACE INTO misc (key, value) VALUES ('trending', ?)",
                          (pickle.dumps(trending_with_sentiment),))
            except:
                pass
            c.execute('COMMIT')


    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')
    finally:
        Timer(5, generate_trending).start()


Timer(1, generate_trending).start()

while True:

    try:
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        twitterStream = Stream(auth, listener(lock))
        twitterStream.filter(track=["a", "e", "i", "o", "u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)
