import tweepy

consumer_key = 'T9519DblC28THZMuLTu1isFt6'
consumer_secret = 'qhikhDgfOtYZbT9tOcNriki5MBqNS4OuqjsP4rAHyPpbw2Nk5R'
access_token = '1210390927-55IrDjPoL1YOIyjLv4atS4goCvq1m3hSZICjNxW'
access_token_secret = 'Mb7t7y7xMbg5M8pRwhMfCaN74Jbg68mVPCD2QMKuAddSl'
    
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)
