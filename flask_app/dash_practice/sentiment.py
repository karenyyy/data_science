from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analysis = TextBlob('TextBlob sure looks like it has some interesting features!')

print(dir(analysis))
print(analysis.sentiment)
print(analysis.tags)
print(analysis.translate(to='zh'))

text = '''
Hong Kong was formerly a colony of the British Empire, after Qing China ceded Hong Kong Island at the conclusion of the First Opium War in 1842.
The colony expanded to the Kowloon Peninsula in 1860 after the Second Opium War and was further extended when Britain obtained a 99-year lease of the New Territories in 1898. 
The entire territory was returned to China when this lease expired in 1997. 
As a special administrative region, Hong Kong's system of government is separate from that in mainland China.
Originally a lightly populated area of farming and fishing villages, the territory has become one of the most significant financial centres and trade ports in the world. 
It is the world's seventh-largest trading entity an
d its legal tender, the Hong Kong dollar, is the 13th-most traded currency.
Although the city boasts one of the highest per capita incomes in the world, it suffers severe income inequality.
'''

print(TextBlob(text=text).sentiment)

analyzer = SentimentIntensityAnalyzer()

threshold = 0.5

pos_count = 0
pos_correct = 0

with open("data/positive.txt", mode="r", encoding='latin') as f:
    for line in f.read().split('\n'):
        # analysis = TextBlob(line)
        # if analysis.sentiment.polarity > 0:
        score = analyzer.polarity_scores(line)
        if not score['neg'] > 0.1:
            if score['pos'] - score['neg'] >= 0:
                pos_correct += 1
            pos_count += 1

neg_count = 0
neg_correct = 0

with open("data/negative.txt", mode="r", encoding='latin') as f:
    for line in f.read().split('\n'):
        # analysis = TextBlob(line)
        # if analysis.sentiment.polarity <= 0:
        score = analyzer.polarity_scores(line)
        if not score['pos'] > 0.1:
            if score['pos'] - score['neg'] <= 0:
                neg_correct += 1
            neg_count += 1

print("Positive accuracy = {}% via {} samples".format(pos_correct / pos_count * 100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct / neg_count * 100.0, neg_count))
