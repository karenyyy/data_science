import sqlite3
import pandas as pd

conn = sqlite3.connect('../twitter.db')
df = pd.read_sql("select * from sentiment limit 10", conn)
print(df.head())
df.sort_values('unix', inplace=True)

df['sentiment'] = df['sentiment'].rolling(int(len(df)/5)).mean()

df.dropna(inplace=True)

print(df.tail())