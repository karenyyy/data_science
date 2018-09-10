# A pipeline for gathering tweets, removing non-ASCII characters, and saving to CSV file

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

import sys
import os
import json
import time
import datetime
import re

import pandas as pd



# Define the listener
class MyListener(StreamListener):
    def __init__(self, raw_file, csv_file, text_file, max_num=300):
        super().__init__()
        self.raw_file = raw_file
        self.csv_file = csv_file
        self.text_file = text_file
        self.max_num = max_num
        self.count = 0
        self.start_time = time.time()

    def on_data(self, data):
        # Filter out special cases
        if data.startswith('{"limit":'):
            return

        # Filter out non-English tweets
        tweet = json.loads(data)
        if tweet['lang'] != 'en':
            return
        # if 'retweeted_status' in tweet:
        #     return

        # Extract fields from tweet and write to csv_file
        user_id = tweet['user']['id']
        user_name = tweet['user']['name']
        tweet_time = tweet['created_at']
        location = tweet['user']['location']
        text = tweet['text'].strip().replace('\n', ' ').replace('\t', ' ')

        # Remove non-ASCII characters and commas in user_name and location
        if user_name is not None:
            user_name = ''.join([c if ord(c) < 128 else '' for c in user_name])
            user_name = user_name.replace(',', '')
        if location is not None:
            location = ''.join([c if ord(c) < 128 else '' for c in location])
            location = location.replace(',', '')

        # Remove non-ASCII characters in text
        text = ''.join([c if ord(c) < 128 else '' for c in text])
        # Replace commas with space
        text = text.replace(',', ' ')
        # Replace double quotes with blanks
        text = re.sub(r'\"', '', text)
        # Replace consecutive underscores with space
        text = re.sub(r'[_]{2,}', ' ', text)
        # Remove all consecutive whitespace characters
        text = ' '.join(text.split())

        # Check if csv_file, text_file exist
        # If not, create them and write the heads
        if not os.path.isfile(self.csv_file):
            with open(self.csv_file, 'w') as f:
                f.write(','.join(['user_id', 'user_name', 'tweet_time', 'location', 'text']) + '\n')
        if not os.path.isfile(self.text_file):
            with open(self.text_file, 'w') as f:
                f.write('text\n')

        with open(self.raw_file, 'a') as f_raw, open(self.csv_file, 'a') as f_csv, open(self.text_file, 'a') as f_text:
            # Write to files
            f_raw.write(data.strip() + '\n')
            f_csv.write(','.join(map(str, [user_id, user_name, tweet_time, location, text])) + '\n')
            f_text.write(text + '\n')

            # Increment count
            self.count += 1
            # if self.count % 10 == 0 and self.count > 0:
            sys.stdout.write('\r{}/{} tweets downloaded'.format(self.count, self.max_num))
            sys.stdout.flush()

            # Check if reaches the maximum tweets number limit
            if self.count == self.max_num:
                print('\nMaximum number reached.')
                end_time = time.time()
                elapse = end_time - self.start_time
                print('It took {} seconds to download {} tweets'.format(elapse, self.max_num))
                sys.exit(0)

    def on_error(self, status):
        print(status)
        return True


# Get the str representation of the current date and time
def current_datetime_str():
    # return f'{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# Main
def main():
    # Key and token info needed
    consumer_key = "kW6FPW7k8IO3EnPXpWAjnuXKo"
    consumer_secret = "ZamoT9QGgCm0PyEJPeZ81F60tYDzBnDOUMbuxv5ExOQP720Rfp"
    access_token = "1006940361640603648-j6AIjLx2gRiV1S6GKlK6M58J2U4lsi"
    access_secret = "WdJUoc5VLXCsjR4x4nA8lKKjPTbOQXU2KHP6q1FO8m2Wm"

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)

    # Welcome
    print('===========================================================')
    print('Welcome to the user interface of gathering tweets pipeline!')
    print('You can press "Ctrl+C" at anytime to abort the program.')
    print('===========================================================')
    print()

    # Prompt for input keywords
    methods = ['manual', 'file']
    print('How do you want to specify your key words?')
    while True:
        m = input('Type "manual" or "file" >>> ')
        if m in methods:
            break
        else:
            print('\"{}\" is an invalid input! Please try again.\n'.format(m))

    # Choose keywords:
    if m == 'file':
        print('===========================================================')
        print('Please input the file name that contains your key words.')
        print('Notes:')
        print('    The file should contain key words in one or multiple lines, and multiple key words should be separated by *COMMA*.')
        print('        For example: NBA, basketball, Lebron James')
        print('    If the file is under the current directory, you can directly type the file name, e.g., "keywords.txt".')
        print('    If the file is in another directory, please type the full file name, e.g., "C:\\Downloads\\keywords.txt" (for Windows), or "/Users/xy/Downloads/keywords.txt" (for MacOS/Linux).')

        while True:
            file_name = input('Type your file name >>> ')
            if os.path.isfile(file_name):
                break
            else:
                print('"{}" is not a valid file name! Please check if the file exists.\n'.format(file_name))

        # Check the content of keywords file
        key_words = []
        with open(file_name, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                print('\n{} is an empty file!\nTask aborted!'.format(file_name))
                sys.exit(1)

            for line in lines:
                line = line.strip()
                # Detect non-ASCII characters
                for c in line:
                    if ord(c) >= 128:
                        print('\n{} contains non-ASCII characters: "{}" \nPlease remove them and try again'.format(file_name, c))
                        sys.exit(1)
                # Check delimiters
                if line.count(' ') > 1 and ',' not in line:
                    print('\nMore than 1 <space> symbols exist in the key words file, but none comma exists')
                    print('I\'m confused about your keywords. Please separate your key words by commas.')
                    sys.exit(1)

                words = line.split(',')
                for w in words:
                    if len(w.strip()) > 0:
                        key_words.append(w.strip())

        # Check key_words
        if len(key_words) == 0:
            print('\nZero key words are found in {}! Please check your key words file.'.format(file_name))
            sys.exit(1)

    elif m == 'manual':
        print('===========================================================')
        print('Please input your key words (separated by comma), and hit <ENTER> when done.')

        while True:
            line = input('Type the key words >>> ')
            line = line.strip()

            invalid_flag = False
            # Check empty
            if len(line) == 0:
                print('\nYour input is empty! Please try again.')
                invalid_flag = True
            # Detect non-ASCII characters
            for c in line:
                if ord(c) >= 128:
                    print('\nYour input contains non-ASCII characters: "{}"! Please try again.'.format(c))
                    invalid_flag = True
                    break
            # Check delimiters
            if line.count(' ') > 1 and ',' not in line:
                print('\nMore than 1 <space> symbols exist in your input, but none comma exists')
                print('I\'m confused about your keywords. Please try again')
                invalid_flag = True

            if invalid_flag:
                continue
            else:
                break

        # Process input
        key_words = []
        for w in line.split(','):
            if len(w.strip()) > 0:
                key_words.append(w.strip())

    # Print valid key words
    key_words = list(set(key_words))
    print('\n{} unique key words being used: '.format(len(key_words)), key_words)

    # Prompt for number of tweets to be gathered
    print('===========================================================')
    print('How many tweets do you want to gather? \nInput an integer number, or just hit <ENTER> to use the default number 300.')
    num_tweets = 300
    while True:
        s = input('Input an integer >>> ')
        s = s.strip()
        if len(s) == 0:
            break
        elif s.isdigit():
            num = int(s)
            if num > 0:
                num_tweets = num
                break
            else:
                print('\nPlease input a number that is greater than 0.')
        else:
            print('\nPlease input a valid integer number.')

    print('{} tweets to be gathered.'.format(num_tweets))

    # Streaming
    # TODO: remvoe '\t', '\n' and ',' in text field, also remove empty text
    print('===========================================================')
    print('Start gathering tweets ...')

    postfix = current_datetime_str()
    raw_file = 'raw_{}.json'.format(postfix)
    csv_file = 'data_{}.csv'.format(postfix)
    text_file = 'text_{}.csv'.format(postfix)

    twitter_stream = Stream(auth, MyListener(raw_file=raw_file, csv_file=csv_file, text_file=text_file, max_num=num_tweets))
    twitter_stream.filter(track=key_words)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nTask aborted!')
