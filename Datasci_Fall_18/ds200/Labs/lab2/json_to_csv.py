# Convert json data to csv data

import pandas as pd
import sys
import json


# Main
if __name__ == '__main__':
    # Parse arguments
    if len(sys.argv) != 2:
        print('Please provide one input file')
    else:
        input_file = sys.argv[1]
        output_file = input_file.split('.')[0] + '.csv'

        data = []
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                tweet = json.loads(line)

                # skip over non-English tweets
                if tweet['lang'] != 'en':
                    continue

                d = {}
                d['user_id'] = tweet['user']['id']
                d['user_name'] = tweet['user']['name']
                d['time'] = tweet['created_at']
                d['location'] = tweet['user']['location']
                d['text'] = tweet['text']
                data.append(d)

        df = pd.DataFrame(data, columns=data[0].keys())
        df.to_csv(output_file, index=False, sep='\t', encoding='utf-8')
