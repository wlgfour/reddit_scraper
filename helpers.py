import json
import math
import os
import random
import time
import re

import numpy as np
import pandas as pd
import requests

from proxies import PROXIES

fields = {
    'submission': ['id', 'author', 'created_utc', 'permalink', 'score', 'upvote_ratio', 'num_comments'],
    'comment': ['id', 'author', 'created_utc', 'permalink', 'score', 'body'],
    'subreddit': []
}


def read_auth(path='config'):
    return json.load(open(path))


class Config:
    def __init__(self, filename='cfg', atts=None):
        self.filename = filename
        if atts is not None:
            self.atts = atts
        elif os.path.isfile(filename):
            self.atts = json.load(open(filename))
        else:
            raise ValueError(
                'Must supply Config with either a filename that points'
                ' to an existing configuration or atts dict.'
            )

    def write(self):
        json.dump(self.atts, open(self.filename, 'w'))

    def __getattr__(self, name):
        if name != 'atts' and name in self.atts:
            return self.atts[name]


def get_data(sub, after, before, mode, use_proxy=True):
    """
    """
    before, after = int(before), int(after)
    url = f'http://api.pushshift.io/reddit/search/{mode}/?size=1000&' + \
          f'after={after}&before={before}&subreddit={sub}&' + \
          f'fields={",".join(fields[mode])}&' + \
          f'sort=desc&sort_type=created_utc'
    i = random.randint(0, len(PROXIES) - 1)
    p = {'http': PROXIES[i]}
    # Get the data and try again if fail
    try:
        r = requests.get(url, proxies=p, timeout=5)
        status = r.status_code
    except Exception:
        status = -1
    if status != 200:
        time.sleep(np.random.rand() * 2.5)
        return get_data(sub, after, before, mode=mode)
    # Parse the data and error if fail
    try:
        data = json.loads(r.text)['data']
    except Exception as e:
        print(r)
        raise e
    df = pd.DataFrame(data, columns=fields[mode])
    df = df.set_index('id')
    return df


def get_range(dq, q, sub, mode):
    # Decrease before to the min of utc returned by each query
    while not dq.empty():
        after, before = dq.get()
        while before > after:
            try:
                df = get_data(sub, after, before, mode=mode)
            except Exception as e:
                print(f'{before} -> {after} failure')
                q.put('END')
                raise e
            before = df['created_utc'].min()
            if len(df) == 0:
                break
            q.put(df)
    q.put('END')


class DumbQueue:
    def __init__(self):
        self.data = []

    def put(self, obj):
        self.data.append(data)

    def get(self):
        return self.data.pop(0)

    def empty(self):
        return len(self.data) == 0

# ============ SENTIMENT analysis helpers =====================


class VaderSentiment:
    """ Predict fine-grained sentiment classes using Vader.
        https://towardsdatascience.com/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4
    """

    def __init__(self, model_file: str = None) -> None:
        super().__init__()
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        return self.vader.polarity_scores(text)['compound']


def clean_data(df, col='body', rcol='body_', lower=False):
    pass


def by_word(q, sub, df, col='body', rcol='body_'):
    """ Separates dataframe into words and computes a count and score for each word
        alongw ith a total score
    """
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    m = re.compile(r'[^\w\s_]+')

    # Clean the strings -- applied row-wise
    def clean(s):
        def rm_stop(words): return ' '.join(word.lower()
                                            for word in words.split() if word.lower() not in stop)
        try:
            s = re.sub(m, '', s).strip()
            return rm_stop(s)
        except Exception:
            return ''
    # Count words in dataframe
    def count_words(df): return df.body_.str.split().explode().value_counts()
    
    # Execute it all
    try:
        df[rcol] = df[col].apply(clean)
        q.put((sub, count_words(df)))
    except Exception as e:
        print(f'Couldn\'t process df for {sub}')
        raise e
