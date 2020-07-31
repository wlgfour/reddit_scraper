""" Will process all files that match data/{sub}_comment.(csv|pkl) and output them to processed/{sub}_comment.csv
"""

import re
import os
import pandas as pd
from time import sleep
from helpers import apply_vader
from multiprocessing import Process, Queue

p = re.compile(r'^([a-zA-Z0-9_]+)_comment\.(csv|pkl)$')
data = {}
print('Reading data...')
for file in os.listdir('data'):
    m = p.match(file)
    if m is None:
        continue
    sub, extension = [m.group(i) for i in [1, 2]]
    
    print(f'\t{sub}')
    if extension == 'csv':
        df = pd.read_csv(os.path.join('data', file))
    elif extension == 'pkl':
        df = pd.read_pickle(os.path.join('data', file))
    else:
        continue
    data[sub] = df



# Add sentiment analysis
procs = {}
for sub in data:
    q = Queue(1)
    df = data[sub]['body'].astype(str)
    p = Process(target=apply_vader, args=[q, df])
    p.start()
    procs[sub] = (q, p)


done = []
dfs = {}
i = 0
print('\nProcessing sentiment scores...')
while len(done) < len(procs):
    for sub in procs:  # get an object from each queue once
        q, p = procs[sub]
        if sub in done:
            continue
        if not q.empty():
            df = q.get()
            if df is not None:
                Df = data[sub]
                Df['sentiment'] = df
                dfs[sub] = Df
            q.close()
            p.join()
            done.append(sub)
            i += 1
            print(f'\t[{i}/{len(procs)}] {sub}')
    sleep(1.5)


# Write the data
print('\nWriting Data...')
if not os.path.isdir('processed'):
    os.mkdir('processed')
for sub in data:
    filename = os.path.join('processed', f'{sub}_comment.csv')
    print(f'\t{filename}')
    data[sub].to_csv(filename)