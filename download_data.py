import argparse
import datetime
import os
from multiprocessing import Process, Queue
from time import sleep

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import praw

from helpers import *

# from mpl_toolkits.mplot3d import Axes3D


mpl.use('Agg')

pd.options.mode.chained_assignment = None


def Date(s): return datetime.datetime.strptime(s, '%Y-%m-%d')


if __name__ == '__main__':
    # ============================= PARSE ARGS =======================
    parser = argparse.ArgumentParser(
        description='Download data from Reddit in large quantities and quickly.')

    parser.add_argument('--subreddit', default='dataisbeautiful',
                        help='The name of the subreddit to scrape.')
    parser.add_argument('--mode', default='submission',
                        help='The type of result to get. Should be either submission or comment.')
    parser.add_argument('--saveformat', default='csv',
                        help='The format to save the data in. Can be csv or pkl.')

    parser.add_argument('--ncores', default=8,
                        help='Number of concurrent processes.', type=int)
    parser.add_argument('--chunk_size', default=datetime.timedelta(days=1).total_seconds(),
                        help='Size of time chuncks to be processed at once', type=int)

    parser.add_argument('--startdate', default=datetime.datetime(2018, 6, 1),
                        help='The start date for recorded results. Format is YYYY-MM-DD', type=Date)
    parser.add_argument('--enddate', default=datetime.datetime(2018, 7, 1),
                        help='The end date to cut results off at. Format is YYYY-MM-DD', type=Date)

    args = parser.parse_args()

    cfg_atts = {
        'subreddit': args.subreddit,
        'mode': args.mode,
        'saveformat': args.saveformat,
        'ncores': args.ncores,
        'chunk_size': args.chunk_size,
        'startdate': args.startdate,
        'enddate': args.enddate
    }
    # ============================= SETUP ============================
    cfg = Config(filename=cfg_atts['subreddit'], atts=cfg_atts)
    savefilename = f'data/{cfg.subreddit}_{cfg.mode}.{cfg.saveformat}'

    auth = read_auth('config')
    reddit = praw.Reddit(
        client_id=auth['client_id'],
        client_secret=auth['client_secret'],
        user_agent=auth['user_agent']
    )

    subreddit = reddit.subreddit(cfg.subreddit)
    target = [cfg.startdate.timestamp(), cfg.enddate.timestamp()]
    # Load data and get a UTC range for each process to download
    dataframes = []
    totalrange = target[1] - target[0]
    nbins = int(totalrange / cfg.chunk_size) + 1
    bounds = np.linspace(target[0], target[1], nbins)
    ranges = np.concatenate(
        [bounds[:-1, np.newaxis], bounds[1:, np.newaxis]],
        axis=-1
    )
    # ============================= CODE =============================
    # Start all processes
    queues = []
    dq = Queue(len(ranges))
    print('Getting data in chunks...')
    for r in ranges:
        s, e = [datetime.datetime.fromtimestamp(
            s).strftime('%Y-%m-%d') for s in r]
        print(f'\t{s} -> {e}')
        dq.put(r)
    procs = []
    for _ in range(cfg.ncores):
        q = Queue(5)
        queues.append(q)
        args = [dq, q, cfg.subreddit, cfg.mode]
        p = Process(target=get_range, args=args)
        p.start()
        procs.append(p)

    # While we wait
    closed = []
    totalrows = 0
    starttime = datetime.datetime.now()
    while len(closed) < len(queues):
        # Check the queues
        for i in range(len(queues)):
            if i in closed:
                continue
            q = queues[i]
            while not q.empty():
                res = q.get()
                if isinstance(res, str) and res == 'END':
                    closed.append(i)
                else:
                    totalrows += len(res)
                    done = 1 - (dq.qsize() / len(ranges))
                    rps = totalrows / \
                        (datetime.datetime.now() - starttime).total_seconds()
                    dataframes.append(res)
                    print(f'{int(100 * done)}% -- {totalrows} rows -- {int(rps)} rps')
        sleep(0.5)
    # Join subprocesses
    for p in procs:
        p.join()
    dq.close()
    for q in queues:
        q.close()
    # Merge dataframes and save
    if len(dataframes) == 0:
        print(f'No data for {cfg.subreddit} in the specified time range')
        exit()
    else:
        df = pd.concat(dataframes, copy=False)
    # df = df.loc[~df.index.duplicated(keep='first')]
    print(f'Saving {len(df)} rows to {savefilename}')
    if cfg.saveformat == 'pkl':
        df.to_pickle(savefilename)
    elif cfg.saveformat == 'csv':
        df.to_csv(savefilename)
    else:
        raise ValueError('Enter a supported save format.')
