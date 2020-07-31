## TODO:

- Preform topic analysis on comments
  - https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc
- clean/organize code

# Usage

## Terminology

- `type` is used to refer to the data consisting of either comments or submissions and will take the value of `'comment'` or `'submission'`.
- 

## Help

`python download_data.py -h` outputs the following

```
usage: download_data.py [-h] [--subreddit SUBREDDIT] [--mode MODE] [--saveformat SAVEFORMAT] [--ncores NCORES] [--chunk_size CHUNK_SIZE] [--startdate STARTDATE] [--enddate ENDDATE]

Download data from Reddit in large quantities and quickly.

optional arguments:
  -h, --help            show this help message and exit
  --subreddit SUBREDDIT
                        The name of the subreddit to scrape.
  --mode MODE           The type of result to get. Should be either submission or comment.
  --saveformat SAVEFORMAT
                        The format to save the data in. Can be csv or pkl.
  --ncores NCORES       Number of concurrent processes.
  --chunk_size CHUNK_SIZE
                        Size of time chuncks to be processed at once
  --startdate STARTDATE
                        The start date for recorded results. Format is YYYY-MM-DD
  --enddate ENDDATE     The end date to cut results off at. Format is YYYY-MM-DD
```

## Downloading Data

To download comments and submissions for multiple subreddits at once (using the default settings), use `source download.sh subreddit1 [subreddit2 ...]`. Data is downloaded via [pushshift.io](https://pushshift.io/), so it is important to note that the update process for the data is somewhat long. In other words, for reliable counts in fields such as `score`, use older data.

Another important factor to note is that when data is downloaded, API requests are randomly routed through a proxy from `proxies.py`. The proxies come from [a free proxy list](http://free-proxy.cz/en/proxylist/country/all/http/ping/all). There is a function on the website that allows about 20 proxies to be copied to the clipboard at once and it is easy to add more to the list. The effect of this is that the pushshift API ratelimit is completely avoided so data can be downloaded much faster.

## Scripts

- `download_data.py` downloads one type of data into the `./data` drectory. Several commandline options can be specified.
- `donload.sh` will download both types of data into `./data` for as many subreddits as are specified at the time of running. It will run `download_data.py` with the default parameters.
- `process_comments.pu` will preform the following operations on all comment data in `./data` and save the processed data to the `./processed` directory:
  - [vaderSentiment](https://github.com/cjhutto/vaderSentiment) analysis.

## Plots

- Plots are generated in the `charts.ipynb` jupyter notebook, but in the future, some of them may be migrated to different scripts.

