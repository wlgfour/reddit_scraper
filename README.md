## TODO:

- Preform topic analysis on comments
  - https://towardsdatascience.com/the-complete-guide-for-topics-extraction-in-python-a6aaa6cedbbc

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

## Download

To download comments and submissions for multiple subreddits at once (using the default settings), use `source download.sh subreddit1 [subreddit2 ...]`.

## Scripts
- `download_data.py` downloads one type of data into the `./data` drectory. Several commandline options can be specified.
- `donload.sh` will download both types of data into `./data` for as many subreddits as are specified at the time of running. It will run `download_data.py` with the default parameters.
- `process_comments.pu` will preform the following operations on all comment data in `./data` and save the processed data to the `./processed` directory:
  - [vaderSentiment](https://github.com/cjhutto/vaderSentiment) analysis.
-