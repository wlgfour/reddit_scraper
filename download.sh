
if [ ! -d log ]; then
    mkdir log
fi

for var in "$@"
do
    if [ ! -f data/"$var"_submission.csv ]; then
        echo data/"$var"_submission.csv
        python download_data.py --subreddit $var > log/submission_$var &
    fi
    if [ ! -f data/"$var"_comment.csv ]; then
        echo data/"$var"_comment.csv
        python download_data.py --subreddit $var --mode comment > log/comment_$var &
    fi
done
