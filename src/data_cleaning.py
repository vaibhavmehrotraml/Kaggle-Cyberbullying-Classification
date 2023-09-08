"""
File to preprocess & clean training data for a model.
Arguments are fed in with argparser.
"""
import pandas as pd
import logging
import argparse

logging.basicConfig(filename='preprocessing.log', level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

# Set up parser for CLI arguments
parser = argparse.ArgumentParser(
    description="Preprocess the given files using the given arguments.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--input-file",
                    action='store',
                    dest='filename',
                    type=str,
                    help="Define the file path of the data to be cleaned.")
parser.add_argument("-o", "--output",
                    action='store',
                    dest='output',
                    required=True,
                    type=str,
                    help="Define the file path of the cleaned datafile.")
parser.add_argument("-m", "--min-char",
                    action='store',
                    dest='min_char',
                    type=int,
                    default=100,
                    help="Define the minimum length of a given tweet.")


if __name__ == '__main__':
    logging.info(f"{'###' * 5} Cleaning Input Tweets File {'###' * 5}")

    # Parse arguments
    args = parser.parse_args()
    filename = args.filename
    output_file = args.output
    min_char = args.min_char

    # Load data
    df = pd.read_csv(filename)

    # Clean data
    logging.info(f"Raw data has {df.shape[0]} rows and {df.shape[1]} columns.")
    logging.info(f"Columns are {list(df.columns)}")

    # Drop null data
    prev = df.shape[0]
    df = df.dropna()
    logging.info(f"Removing null tweets & cyberbullying classes reduced data by {(1-(df.shape[0]/prev)):.2%}.")

    # Removing tweets above threshold of characters
    prev = df.shape[0]
    df = df[df.tweet_text.str.len() > min_char]
    logging.info(f"Removing tweets that have less than {min_char} chars, reduced data by {(1-(df.shape[0]/prev)):.2%}.")

    # Removing @ twitter-handles
    prev = df.shape[0]
    df['tweet_text'] = df.tweet_text.str.replace(r'@\w+', '', regex=True)

    # Remove unnecessary class and duplicates
    df = df[df.cyberbullying_type != 'other_cyberbullying']
    df = df[~df.tweet_text.duplicated()]
    df = df[~(df.tweet_text == '')]

    logging.info(f"Removing empty, duplicated, or 'other cyberbullying' tweets reduced data by {(1-(df.shape[0]/prev)):.2%}.")
    logging.info(f"Final data has {df.shape[0]} rows and {df.shape[1]} columns.")

    # Store data
    df.to_csv(output_file, index=False)
    logging.info(f"{'###'*5} Cleaned Input Tweets File {'###'*5}")
