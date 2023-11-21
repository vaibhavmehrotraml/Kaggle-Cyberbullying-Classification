"""
Featurize cleaned data into a parquet file
"""

import pandas as pd
import argparse
from profanity_check import predict_prob

parser = argparse.ArgumentParser(
    description="Preprocess the given files using the given arguments.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--input-file",
                    action='store',
                    dest='filename',
                    type=str,
                    help="Define the file path of the cleaned CSV data.")
parser.add_argument("-o", "--output",
                    action='store',
                    dest='output',
                    required=True,
                    type=str,
                    help="Define the file path of the featurized parquet file.")

if __name__ == '__main__':

    # Parse arguments
    args = parser.parse_args()
    filename = args.filename
    output_file = args.output

    if output_file.split('.')[-1] != 'parquet':
        output_file = output_file + '.parquet'

    df = pd.read_csv(filename)

    df['profanity_prob'] = predict_prob(df.tweet_text)

    df.to_parquet(f'{output_file}')
