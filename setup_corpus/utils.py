"""This file holds utility functions for 'music_pattern_analysis' repository"""

from collections import Counter

import os
import pandas as pd


def read_csv(inpath):

    """
    Reads csv file to Pandas dataframe. Returns a single-item dict, formatted per:
    filename (key): dataframe (value).
    """

    data = pd.read_csv(inpath, index_col=0)
    print(f"\nReading data from:\n{inpath}")
    print(data.head())
    filename = inpath.split('/')[-1][:-4]
    return {filename: data}


def write_to_csv(df, outpath, filename):
    os.makedirs(outpath, exist_ok=True)
    df.to_csv(f"{outpath}/{filename}.csv", encoding='utf-8')
    return None


def extract_feature_sequence(df, seq):
    return df[seq].to_numpy()


def filter_dataframe(df, seq, threshold=80):
    return df[df[seq] > threshold]


def find_most_frequent_value_in_seq(df, seq):
    hist = Counter(extract_feature_sequence(df, seq))
    # return only the dict key with the max associated value:
    res = max(hist.items(), key=lambda x: x[1])
    return res[0]


def remove_cols_from_dataframe(df, col_names):
    df.drop(col_names, axis=1, inplace=True)
    print(df.head())
    return df


def main():
    print('Running utils.py')

if __name__ == "__main__":
    main()


