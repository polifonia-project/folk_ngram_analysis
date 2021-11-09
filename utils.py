"""This file holds utility functions for 'music_pattern_analysis' repository"""

from collections import Counter

from bs4 import BeautifulSoup
import os
import pandas as pd
import requests


def reformat_midi_filenames(indir):
    """Strips special characters and spaces from filenames in input MIDI corpus"""
    for root, dirs, filenames in os.walk(os.path.abspath(indir)):
        for filename in filenames:
            alnum_name = [ch for ch in filename[:-4] if ch.isalnum()]
            reformatted_name = f"{''.join(alnum_name)}.mid"
            os.rename(src=os.path.join(root, filename), dst=os.path.join(root, reformatted_name))


def get_url_paths_for_online_midi_corpus(url):

    """
    Returns a list of file paths when passed the url for a directory of online MIDI files.
    """

    response = requests.get(url)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    corpus_paths = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.mid')]
    return corpus_paths


def read_csv(inpath):

    """
    Reads csv file to Pandas dataframe. Returns a single-item dict, formatted per:
    filename (key): dataframe (value).
    """

    data = pd.read_csv(inpath, index_col=0)
    print(f"\nReading data from:\n{inpath}")
    print(data.head())
    filename = inpath.split('/')[-1][:-4]
    return filename, data


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
