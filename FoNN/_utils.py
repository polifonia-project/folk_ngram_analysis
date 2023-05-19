"""This file holds FoNN module utility/helper functions"""

from collections import Counter

from bs4 import BeautifulSoup
import os
import pandas as pd
import requests


def concatenate_abc_files(in_path=None, outfile_name=None):
    """If reading a corpus from an input folder containing multiple ABC files, this helper function can be used to
    concatenate the ABC files."""

    concatenated = open(in_path + f'/{outfile_name}.abc', 'w')

    for file in os.listdir(in_path):
        if (file.endswith('.abc') and not file.startswith('.') and file != f"{outfile_name}.abc"):
            with open(in_path + '/' + file, 'r') as raw:
                contents = raw.read()
                concatenated.write(contents)
    concatenated.close()
    return concatenated


def reformat_midi_filenames(indir):
    """Strips special characters and spaces from filenames in input MIDI cre_corpus"""
    for root, dirs, filenames in os.walk(os.path.abspath(indir)):
        for filename in filenames:
            alnum_name = [ch for ch in filename[:-4] if ch.isalnum()]
            reformatted_name = f"{''.join(alnum_name)}.mid"
            os.rename(src=os.path.join(root, filename), dst=os.path.join(root, reformatted_name))
    return None


def get_url_paths_for_online_midi_corpus(url):
    """Returns a list of file paths when passed the url for a directory of online MIDI files."""
    response = requests.get(url)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    corpus_paths = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith('.mid')]
    return corpus_paths


def read_csv(inpath):
    """Reads csv file to Pandas dataframe. Returns a two-tuple, formatted per: (filename, dataframe)."""
    data = pd.read_csv(inpath, index_col=0)

    filename = inpath.split('/')[-1][:-4]
    return filename, data


def write_to_csv(df, outpath, filename):
    "Writes Pandas dataframes to csv and creates subdirectories."
    os.makedirs(outpath, exist_ok=True)
    df.to_csv(f"{outpath}/{filename}.csv", encoding='utf-8')
    return None


def find_most_frequent_value_in_seq(df, seq):
    """Returns the most frequent element in a sequence"""

    # extract feature sequence
    feature_sequence = df[seq].to_numpy()
    # create histogram of element occurrences
    hist = Counter(feature_sequence)
    # return only the dict key with the max associated value:
    res = max(hist.items(), key=lambda x: x[1])
    return res[0]


def main():
    print('Running utils.py')


if __name__ == "__main__":
    main()
