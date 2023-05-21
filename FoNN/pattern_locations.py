"""pattern_locations.py extracts onset locations for n-gram patterns. The output of this script is an input requirement
for the FoNN Knowledge Graph (KG). This script must be applied to any corpus for which a KG is being generated."""

import csv
from collections import defaultdict
import numpy as np
import os
import pickle


def read_tune_title(filepath):
    """Extract tune title from file path"""
    return filepath.split('/')[-1][:-4]


def read_tune_paths(root_dir):
    """Read paths of all csv files in corpus root folder"""
    return [f"{root_dir}/{file}" for file in os.listdir(root_dir) if file.endswith('.csv')]


def read_tune_data(filepath, feature):
    """Extract data for target musical feature from csv music data files.
    For comprehensive list of musical feature names and explanations please see feature_sequence_extraction_tools.py docstring."""

    # read csv colunm names and check that a column corresponding to target feature is present
    with open(filepath) as data:
        csv_reader = csv.reader(data, delimiter=',')
        cols = next(csv_reader)
        cols_map = {col_name: i for i, col_name in enumerate(cols)}
        assert feature in cols_map
    # identify index of target column
    target_col = cols_map[feature]
    # extract target column data for all tunes to list
    return np.genfromtxt(filepath, dtype='int16', delimiter=',', usecols=target_col, skip_header=1)


def extract_ngrams(data, n_vals):
    """Extract n-grams from feature data extracted via read_tune_data()"""
    # remove NaN values from input data
    data = data[~np.isnan(data)]
    return (tuple((data[i:i + n])) for n in n_vals for i in range(len(data) - n + 1))


def find_ngram_indices(ngrams):

    """Find the onset location at which each n-gram pattern occurs (i.e. the starting index in the input feature
     sequence). Combine with the output of extract_ngrams() and return in defaultdict per ngram patterns (keys):
     locations (vals).

     Args:
         ngrams -- tuple of n-gram patterns outputted by extract_ngrams()."""

    indices = defaultdict(list)
    for i, ngram in enumerate(ngrams):
        indices[ngram].append(i)
    return indices


# ----------------------------------------------------------------------------------------------------------------------

# TODO: Move below to notebook?

# Run on full corpus:

# corpus dir
root_dir = '/Users/dannydiamond/NUIG/Polifonia/MTC/MTC-ANN-2.0/mtc_ann_feat_seq_corpus/feat_seq_dw'
# select musical feature
feature = 'diatonic_scale_degree'
# define n-values (range of pattern lengths)
n_vals = list(range(4, 6))

# run
results = {}
tune_paths = read_tune_paths(root_dir)
for path in tune_paths:
    title = read_tune_title(path)
    data = read_tune_data(path, feature)
    ngrams = list(extract_ngrams(data, n_vals))
    indices = find_ngram_indices(ngrams)
    results[title] = dict(indices)

# # store output as pickle
out_dir = '/Users/dannydiamond/NUIG/Polifonia/MTC/MTC-ANN-2.0/mtc_ann_feat_seq_corpus/locations/'
out_file = 'dw_locations.pkl'
out_path = f"{out_dir}/{out_file}"
with open(out_path, 'wb') as f_out:
    pickle.dump(results, f_out)
