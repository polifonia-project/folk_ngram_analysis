"""
pattern_locations.py extracts pattern location information for all n-gram pattern occurrences in input feature sequence
data at a user-selectable pattern length. This script must be applied to any corpus for which a Knowledge Graph is being
 generated via the Polifonia Patterns Knowledge Graph pipeline at
 https://github.com/polifonia-project/patterns-knowledge-graph.

What we call 'locations' are the onset location or index of a pattern occurrence in a given feature sequence
representing a single tune. For example, pattern [1 2 3 4] occurring in tune [1 2 3 4 5 1 2 3 4 5] will have locations
0 and 5 representing the two indices at which the pattern's first element occurs in the tune sequence.

Functions from this script are applied in ../patterns_knowledge_graph_pipeline/patterns_kg_data_extraction.ipynb
notebook, which is step 1 of FoNN's 2-step Polifonia Patterns Knowledge Graph pipeline.
 """

import csv
from collections import defaultdict
import numpy as np
import os
import pickle

def read_tune_title(filepath):
    """Extract tune title from file path"""
    return filepath.split('/')[-1][:-4]


def read_file_paths(root_dir):
    """Read paths of all csv files in corpus root folder"""
    return [f"{root_dir}/{file}" for file in os.listdir(root_dir) if file.endswith('.csv')]


def read_tune_data(filepath, feature):

    """
    Extract data for target musical feature from csv music data files.
    For comprehensive list of musical feature names and explanations please see feature_sequence_extraction_tools.py
    docstring.
    """

    # read csv column names and check that a column corresponding to target feature is present
    with open(filepath) as data:
        csv_reader = csv.reader(data, delimiter=',')
        cols = next(csv_reader)
        cols_map = {col_name: i for i, col_name in enumerate(cols)}
        assert feature in cols_map
    # identify index of target column
    target_col = cols_map[feature]
    # extract target column data for all tunes to list
    return np.genfromtxt(filepath, dtype='int16', delimiter=',', usecols=target_col, skip_header=1)


def extract_patterns(data, n):
    """Extract n-gram patterns from feature data extracted via read_tune_data()"""
    # remove NaN values from input data
    data = data[~np.isnan(data)]
    return (tuple((data[i:i + n])) for i in range(len(data) - n + 1))


def find_pattern_locations(patterns):

    """Find the onset location at which each n-gram pattern occurs (i.e. the starting index in the input feature
     sequence). Combine with the output of extract_ngrams() and return in defaultdict per ngram patterns (keys):
     locations (vals).

     Args:
         ngrams -- tuple of n-gram patterns outputted by extract_ngrams()."""

    indices = defaultdict(list)
    for i, ngram in enumerate(patterns):
        indices[ngram].append(i)
    return indices
