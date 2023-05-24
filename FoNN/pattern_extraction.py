"""
pattern_extraction.py contains pattern extraction tools and outputs a 'pattern corpus' which can be explored via
similarity_search.py.

NgramPatternCorpus class takes as input a music corpus in feature sequence representation, generated via
corpus_setup_tools.py.

For a user-selectable musical feature and level of data granularity,
NgramPatternCorpus class extracts and represents all local patterns between 3 and 12 elements in length,
 which occur at least once in the corpus. The number of occurrences of each pattern in each tune is calculated and
 stored in a sparse matrix (NgramPatternCorpus.pattern_frequency_matrix). A weighted version of this data is also calculated, which holds
 TF-IDF values for each pattern in each tune rather than raw occurrence counts (NgramPatternCorpus.freq). This helps
 suppress the prominence of frequent-but-insignificant 'stop word' patterns. These matrices are core requirements for
 FoNN's similarity_search.py similarity tools.

 Pattern extraction and matrix tabulation is via scipy.sparse and sklearn.feature_extraction.text tools which allow
 fast & memory-efficient performance.

  The indexes of the pattern occurrence matrices are stored and written to file in separate arrays
  (NgramPatternCorpus.patterns and NgramPatternCorpus.titles).

  The class can also calculate an additional output: Cosine similarity between column vectors in the TFIDF matrix.
   This gives a pairwise tune similarity matrix for the entire corpus. This calculation is implemented in
   NgramPatternCorpus.calculate_tfidf_vector_cos_similarity() method, using sklearn.metrics.pairwise.linear_kernel.
"""

import csv
import gc
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm


class NgramPatternCorpus:

    """For a user-selectable musical feature, level of data granularity, and range of pattern lengths,
    NgramPatternCorpus class extracts and represents all local patterns which occur at least once in the corpus.

    Class attributes:
        LEVELS -- Defines allowable levels of input data granularity (vals) and their labels (keys)
        FEATURES -- Defines the set of musical feature names allowable as inputs

    Attributes:
        feature -- a single input feature name selected from FEATURES
        in_dir -- path to dir containing csv feature sequence files representing each tune in a music corpus, as
        outputted by feature_sequence_extraction_tools.py
        out_dir -- directory to write output files
        name -- Corpus name. Derived automatically from in_dir directory name
        data -- input data read from csv files
        titles -- tune titles extracted from data
        patterns -- array of all unique local patterns extracted from data
        freq -- sparse matrix storing occurrences of all patterns (index) in all tunes (columns)
        tfidf -- sparse matrix storing tfidf values of all patterns (index) in all tunes (columns)
        """

    # define slots to improve performance
    __slots__ = [
        'name',
        'level',
        'feature',
        'data',
        'in_dir',
        'out_dir',
        'titles',
        'patterns',
        'freq',
        'tfidf',
        '__dict__'
    ]

    LEVELS = {'note': 'note-level', 'duration_weighted': 'note-level (duration-weighted)', 'acc': 'accent-level'}

    FEATURES = {
        'eighth_note',
        'midi_note_num',
        'diatonic_note_num',
        'chromatic_pitch_class',
        'onset',
        'velocity',
        'bar_count',
        'relative_chromatic_pitch',
        'relative_diatonic_pitch',
        'chromatic_scale_degree',
        'diatonic_scale_degree',
        'chromatic_interval',
        'diatonic_interval',
        'parsons_code',
        'parsons_cumsum',
        'relative_pitch_class'
    }

    def __init__(self, feature='diatonic_scale_degree', in_dir=None, out_dir=None):

        """Initialize NgramPatternCorpus class object
        Args:
            please see class attributes above"""

        assert feature in NgramPatternCorpus.FEATURES
        self.feature = feature
        assert os.path.isdir(in_dir)
        self.in_dir = in_dir
        self.out_dir = out_dir
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        self.name = in_dir.split('/')[-3]
        for l in NgramPatternCorpus.LEVELS:
            if l in in_dir:
                self.level = NgramPatternCorpus.LEVELS[l]
        self.data = self.check_data_for_null_entries(self.read_input_data())
        self.titles = list(self.data)
        self.patterns = None
        self.pattern_freq_matrix = None
        self.pattern_tfidf_matrix = None

    def __repr__(self):

        """Custom __repr__ string giving corpus name, number of tunes, and number of patterns extracted."""

        titles = self.titles
        return f"\nCorpus name: {self.name}\nLevel: {self.level}" \
               f"\nInput directory: {self.in_dir}\nCorpus contains {len(titles)} tunes.\n" \
               f"Number of patterns extracted: {0 if self.patterns is None else len(self.patterns)}\n"

    def read_input_data(self):

        """Extract feature sequence data from all csv files in corpus and return in dict"""

        in_dir = self.in_dir
        feature = self.feature
        titles = [filename[:-4] for filename in os.listdir(in_dir)]
        # list csv files in 'inpath' dir and read each to a pandas DataFrame using FoNN.utils.read_csv():
        inpaths = [f"{in_dir}/{filename}" for filename in os.listdir(in_dir) if filename.endswith(".csv")]
        # identify csv col names
        with open(inpaths[0]) as testfile:
            csv_reader = csv.reader(testfile, delimiter=',')
            cols = next(csv_reader)
            colsmap = {col_name: i for i, col_name in enumerate(cols)}
            assert feature in colsmap
            # print('Features:')
            # for feat in colsmap:
            #     print(feat)
        # identify target feature data column by index
        target_col_idx = colsmap[feature]
        # extract target column data from each tune to list
        target_data = [
            np.genfromtxt(
                path,
                dtype='str',
                delimiter=',',
                usecols=target_col_idx,
                skip_header=1)
            for path in tqdm(inpaths, desc="Reading input data")
        ]
        # format output data into dict per tune titles (keys): feature data (vals)
        data_out = dict(zip(titles, [i.tolist() for i in tqdm(target_data, desc='Formatting data')]))
        print("Process completed.")
        return data_out

    def save_tune_titles_to_file(self):

        """Extract tune titles from csv filenames and write to disc as array."""

        titles = self.titles
        np.save(f"{self.out_dir}/titles", titles, allow_pickle=True)

    def create_pattern_frequency_matrix(self, write_output=True):

        """Extract feature sequence n-gram patterns; create sparse matrix of their occurrences and write both to disc.

        Args:
            write_output -- Boolean flag: 'True' writes output to disc, 'False' does not."""

        data = self.data.values()
        # extract n-gram patterns using sklearn.feature_extraction.text.CountVectorizer
        vec = CountVectorizer(
            input='content',
            lowercase=False,
            tokenizer=lambda x: x,
            ngram_range=(6, 6),
            analyzer='word'
        )
        # calculate sparse matrix of n-gram pattern occurrences:
        freq = vec.fit_transform(data)
        # store all unique n-gram patterns:
        patterns = [np.array([int(float(elem.strip())) for elem in pattern.split()], dtype='int16')
                    for pattern in vec.get_feature_names()]

        # optionally write outputs to disc
        if write_output:
            sparse.save_npz(f"{self.out_dir}/freq_matrix", freq)
            np.save(f"{self.out_dir}/patterns", patterns, allow_pickle=True)

        self.patterns = patterns
        self.pattern_freq_matrix = freq
        # memory management
        self.data = None
        gc.collect()
        return None

    def calculate_tfidf_vals(self, write_output=True):

        """Calculates TF-IDF weighting for freq matrix, stores sparse matrix output as tfidf attr

        Args:
            write_output -- Boolean flag: 'True' writes output to disc, 'False' does not."""

        input_data = self.pattern_freq_matrix
        # Convert pattern occurrences to TF-IDF values via using sklearn.feature_extraction.text.TfidfTransformer()
        tfidf = TfidfTransformer()
        tfidf = tfidf.fit_transform(input_data)
        # optionally write output to disc
        if write_output:
            sparse.save_npz(f"{self.out_dir}/tfidf_matrix", tfidf)
        self.pattern_tfidf_matrix = tfidf
        # memory management
        self.pattern_freq_matrix = None
        gc.collect()
        return None

    def calculate_tfidf_vector_cos_similarity(self):

        """Calculate pairwise Cosine similarity between TFIDF vectors of all tunes in NgramPatternCorpus.tfidf,
        save as matrix."""

        input_data = self.pattern_tfidf_matrix
        # Calculate Cosine similarity via sklearn.metrics.pairwise.linear_kernal
        cos_similarity = linear_kernel(X=input_data, Y=None, dense_output=False).todense().astype('float16')
        # As matrix is symmetrical, convert to triangular for memory efficiency
        triangular = np.triu(cos_similarity)
        # write output to memory map
        output = np.memmap(
            f"{self.out_dir}/tfidf_vector_cos_sim.mm",
            dtype='float16',
            mode='w+',
            shape=cos_similarity.shape
        )
        output[:] = triangular[:]
        self.pattern_tfidf_matrix = None
        output.flush()
        # memory management
        gc.collect()
        return None

    def convert_matrix_to_df(self, matrix, write_output=True, filename=None):

        """Transpose input matrix, add indices and create pandas DataFrame for ease of user inspection.
        NOTE: For large corpora this method may cause memory performance issues.

        Args:
            matrix -- input matrix, can be either NgramPatternCorpus.pattern_freq_matrixor NgramPatternCorpus.tfidf
            write_output -- Boolean flag: 'True' writes output to disc, 'False' does not
            filename -- output file name"""

        titles = self.titles
        patterns = self.patterns
        # create DataFrame
        df = pd.DataFrame.sparse.from_spmatrix(matrix.T).astype("Sparse[float16, nan]")
        df.columns = titles
        df['patterns'] = patterns
        print(df)
        df.set_index('patterns', inplace=True, drop=True)
        # print and optionally write to file
        print(df)
        print(df.info())
        if write_output:
            df.to_pickle(f"{self.out_dir}/{filename}.pkl")
        return None

    @staticmethod
    def check_data_for_null_entries(data):

        """Helper function to detect and remove any empty csv files from input processing."""

        errors = []
        for k, v in data.items():
            if np.size(v) < 2:
                errors.append(k)
        if errors:
            print("Null sequences detected and removed from input data")
            for err in errors:
                print(f"{err}")
                data.pop(err, None)

        return data
