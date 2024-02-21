
"""
pattern_extraction.py contains a single class, NgramPatternCorpus, containing pattern extraction tools which output data
 to ../[corpus name]/pattern_corpus. These files are input requirements for the tune similarity tools stored in
 similarity_search.py.

Initialization of a NgramPatternCorpus object requires feature sequence csv files representing a music corpus, as
generated via corpus_setup_tools.py.

For a user-selectable musical feature and level of data granularity,
NgramPatternCorpus class extracts and represents all local patterns between 3 and 12 elements in length which occur at
least once in the feature sequence corpus files. The number of occurrences of each pattern in each tune is calculated
and stored in a sparse matrix (NgramPatternCorpus.pattern_frequency_matrix). A weighted version of this data is also
calculated, holding TF-IDF values for each pattern in each tune rather than frequency counts
(NgramPatternCorpus.pattern_tfidf_matrix). Use of TF-IDF helps suppress the prominence of frequent-but-insignificant
stop word' patterns.

Pattern extraction and matrix tabulation is via scipy.sparse and sklearn.feature_extraction.text tools which allow
fast & memory-efficient performance. The indexes of the pattern occurrence matrices are stored and written to file in
separate arrays (NgramPatternCorpus.patterns and NgramPatternCorpus.titles).

NgramPatternCorpus class can also calculate Cosine similarity between column vectors in the TFIDF matrix. This gives a
pairwise tune similarity matrix for the entire corpus. This calculation is implemented in
NgramPatternCorpus.calculate_tfidf_vector_cos_similarity() method, using sklearn.metrics.pairwise.linear_kernel.

The pattern occurrence matrices can be written to pandas DataFrames via NgramPatternCorpus.convert_matrix_to_df(), but
this process is memory-intensive and is recommended for small corpora (> approx. 10k tunes) only.
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

    """
    For a user-selectable musical feature, level of data granularity, and range of pattern lengths,
    NgramPatternCorpus class extracts and represents all local patterns which occur at least once in the corpus.

    Class attributes:
        LEVELS -- Defines allowable levels of input data granularity (vals) and their labels (keys)
        FEATURES -- Defines the set of musical feature names allowable as inputs

    Attributes:
        feature -- a single input feature name selected from FEATURES
        in_dir -- path to dir containing csv feature sequence files representing each tune in a music corpus, as
                  outputted by feature_sequence_extraction_tools.py
        out_dir -- directory to write output files
        n_vals -- a tuple holding two integers representing the min and max pattern lengths for which patterns will be
                  extracted. E.g.:
                  n_vals = (4, 4) extracts patterns of 4 elements in length; n_vals = (4, 6) extracts patterns of
                  4-6 elements in length.
        name -- Corpus name. Derived automatically from in_dir directory name
        data -- input data read from csv files
        titles -- tune titles extracted from data
        patterns -- array of all unique local patterns extracted from data
        pattern_freq_matrix -- sparse matrix storing occurrences of all patterns (index) in all tunes (columns)
        pattern_tfidf_matrix -- sparse matrix storing tfidf values of all patterns (index) in all tunes (columns)
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

    LEVELS = {'note': 'note-level', 'duration_weighted': 'note-level (duration-weighted)', 'accent': 'accent-level'}

    FEATURES = {
        'eighth_note',
        'midi_note_num',
        'diatonic_note_num',
        'beat_strength',
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

    def __init__(self, feature='diatonic_scale_degree', in_dir=None, out_dir=None, n_vals=None):

        """
        Initialize NgramPatternCorpus class object.

        Args:
            in_dir -- path to dir containing csv feature sequence files representing each tune in a music corpus.
            out_dir -- directory to write output files.
            n_vals -- a tuple holding min and max pattern length values for which patterns will be extracted. E.g.:
                      n_vals = (4, 4) extracts patterns of 4 elements in length; n_vals = (4, 6) extracts patterns of
                      4-6 elements in length.
        """

        assert feature in NgramPatternCorpus.FEATURES
        self.feature = feature
        assert os.path.isdir(in_dir)
        self.in_dir = in_dir
        self.out_dir = out_dir
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        for n in n_vals:
            assert 3 <= n <= 16
        self.n_vals = n_vals
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

        """Display custom __repr__ giving corpus name, number of tunes, and number of patterns extracted."""

        titles = self.titles
        return f"\nCorpus name: {self.name}\nLevel: {self.level}" \
               f"\nInput directory: {self.in_dir}\nCorpus contains {len(titles)} tunes.\n" \
               f"Number of patterns extracted: {0 if self.patterns is None else len(self.patterns)}\n"

    def read_input_data(self):

        """Read feature sequence data from all csv files in corpus and return in dict."""

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
        """Extract tune titles from csv filenames and write to disc as numpy array."""
        titles = self.titles
        np.save(f"{self.out_dir}/titles", titles, allow_pickle=True)

    def create_pattern_frequency_matrix(self, write_output=True):

        """
        Extract all feature sequence n-gram patterns in corpus to numpy array (NgramPatternCorpus.patterns);
        create as SciPy sparse CSR matrix of pattern occurrences per tune (NgramPatternCorpus.pattern_freq_matrix),
        and optionally write both to disc.

        Args:
            write_output -- Boolean flag: 'True' writes output to disc, 'False' does not.
        """

        data = self.data.values()
        # extract n-gram patterns using sklearn.feature_extraction.text.CountVectorizer
        vec = CountVectorizer(
            input='content',
            lowercase=False,
            tokenizer=lambda x: x,
            ngram_range=self.n_vals,
            analyzer='word'
        )
        # calculate sparse matrix of n-gram pattern occurrences:
        pattern_freq_matrix = vec.fit_transform(data)
        # store all unique n-gram patterns:
        patterns = [
            np.array(
                [int(float(elem.strip())) for elem in pattern.split()], dtype='int16')
            for pattern in vec.get_feature_names()
        ]
        
        # optionally write pattern and frequency outputs to disc
        if write_output:
            file_name = f"{self.n_vals[0]}grams_tfidf_matrix"
            sparse.save_npz(f"{self.out_dir}/{file_name}", pattern_freq_matrix)
            np.save(f"{self.out_dir}/patterns", np.array(patterns, dtype=object), allow_pickle=True)

        self.patterns = patterns
        self.pattern_freq_matrix = pattern_freq_matrix
        # memory management
        self.data = None
        return None

    def calculate_tfidf_vals(self, write_output=True):

        """
        Calculate TF-IDF weighting for freq matrix, stores output as SciPy sparse CSR matrix
        (NgramPatternCorpus.pattern_tfidf_matrix) and optionally write to disc.

        Args:
            write_output -- Boolean flag: 'True' writes output to disc, 'False' does not."""

        input_data = self.pattern_freq_matrix
        # Convert pattern occurrences to TF-IDF values via using sklearn.feature_extraction.text.TfidfTransformer()
        pattern_tfidf_matrix = TfidfTransformer()
        pattern_tfidf_matrix = pattern_tfidf_matrix.fit_transform(input_data)
        # optionally write output to disc
        if write_output:
            file_name = f"{self.n_vals[0]}grams_freq_matrix"
            sparse.save_npz(f"{self.out_dir}/{file_name}", pattern_tfidf_matrix)
        self.pattern_tfidf_matrix = pattern_tfidf_matrix
        # memory management
        self.pattern_freq_matrix = None
        return None

    def calculate_tfidf_vector_cos_similarity(self):

        """
        Calculate pairwise Cosine similarity between TFIDF vectors of all tunes in NgramPatternCorpus.tfidf,
        save as matrix.
        """

        input_data = self.pattern_tfidf_matrix
        # Calculate Cosine similarity via sklearn.metrics.pairwise.linear_kernal
        cos_similarity = linear_kernel(X=input_data, Y=None, dense_output=False).todense().astype('float16')
        # write output to memory map
        output = np.memmap(
            f"{self.out_dir}/tfidf_vector_cos_sim.mm",
            dtype='float16',
            mode='w+',
            shape=cos_similarity.shape
        )
        output[:] = cos_similarity[:]
        self.pattern_tfidf_matrix = None
        # memory management
        output.flush()
        return None

    def convert_matrix_to_df(self, matrix, write_output=True, filename=None):

        """
        Transpose input matrix, add indices and create pandas DataFrame for ease of user inspection.
        NOTE: For large corpora this method may cause memory performance issues.

        Args:
            matrix -- input matrix, can be either NgramPatternCorpus.pattern_freq_matrixor NgramPatternCorpus.tfidf
            write_output -- Boolean flag: 'True' writes output to disc, 'False' does not
            filename -- output file name
        """

        titles = self.titles
        patterns = self.patterns
        # create DataFrame
        df = pd.DataFrame.sparse.from_spmatrix(matrix.T).astype("Sparse[float16, nan]")
        df.columns = titles
        df['patterns'] = patterns
        df.set_index('patterns', inplace=True, drop=True)
        # print and optionally write to file
        print(df.head())
        print(df.info())
        if write_output:
            df.to_pickle(f"{self.out_dir}/{filename}.pkl")
        return None

    @staticmethod
    def check_data_for_null_entries(data):
        """Helper function to filter empty feature sequence csv files from input processing."""
        errors = []
        # identify any feature sequences with length < 2 in input data
        for k, v in data.items():
            if np.size(v) < 2:
                errors.append(k)
        # remove any such items
        if errors:
            print("Null sequences detected and removed from input data")
            for err in errors:
                print(f"{err}")
                data.pop(err, None)

        return data
        

            
            
            

        
