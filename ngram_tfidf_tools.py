"""
'ngram_tfidf_tools.py' is module containing tools to extract all n-grams from a corpus of music in
feature sequence representation.

NgramCorpus class contains methods to extract all n-gram patterns occurring in each melody from a corpus of monophonic
pieces of music, stored in feature sequence format .csv files; it can concatenate this data into a table of unique
corpus-level patterns for a given musical feature; it can also calculate tf-idf values for each instance of each unique
pattern in the corpus.

An NgramCorpus object automatically instantiates an NgramData object for each individual piece of music within the
corpus. NgramData attributes and methods are used by NgramCorpus in ngram extraction and tf-idf calculations.

For more detail, please see docstrings below.
"""

import pathlib

import collections
import numpy as np
import pandas as pd

import utils


class NgramCorpus:

    """
    NgramCorpus class allows extraction of n-gram patters from a corpus of monophonic melodies
    in feature sequence representation. An NgramCorpus object is instantiated by passing a file path ('inpath' arg).
    'inpath' must point to a corpus directory containing multiple csv files containing feature sequence data.

    Attributes:

        feat_seq_corpus -- list of two-item tuples, for which item [0] is the title of a melody and item [1] is
        the associated feature sequence data, read into a Pandas dataframe. This data is read from the directory
        indicated by the 'inpath' arg, which instantiates the NgramCorpus object.

        ngrams -- A list of NgramData class instances, one for each piece of music in the corpus, which are
        automatically instantiated on creation of an NgramCorpus object.

        ngram_freq_corpus -- empty attribute to hold a Dataframe containing frequency counts of all unique ngram
        patterns which occur at least once in the corpus for a given feature. Additional columns give counts of n-gram
        pattern instances for every melody in the corpus.

        ngram_tfidf_corpus -- an equivalent Dataframe to n_gram_corpus, which will hold tf-idf values in place of
        frequency counts.
    """

    def __init__(self, inpath):
        abspaths = [str(filepath.absolute()) for filepath in pathlib.Path(inpath).glob('**/*')]
        self.feat_seq_corpus = [utils.read_csv(file) for file in abspaths if file.endswith('.csv')]
        self.ngrams = [NgramData(feat_seq_df) for feat_seq_df in self.feat_seq_corpus]
        self.ngram_freq_corpus = None
        self.ngram_tfidf_corpus = None

    def extract_corpus_ngrams(self, feature, n_vals):

        """
        This method calls NgramData.extract_ngrams() method on all NgramData objects listed in NgramCorpus.ngrams:
        For a give feature (feature name passed to 'feature' arg) and range of n-values (passed as list to 'n_vals' arg)
        , NgramData.extract_ngrams() extracts all unique n-gram patterns which occur at least one time, counts their
        frequencies, and ranks the patterns by frequency of occurrence. This is applied to the feature sequence data for
         each tune stored in NgramCorpus.feat_seq_data, and the n-gram results for each individual tune are
         listed in NgramCorpus.ngram_freq_corpus.
        """
        print("\nExtracting and sorting n-grams...")
        for ngram_data in self.ngrams:
            ngram_data.extract_ngrams(feature, n_vals)
        return self.ngrams

    def create_corpus_level_ngrams_dataframe(self):

        """
        Concatenates the dataframes created by NgramCorpus.extract_corpus_ngrams() into a single corpus-level n-gram
        frequency dataframe, which is sorted by corpus-level n-gram frequency and stored at
        NgramCorpus.ngram_freq_corpus.
        """

        # concatenate ngram dataframes
        self.ngram_freq_corpus = pd.concat(
            ngram_data.ngrams for ngram_data in self.ngrams).groupby(["ngram"]).sum().reset_index()
        # convert floats to int:
        float_cols = self.ngram_freq_corpus.select_dtypes(include=['float64'])
        for col in float_cols.columns.values:
            self.ngram_freq_corpus[col] = self.ngram_freq_corpus[col].astype('int64')
        # add summing 'freq' column
        self.ngram_freq_corpus.loc[:, 'freq'] = self.ngram_freq_corpus.sum(axis=1)
        # sort:
        self.ngram_freq_corpus.sort_values(by=['freq'], axis=0, ascending=False, inplace=True, ignore_index=True)
        return self.ngram_freq_corpus

    def calculate_corpus_idf_values(self):

        """
        Calculates and appends idf (inverse document frequency) values for every n-gram frequency value in
        NgramCorpus.ngram_freq_corpus dataframe.
        """

        # normalize doc freq to give idf
        self.ngram_freq_corpus['idf'] = np.log((self.ngram_freq_corpus['freq'].sum() /
                                                self.ngram_freq_corpus['freq']) + 1).round(decimals=3)
        print("N-gram processing complete.\n")
        return self.ngram_freq_corpus

    def calculate_corpus_tfidf_values(self):

        """
        Applies NgramData.calculate_tfidf() to all NgramData objects listed in NgramCorpus.ngrams, which calculates a
        tf-idf (term frequency-inverse document frequency) value for each instance of ech n-gram pattern in every tune
        in the corpus.

        NgramData.calculate_tfidf() requires an argument, lookup_table, which is a dataframe containing idf
        values for each n-gram. This must be assigned to NgramCorpus.ngram_freq_corpus.
        """

        print("Calculating tf-idf values...")
        for ngram_data in self.ngrams:
            ngram_data.calculate_tfidf(self.ngram_freq_corpus)
        return self.ngrams

    def create_corpus_level_tfidf_dataframe(self):

        """
        Concatenates the dataframes created by NgramCorpus.calculate_corpus_tfidf_values() into a single corpus-level
        tf-idf dataframe, which is sorted by n-gram idf value, and stored at NgramCorpus.ngram_tfidf_corpus.
        """

        # concatenate tfidf dataframes:
        self.ngram_tfidf_corpus = pd.concat(
            ngram_data.tfidf for ngram_data in self.ngrams).groupby(["ngram"]).sum().reset_index()
        # lookup idf value for each n-gram from NgramCorpus.ngram_freq_corpus, and sort by idf:
        self.ngram_tfidf_corpus['idf'] = self.ngram_tfidf_corpus['ngram'].map(
            self.ngram_freq_corpus.set_index('ngram')['idf'])
        self.ngram_tfidf_corpus.sort_values(by=['idf'], axis=0, ascending=True, inplace=True, ignore_index=True)
        print("tf-idf calculations complete.\n")
        return self.ngram_tfidf_corpus

    def save_corpus_data(self, outpath, corpus_name):

        """
        Saves top 500 rows of corpus n-gram and tfidf dataframes to csv report files, and the entire dataframes to
        feather files.

        outpath -- path to location at which output is to be saved.
        corpus_name -- string label which can be appended to to the auto-formatted output filenames.
        """

        # Save excerpt to csv for human-viewable report:
        print(f"Writing report data to csv at: {outpath}")
        utils.write_to_csv(self.ngram_freq_corpus.head(500), outpath, f"{corpus_name}_ngrams_freq")
        utils.write_to_csv(self.ngram_tfidf_corpus.head(500), outpath, f"{corpus_name}_ngrams_tfidf")


        # Save entire corpus to feather:
        print(f"Writing corpus data to feather at: {outpath}")
        self.ngram_freq_corpus.to_feather(f"{outpath}/{corpus_name}_ngrams_freq.ftr")
        self.ngram_tfidf_corpus.to_feather(f"{outpath}/{corpus_name}_ngrams_tfidf.ftr")


class NgramData:

    """An NgramData is instantiated with feature sequence data for an individual monophonic piece of music, in the
    format of a two-tuple, where item [0] is the title of the piece and item [1] is the feature sequence data.

    This format is provided by utils.read_csv(), and is used in NgramCorpus.__init__() to automatically initialize an
    NgramData object for each file in the corpus directory.

    NgramData attributes:

    title -- title of the piece of music (derived from filename via utils.read_csv()).

    feat_seq_data -- Pandas dataframe containing feature sequence data for an individual monophonic piece of music.

    feature -- name of musical feature under investigation: see setup_corpus.main() doscstring for list of available
    features.

    ngrams -- empty attribute to contain pandas Dataframe holding all unique n-gram patterns extracted from
    feat_seq_data, and the frequency of occurrence of each pattern.

    tfidf -- per ngrams, but the Dataframe will hold tfidf values rather than simple frequency for each n-gram pattern.
    """

    def __init__(self, feat_seq):
        self.title = feat_seq[0]
        self.feat_seq_data = feat_seq[1]
        self.feature = None
        self.ngrams = None
        self.tfidf = None

    def extract_ngrams(self, feature, n_vals):

        """
        For a give feature (feature name passed to 'feature' arg) and range of n-values
        (passed as list to 'n_vals' arg), NgramData.extract_ngrams() extracts all unique n-gram patterns which occur at
        least one time in NgramData.feat_seq_data, counts their frequencies, and ranks the patterns by frequency of
        occurrence.

        Results are stored in NgramData.ngrams attribute.

        EG: feature='pitch_class', n_vlas = [5, 6, 7] will return all unique pitch class patterns of 5-7 items in length
        ; with results ranked by pattern frequency.
        """
        self.feature = feature
        target_feat_seq = self.feat_seq_data[feature]
        # extract n-grams:
        ngrams = [tuple(target_feat_seq[i:i+n]) for n in n_vals for i in range(len(target_feat_seq)-n+1)]
        # count and rank n-grams in dict:
        ngram_count = dict(collections.Counter(ngrams).most_common())
        # store n-grams and counts in dataframe:
        self.ngrams = pd.DataFrame.from_dict(ngram_count, orient='index')
        self.ngrams.reset_index(inplace=True)
        self.ngrams.columns = ['ngram', f'{self.title}_{feature}_freq']
        self.ngrams['ngram'] = [tuple(ngram) for ngram in self.ngrams['ngram']]

        return self.ngrams

    def calculate_tfidf(self, lookup_table):

        """
        Calculates tf-idf value for every n-gram pattern which occurs at least one time in input feature sequence.
        'lookup_table' arg must be assigned to a Pandas dataframe which provides an idf value for each unique
        n-gram pattern.

        Results are stored in NgramData.tfidf attribute.
        """

        # set up dataframe:
        self.tfidf = self.ngrams[['ngram']]
        # calculate tf, idf, and tf-idf:
        tf = self.ngrams[f'{self.title}_{self.feature}_freq'] / \
             self.ngrams[f'{self.title}_{self.feature}_freq'].sum().round(decimals=3)
        idf = self.ngrams['ngram'].map(lookup_table.set_index('ngram')['idf'])
        self.tfidf[f'{self.title}_{self.feature}_freq'] = (tf * idf).round(decimals=3)
        # sort dataframe by tf-idf
        self.tfidf.sort_values(by=[f'{self.title}_{self.feature}_freq'], axis=0, ascending=False, inplace=True)
        return self.tfidf
