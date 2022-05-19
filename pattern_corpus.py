"""
'pattern_corpus.py' is a module containing tools to extract n-gram patterns from a music corpus in
feature sequence representation (as outputted by 'corpus_processing_tools.py' module); calculate frequency and TF-IDF
statistics for each pattern; and write the results to file in a sparse data table.

This module contains two classes, PatternCorpus and TuneData.

PatternCorpus class allows extraction of all unique n-gram patterns over a user-selectable range of pattern lengths
(n-values) from an input corpus of monophonic tunes in feature sequence format.
PatternCorpus takes a directory of csv files as input, with each file containing feature sequence representation of a
monophonic tune. PatternCorpus can concatenate this data into a sparse high-level pandas Dataframe holding all unique
patterns which occur at least once in the corpus. It can also calculate simple statistics
(frequency, document frequency) for each pattern, and can calculate TF-IDF values for each pattern instance across the
corpus.

An PatternCorpus object automatically instantiates a TuneData object for each individual tune within the
corpus. TuneData attributes and methods are used by PatternCorpus in pattern extraction and TF-IDF calculations.
The 'pattern corpus' outputted by pattern_corpus.py provides the input for 'similarity_search.py'.

For more detail, please see docstrings below.
"""

import collections
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import read_csv


class PatternCorpus:

    """
    A PatternCorpus object allows extraction of n-gram patters from an input corpus of monophonic melodies in feature
    sequence representation. An PatternCorpus object is instantiated by passing a file path ('inpath' arg).
    'inpath' must point either to a corpus directory containing at least one feature sequence csv file, or to a pickled
     Corpus object, both of which can be outputted by corpus_processing_tools.py.


    Attributes:

        tunes -- providing an 'inpath' argument automatically instantiates a TuneData object for each tune in the input
        corpus, and populates it with feature sequence data. These objects are store in a list at 'tunes' attr.
        If instantiating an PatternCorpus from a pickled Corpus object, a 'target' arg must also be provided.
        This specifies which Corpus attr to read, as explained below in '__init__' docstring.

        pattern_corpus_freq -- Empty attr which will hold a corpus-level sparse dataframe of all unique n-gram patterns
        which occur at least once in the corpus, the frequency of each pattern in each tune, and simple corpus-level
        statistics (frequency, document frequency, IDF) for each pattern.
        This Dataframe is created and populated by PatternCorpus.create_pattern_corpus() and
        PatternCorpus.populate_pattern_corpus() methods.

        tfidf_corpus -- an equivalent Dataframe to pattern_corpus_freq, holding TF-IDF rather than frequency values for
        all n-gram pattern instances in the corpus.
        pattern_corpus_freq -- Empty attr to hold a corpus-level Dataframe of all unique n-gram patterns which occur
        at least once in the corpus.
        pattern_corpus_path -- path at which n-gram pattern frequency and TF-IDF corpora are written to pkl.
    """

    def __init__(self, inpath, target_attr=None):

        """
        Initializes PatternCorpus object.

        Args:
            inpath -- path to either
            1. A directory containing at least one feature sequence csv file, or
            2. A pickled Corpus object.

            target_attr -- if 'inpath' arg points to a pickled Corpus object, use this arg to specify the
            Corpus object attribute from which feature sequence data will be read. Options are:
            1. Note-level feature sequence data (target='feat_seq')
            2. Accent-level feature sequence data (target='feat_seq_accents');
            3. Duration-weighted note-level feature sequence data (target='duration_weighted')
            4. Duration-weighted accent-level feature sequence data (target='duration_weighted_accents')
        """

        # if 'inpath' points to a corpus of csv files:
        if os.path.isdir(inpath):
            # list csv files in 'inpath' dir and read each to a pandas Dataframe using utils.read_csv():
            inpaths = (f"{inpath}/{file}" for file in os.listdir(inpath) if file.endswith(".csv"))
            feat_seq_corpus = [read_csv(file) for file in inpaths]
            # Instantiate a TuneData object from each Dataframe:
            self.tunes = [TuneData(feat_seq_df) for feat_seq_df in feat_seq_corpus]
        # Or, if 'inpath' points to a pickled Corpus object:
        elif os.path.isfile(inpath):
            with open(inpath, 'wb') as f_in:
                corpus = pickle.load(f_in)
                # Read Corpus attr specified in 'target_attr' variable and initialize TuneData objs from it:
                self.tunes = [TuneData(getattr(tune, target_attr)) for tune in corpus]
        else:
            print("PatternCorpus objects can only be instantiated in two ways: "
                             "1: From a directory of csv files containing properly-formatted feature sequence data, or"
                             "2. From a Corpus object stored in a pickle (.pkl) file.")

        self.pattern_corpus_freq = None
        self.pattern_corpus_tfidf = None
        self.pattern_corpus_path = None

    def create_pattern_corpus(self, feature, n_vals):

        """
        Calls TuneData.extract_ngrams() which extracts and counts all unique n-grams patterns from each tune in the
        corpus, storing the results at TuneData.ngrams attr.
        These outputs are concatenated into a single corpus-level Dataframe, indexing and counting occurrences of all
        unique n-gram patterns which occur at least once in the corpus.

        This method also calculates simple high-level statistics for each
        pattern (frequency, document frequency and IDF) and appends as columns to the corpus-level Dataframe, which is
        stored at PatternCorpus.pattern_corpus_freq attr.

        Args:
            feature -- name of the musical feature under investigation, for which patterns will be extracted. For a full
            list of feature names see introductory docstring at: /setup_corpus/corpus_processing_tools.py.
            n_vals -- list of n-values (i.e.: pattern lengths) for which patterns are to be extracted.
        """

        # call TuneData.extract_ngrams() for each tune:
        ngrams = [tune.extract_ngrams(feature, n_vals) for tune in self.tunes]
        # concat results into corpus-level dataframe of all n-grams and their frequencies in each tune:
        corpus = pd.concat(ngrams)
        # calc doc freq values & add as col:
        doc_freq = corpus['ngram'].value_counts()
        # group rows on 'ngram' col. Doing so aggregates the freq and doc_freq values for all duplicate n-grams.
        # Now we have a corpus-level dataframe of all unique n-grams, their corpus-level frequencies and doc frequencies
        corpus = corpus.groupby(["ngram"]).sum(min_count=1)
        corpus['doc_freq'] = doc_freq
        # calc idf values and add as col:
        corpus['idf'] = np.log((len(doc_freq) / doc_freq) + 1).round(decimals=5)
        # sort
        corpus.sort_values(by='doc_freq', inplace=True, ascending=False)
        corpus.reset_index(inplace=True)
        print('\b\b')
        print("Initial n-gram corpus data:")
        print(corpus.head())
        print(corpus.info())
        print('\b\b')
        self.pattern_corpus_freq = corpus

    def populate_pattern_corpus(self):

        """Expands the table of corpus-level patterns created in PatternCorpus.create_ngram-corpus() above:
        For every tune in the corpus this method adds a sparse column of pattern frequency counts, read from the
        TuneData.ngrams attr of each TuneData object listed in PatternCorpus.tunes. Output is stored at
        PatternCorpus.pattern_corpus_freq attr and written to pkl."""

        # for all tunes in corpus:
        for tune in tqdm(self.tunes, desc='Extracting n-gram patterns...'):
            if tune.ngrams is not None:
                # Append n-gram frequency counts for each tune to a new column in PatternCorpus.pattern_corpus_freq
                # NaN values are filled with 0 and columns use pandas sparse dtype to save memory for large corpora:
                self.pattern_corpus_freq[f'{tune.title}'] = self.pattern_corpus_freq['ngram'].map(
                    tune.ngrams.set_index('ngram')['freq']).fillna(0).astype('Sparse[int]')

        print('\b\b')
        print("Populated n-gram corpus dataframe:")
        print(self.pattern_corpus_freq.head())
        print(self.pattern_corpus_freq.info())
        print('\b\b')
        # write output:
        if not os.path.isdir(self.pattern_corpus_path):
            os.makedirs(self.pattern_corpus_path)
        self.pattern_corpus_freq.to_pickle(f"{self.pattern_corpus_path}/freq.pkl")
        self.pattern_corpus_freq = None

    def setup_tfidf_corpus(self):

        """This method reads a pickled PatternCorpus.pattern_corpus_freq object, removes all sparse frequency count
        columns added by populate_pattern_corpus(), and assigns this dataframe to PatternCorpus.pattern_corpus_tfidf
        attr. TF-IDF columns can be appended by PatternCorpus.calculate_tfidf() below."""

        # read dataframe and remove frequency cols:
        self.pattern_corpus_freq = pd.read_pickle(f"{self.pattern_corpus_path}/freq.pkl")
        self.pattern_corpus_tfidf = self.pattern_corpus_freq[['ngram', 'freq', 'doc_freq', 'idf']]
        # manually clear memory:
        self.pattern_corpus_freq = None
        return self.pattern_corpus_tfidf

    def calculate_tfidf(self):

        """For every tune in the corpus, this method calls TuneData.calculate_tfidf(), which calculates TF-IDF values
        for all unique patterns occurring at least once in the tune.
        Results for each tune are stored at TuneData.ngrams attr, and can be concatenated into a corpus-level table via
        PatternCorpus.populate_tfidf_corpus() below."""

        for tune in tqdm(self.tunes, desc='Calculating TF-IDF values...'):
            tune.calculate_tfidf(self.pattern_corpus_tfidf)

    def populate_tfidf_corpus(self):

        """Appends TF-IDF values for each unique pattern in each tune to PatternCorpus.pattern_corpus_tfidf as a new
        column."""
        # for all tunes in corpus:
        for tune in tqdm(self.tunes, desc='Populating TF-IDF corpus...'):
            if tune.ngrams is not None:
                # Append TF_IDF values for each tune to a new column in PatternCorpus.pattern_corpus_tfidf
                # NaN values are filled with 0 and columns use pandas sparse dtype to save memory for large corpora.
                # Note: TF-IDF values are saved as int to save memory. See TuneData.calculate_tfidf() for more info.
                self.pattern_corpus_tfidf[f'{tune.title}'] = self.pattern_corpus_tfidf['ngram'].map(
                    tune.ngrams.set_index('ngram')['tfidf']).fillna(0).astype('Sparse[int]')

        # sort, print and write corpus-level Dataframe to file.
        self.pattern_corpus_tfidf.sort_values(by='idf', inplace=True, ascending=False)
        print('\b\b')
        print("Populated TF-IDF corpus dataframe:")
        print(self.pattern_corpus_tfidf.head())
        print(self.pattern_corpus_tfidf.info())
        self.pattern_corpus_tfidf.to_pickle(f"{self.pattern_corpus_path}/tfidf.pkl")


class TuneData:

    """
    A TuneData object is initialized from feature sequence data representing a monophonic tune.
    TuneData instance methods allow extraction of all unique n-gram patterns from feature sequence data for a
    user-selectable musical feature at a user-defined range of pattern lengths (n-values).

    The class also can calculate frequency and TF-IDF values for each unique n-gram pattern extracted from the feature
    sequence data.

    Attributes:
        title -- tune title string extracted from feature sequence filename by utils.read_csv() helper function.
        feat_seq_data -- pandas Dataframe storing feature sequence data extracted from csv file by utils.read_csv()

        feature -- Empty attr to hold the name of musical feature for which data is to be extracted
        ('pitch' / 'interval', 'pitch_class', etc.)

        n_vals -- Empty attr to hold list of integer 'n' values indicating the length(s) of patterns to be extracted for
        the feature selected above.
        E.G.: n_vals = [3, 4, 5] will extract all 3-item, 4-item, and 5-item patterns for a selected musical feature.

        ngrams -- empty attr to hold a pandas Dataframe of the unique patterns extracted from a tune, their frequencies,
        and/or their TF-IDF values.
    """

    def __init__(self, feat_seq):

        """
        Initializes TuneData object.

        Args:
            feat_seq -- a two-tuple formatted per: (tune title (str), feature sequence data (pandas.Dataframe)).
            This specific format corresponds to that outputted by helper function utils.read_csv(), which
            is used to read csv data to a pandas dataframe while also retaining tune title information extracted from
            the csv filename.
        """

        self.title = feat_seq[0]            # tune title from input feature sequence data filename
        self.feat_seq_data = feat_seq[1]    # data from input feature sequence data filename
        self.feature = None                 # The musical feature under investigation (eg: 'pitch_class')
        self.n_vals = None                  # The n-values for which patterns are to be extracted
        self.ngrams = None                  # Will hold a dataframe of pattern results

    def extract_ngrams(self, feature, n_vals):

        """
        Extracts all unique n-gram patterns from a tune for a user-defined range of n-values in a user-selected
        musical feature. Returns output to TuneData.ngrams attr.

        Args:
            feature -- name of musical feature for which data is to be extracted.
            n_vals -- Empty attr to hold list of integer 'n' values indicating the length(s) of patterns to be extracted
            for the feature selected above.
        """

        self.feature = feature
        self.n_vals = n_vals
        # load in feature sequence data, and clear memory after loading:
        target_feat_seq = self.feat_seq_data[feature].dropna()
        self.feat_seq_data = None
        # extract n-grams:
        ngrams = (tuple((target_feat_seq[i:i+n])) for n in self.n_vals for i in range(len(target_feat_seq)-n+1))
        # count and rank n-grams in dict:
        ngram_count = collections.Counter(ngrams)
        # store n-grams and counts in dataframe:
        ngrams = pd.DataFrame.from_dict(ngram_count, orient='index')
        ngrams.reset_index(inplace=True)

        # The following try-except block is a hold-over from efforts at optimization, may no longer be necessary:
        # It filters out empty dataframes for 'experimental' pieces such as John Cage's 4'33, which have no musical
        # content
        try:
            ngrams.columns = ['ngram', 'freq']
        except ValueError:
            ngrams = None

        # assign n-gram pattern dataframe to self.ngrams attr:
        if ngrams is not None:
            self.ngrams = ngrams

        return self.ngrams

    def calculate_tfidf(self, lookup_table):

        """
        Calculates TF-IDF values for all unique n-gram patterns extracted from a tune by TuneData.extract_ngrams().

        Args:
            lookup_table -- A Dataframe containing idf values for all unique patterns occurring in the tune.
            When called within an PatternCorpus class object, this arg is assigned to the corpus-level patterns table at
            PatternCorpus.pattern_corpus_freq.
        """

        # Find and skip any empty TuneData.ngrams dataframes:
        if self.ngrams is None or self.ngrams.empty:
            pass
        else:
            # Calculate and append 'tf' column:
            self.ngrams['tf'] = self.ngrams['freq'] / self.ngrams['freq'].sum()
            # Pull in idf column values from lookup Dataframe:
            self.ngrams['idf'] = self.ngrams['ngram'].map(lookup_table.set_index(['ngram'])['idf'])
            # multiply the two above to give TF-IDF values:
            tfidf = (self.ngrams['tf'] * self.ngrams['idf']).fillna(0)
            # multiply TF-IDF values by 10**7 and convert from float to int to save memory.
            self.ngrams['tfidf'] = (tfidf * (10**7)).astype('int')

        return self.ngrams


def main():

    """
    Extracts patterns and calculates frequency & TF-IDF values for accent-level pitch class patterns between 3-7
    items in length. Outputs two sparse panda Dataframes:

    1. A corpus-level table of unique n-grams with n-gram frequency counts for each piece of music,
     and for the entire corpus.
    2. A corpus-level table of unique n-grams with their tf-idf values for each piece of music,
    and corpus-level idf values.

    These tables are written to pkl format for input into 'similarity_search.py' pattern search tools.
    """

    basepath = "./corpus"
    inpath = basepath + "/feat_seq_corpus/feat_seq_accents"
    outpath = basepath + "/pattern_corpus"
    feature = 'relative_pitch_class'
    n_vals = [3, 4, 5, 6, 7]
    thesession = PatternCorpus(inpath)
    thesession.pattern_corpus_path = outpath
    thesession.create_pattern_corpus(feature, n_vals)
    thesession.populate_pattern_corpus()
    thesession.setup_tfidf_corpus()
    thesession.calculate_tfidf()
    thesession.populate_tfidf_corpus()


if __name__ == "__main__":
    main()
