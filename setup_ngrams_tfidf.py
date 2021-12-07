"""
setup_ngrams_tfidf.py is a simple module defining and running flow control for n-gram pattern extraction and tf-idf
calculations via RunNgramsTfidf class.
"""

from ngram_tfidf_tools import NgramCorpus


class SetupNgramsTfidf:

    """
    An ExtractNgrams object sets up and runs n-gram extraction and tfidf calculations on a corpus of feature sequence
    representations of monophonic pieces of music, stored in a single directory.

    ExtractNgrams class objects are instantiated by passing the class an ngram_extraction_tools.NgramCorpus object.
    ExtractNgrams allows use of NgramCorpus methods (and lower-level ngram_extraction_tools.NgramData methods)
     to extract n-gram patterns and calculate tf-idf.

    Attributes:
        corpus -- An NgramCorpus object representing a monophonic music corpus in feature sequence format. Within the
        NgramCorpus object, data for each individual piece of music in the corpus is represented as an NgramData object.

        feature -- name of target feature for which patterns will be extracted. See setup_corpus.main() docstring for
        list of accepted feature names.

        n_vals -- list of n-values for which patterns will be extracted. Each n-value corresponds to the number of items
        in a pattern.

        EG: setting n_vals = [5, 6, 7] will extract all unique 5-grams , 6-grams , and 7-grams from the corpus,
        i.e.: all unique patterns of 5-7 items in length.
    """

    def __init__(self, corpus, feature, n_vals):
        self.corpus = corpus
        self.feature = feature
        self.n_vals = n_vals

    def extract_ngrams(self):

        """
        Extracts all unique n-grams from a feature sequence corpus, for given feature and range of n-values.
        Stores results in a Pandas dataframe with columns for n-gram frequency in each piece of music,
        plus an aggregated corpus-level frequency column. Results are sorted by corpus-level frequency
        and are stored as self.corpus.ngram_freq_corpus.
        """

        self.corpus.extract_corpus_ngrams(self.feature, self.n_vals)
        self.corpus.create_corpus_level_ngrams_dataframe()
        return self.corpus

    def calculate_tfidf(self):

        """
        This method calculates tf (term frequency), idf (inverse document frequency) and tfidf
        (term frequency-inverse document frequency) values from the the output of RunNgramsTfidf.extract_ngrams().

        tf-idf and idf values for all unique n-grams in the input corpus are stored in a Pandas dataframe with columns
        containing tf-idf value per n-gram in each piece of music, plus an aggregated corpus-level idf column.
        Results are sorted by idf and are stored at self.corpus.tfidf_corpus.
        """

        self.corpus.calculate_corpus_idf_values()
        self.corpus.calculate_corpus_tfidf_values()
        self.corpus.create_corpus_level_tfidf_dataframe()
        return self.corpus

    def save_results(self, outpath, corpus_name):
        """Saves results to file via NgramCorpus.save_corpus_data()."""
        self.corpus.save_corpus_data(outpath, corpus_name)


def main():

    """
    Initializes RunNgramsTfidf class instance by passing the class three arguments: inpath, feature, and n_vals.
    Docstring above for RunNgramsTfidf class contains information on feature and n_vals arguments.

    NOTE: An ngram_extraction_tools.NgramCorpus object must be initialized as a preliminary step.
    Per docstrings in ngram_tfidf_tools.py, it must be passed one argument, 'inpath', the path to a directory of
    monophonic music files in feature sequence representation, each stored in an individual csv file.
    The path to this directory is assigned to the 'inpath' variable below.

    With an extract_ngrams.RunNgramsTfidf object set up, we call its extract_ngrams() and calculate_tfidf() methods, and
    save the outputs to csv, giving:

    (1) A corpus-level table of unique n-grams with n-gram frequency counts for each piece of music,
     and for the entire corpus.

    (2) A corpus-level table of unique n-grams with their tf-idf values for each piece of music,
    and corpus-level idf values.

    An excerpt of each is saved to csv for human inspection, while the entire dataframes are saved to feather.

    The data contained in (2) above will be the input for the next phase of work on frequent pattern extraction and
    similarity-based search.
    """

    # TODO: Add CLI to allow modification of in/out paths, target feature and n_vals.
    basepath = "./corpus"
    inpath = basepath + "/feat_seq_data/accent"
    print(inpath)
    feature = "pitch_class"
    n_vals = list(range(5, 10))
    feat_seq_corpus = NgramCorpus(inpath)
    ngram_corpus = SetupNgramsTfidf(feat_seq_corpus, feature, n_vals)
    ngram_corpus.extract_ngrams()
    ngram_corpus.calculate_tfidf()
    ngram_corpus.save_results(outpath=basepath + "/ngrams",
                              corpus_name='cre_pitch_class_accents')
    return ngram_corpus


if __name__ == "__main__":
    main()
