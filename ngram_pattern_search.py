"""
ngram_pattern_search.py contains NgramSimilarity class, which reads the output of setup_ngrams_tfidf.main(), a table
containing:

[rows] All unique n-gram patterns for a given musical feature and range of pattern lengths,
which occur at least once in a given feature sequence format music corpus.

[cols]: tf-idf values for each piece of music in the corpus; plus a single corpus-level idf column.

EG: All accent-level 'pitch_class' feature patterns of 5-10 note-events in length, which occur at least once, extracted
from the MIDI Ceol Rince na hEireann corpus at https://github.com/danDiamo/music_pattern_analysis/tree/master/corpus.

Such a table allows us read tf-idf value per n-gram per piece of music, and, from this input data,
NgramSimilarity can perform the following tasks:

-Extract most frequent n-gram(s) (as ranked by tf-idf) from a single piece of music.
-Find all similar patterns across the corpus via Damerau-Levenshtein distance.
-[Work-in-progress] Identify pieces of music which contain the highest number of similar patterns to the candidate.
"""

from fastDamerauLevenshtein import damerauLevenshtein
import pandas as pd

pd.options.mode.chained_assignment = None


class NgramSimilarity:

    """
    An NgramSimilarity object is instantiated by a single argument, 'inpath', which is the path to a data table of
    n-gram patterns, tf-idf, and idf values, in feather format.

    Attributes:

        input_corpus -- input data file read to pandas Dataframe.

        title -- title of search candidate tune under investigation.
        Assigned in main() and passed to NgramSimilarity.extract_candidate_ngrams().

        candidate_ngrams -- empty attribute, to hold a dict of the most frequent ngram pattern(s) of length n for a
        single candidate piece of music. Attr is populated via NgramSimilarity.extract_candidate_ngrams() method.

        n -- length of candidate pattern(s). Assigned in main() and passed to NgramSimilarity.extract_candidate_ngrams()

        test_corpus -- empty attribute, to hold a filtered version of input_corpus Dataframe, containing only data for
        n-gram patterns with length equal to length of candidate pattern ±1. Attr is populated via
        NgramSimilarity.setup_test_corpus() method.

        ngram_similarity_results -- empty attribute, to hold a filtered version of test_corpus Dataframe, retaining
        similar n-gram patterns to the candidate pattern(s)as measured via Damerau-Levenshtein edit
        distance. Attr is populated via NgramSimilarity.find_similar_patterns() method.

        results -- [work-in-progress] A Dataframe containing two columns, output of
        NgramSimilarity.find_similar_tunes() method:

        [1] 'title: titles of all pieces of music in the corpus which contain at least one of
        the similar patterns held in in NgramSimilarity.ngram_similarity_results.

        [2] 'count': the number of similar n-gram patterns which occur in the tune.

         Current work-in-progress involves evaluating the viability of this table, sorted by count, as a simple
         indicator of melodic similarity. Formal evaluation and additional metrics will be added as work continues.
    """


    def __init__(self, inpath):
        raw_input = pd.read_feather(inpath)
        # convert n-gram arrays to tuples
        raw_input['ngram'] = [tuple(ngram) for ngram in raw_input['ngram']]
        self.input_corpus = raw_input
        self.title = None
        self.candidate_ngrams = None
        self.n = None
        self.test_corpus = None
        self.ngram_similarity_results = None
        self.results = None

    def extract_candidate_ngrams(self, title, n=None, mode=None, indices=None):
        # TODO: Add exceptions for if blocks.

        """
        Extracts frequent n-gram pattern(s) of length 'n' items for a target piece of music.
        Stores results in self.candidate_ngrams dict, formatted per: tune_title [key]: candidate_ngrams [value].

        Args:
            
        title -- title of the candidate tune under investigation. This is equivalent to the filename of the feature
        sequence csv representation of the tune, with the '.csv' suffix removed.

        n -- length of candidate pattern(s).
        EG: n = 6: extracts the most frequent pattern(s) of 6 items in length from the piece of music named in 'title'.

        mode -- either 'tfidf' or 'idx'.
        If mode=='max_tfidf' the method will return all ngram pattern(s) in the target piece of music with the maximum
        tf-ifd value.

        If mode=='idx' the method returns n-gram patterns for a specific range of indices, as ranked by tf-idf.
        The index values are passed as a list to 'indices' parameter.

        indices -- A list of integer column index values. EG: for 'indices'=[0, 1], the top two n-gram patterns as
        ranked by tf-idf will be extracted..

        NOTE: This method will be augmented with n-gram pattern clustering functionality in future work.
        """

        self.n, self.title = n, title
        for col_name in self.input_corpus.columns:
            if self.title in col_name:
                target_col = col_name
        # retain only rows for n-grams of length n:
        filtered_corpus = self.input_corpus[self.input_corpus['ngram'].apply(lambda x: len(x) == self.n)]
        # extract n-gram(s) of max tf-idf value from column in corpus Dataframe corresponding to search candidate tune:
        if mode == 'max_tfidf':
            freq_ngrams = filtered_corpus[['ngram', target_col]][filtered_corpus[target_col] ==
                                                             filtered_corpus[target_col].max()]
        elif mode == 'idx':
            filtered_corpus.sort_values(by=[target_col], ascending=False, inplace=True)
            filtered_corpus.reset_index(inplace=True, drop=True)
            print(filtered_corpus.head())
            freq_ngrams = filtered_corpus[filtered_corpus.index.isin(indices)]

        self.candidate_ngrams = freq_ngrams
        print(f"\nFrequent n-gram pattern(s) extracted from {title}:")
        print(self.candidate_ngrams)
        return self.candidate_ngrams

    def setup_test_corpus(self):

        """
        This method filters an NgramCorpus.corpus Dataframe, retaining only n-gram patterns of length n±1.
        This is to allow the Damerau-Levenshtein edit distance algorithmsdetect similar patterns even if they are of
        slightly different length to the candidate.

        EG: When we look for similar patterns for a given pitch class 6-gram, we extract all 5, 6, and 7-grams using
        this method, and then search them via NgramSimilarity.find_similar_patterns().
        """

        print("\nFiltering corpus n-grams...")
        self.test_corpus = self.input_corpus[self.input_corpus['ngram'].apply(lambda x: self.n-1 <= len(x) <= self.n+1)]
        print("Corpus n-gram filtering complete.")
        return self.test_corpus

    def find_similar_patterns(self, edit_dist_threshold=None):

        """
        This method runs Robert Grigoriou's fastDamerauLevenshtein implementation of the Damerau Levenshtein edit
         distance algorithm, with default weights (a penalty value of 1 for replacement, deletion, swapping or insertion
         of items in sequence)

        Pairwise similarity is calculated for each of the candidate n-gram(s) held in
        NgramSimilarity.candidate_ngrams dict vs each of the n-grams in NgramSimilarity.test_corpus Dataframe.

        The results are filtered according to an integer value, as passed to 'edit_dist_threshold' arg:
        only n-grams with an edit distance less than or equal to the threshold value are retained, and they are returned
        to NgramSimilarity.ngram_similarity_results Dataframe.
        """

        print("\nSearching corpus for similar n-gram patterns...")
        candidates = [candidate for candidate in self.candidate_ngrams['ngram']]
        distances = self.test_corpus[['ngram']].copy()
        # Calculate Damerau-Levenshtein distances:
        for candidate in candidates:
            edit_dists = [damerauLevenshtein(candidate, ngram, similarity=False) for ngram in distances['ngram']]
            distances[f'{candidate}'] = edit_dists
        # Print report:
        distances.set_index('ngram', inplace=True)
        # Filter out rows which do not have any values <= threshold:
        retained = distances[(distances <= edit_dist_threshold).any(1)]
        retained.reset_index(inplace=True)
        # Print report & return results:
        print(f"{len(retained)} Similar patterns detected:")
        print(retained.head())
        self.ngram_similarity_results = retained
        return self.ngram_similarity_results

    def find_similar_tunes(self):

        """
        This method filters NgramSimilarity.test_corpus Dataframe, firstly retaining only rows for n-grams matching
        those in NgramSimilarity.ngram_similarity_results. It then filters the Dataframe columns, retaining only those
        containing any non-zero values.

        After filtering, only pieces of music (i.e.: columns) which themselves contain at least one of the similar
        patterns held in NgramSimilarity.ngram_similarity_results are retained.

        The number of similar patterns contained by each piece of music is counted, and the results are sorted by count.
        Work-in-progress involves evaluation of the efficacy of this count as an indicator of similarity with the
        candidate piece of music.
        """
        print("Searching corpus for similar tunes...")
        # filter NgramSimilarity.test_corpus, retaining only rows for similar n-grams:
        lookup_df = self.test_corpus[self.test_corpus['ngram'].isin(self.ngram_similarity_results['ngram'])]
        lookup_df.drop(['ngram', 'idf'], axis=1, inplace=True)
        # drop all zero-value columns (i.e.: all tunes in which no similar patterns occur):
        lookup_df = lookup_df.loc[:, (lookup_df != 0).any(axis=0)]
        # count non-zero values in all remaining columns (i.e.: for all tunes with at least one similar pattern,
        # return the number of similar n-gram patterns which occur in the tune).
        # NOTE: The frequency of occurrences of each similar n-gram per tune is not yet included in calculations.
        # This will be the first addition for the second draft of deliverable D3.1.
        count = lookup_df.astype(bool).sum(axis=0)
        results = count.to_frame().reset_index()
        results.columns = ['title', 'count']
        results['title'] = results['title'].str.slice(0, -7)
        results.sort_values(by=['count'], ascending=False, inplace=True)
        print(f"Similarity results for {self.title}:")
        print(results.head())
        results.to_csv(f"/Users/dannydiamond/NUIG/Polifonia/CRE_clean/testing/{self.title}.csv")
        self.results = results


def main():

    """
    Initializes NgramSimilarity object by passing the path to a data table of n-gram patterns, tf-idf, and idf values,
    in feather format, as outputted by setup_ngrams_tfidf.main().

    With an NgramSimilarity object set up, we call:

    NgramSimilarity.extract_candidate_ngrams()
    Extracts n-gram patterns from candidate tune under investigation.
    
    Args: 
    
    'title' -- title of candidate piece of music per the filename of the original MIDI file.
    'n' -- pattern length.
    
    'mode' -- If 'tfidf', this method call will extract the most frequent n-gram pattern of length n from the 
    candidate tune, as ranked by tf-idf. If 'idx', the cal will extract the n-grams at indices listed in 
    'indices' arg.
    
    indices -- list of integers corresponding to NgramSimilarity.test_corpus dataframe indices.

    NgramSimilarity.setup_test_corpus()
    Filters corpus Dataframe to include patterns of length n±1.

    NgramSimilarity.find_similar_patterns()
    Finds similar patterns to candidate pattern via Damerau-Levenshtein edit distance.
    arg: 'edit_dist_threshold' = maximum number of edits permitted in Damerau-Levenshtein algorithm.
    NOTE: default edit_dist_threshold=2, but for pieces of music with multiple frequent patterns, reducing this value to 
    1 can prove more effective.

    NgramSimilarity.find_similar_tunes()
    Filters the corpus for pieces of music which include similar patterns to candidate.
    Counts occurrences of similar patterns per piece of music and returns ranked table.
    Saves table to csv.
    
    :returns: NgramSimilarity.results
    
    This is an initial work-in-progress metric of similarity between the search candidate and other pieces of music in
    the corpus. Evaluation and work on additional metrics is ongoing.
    """

    # TODO: Add CLI to allow modification of args.
    inpath = "/Users/dannydiamond/NUIG/Polifonia/CRE_clean/ngrams/cre_pitch_class_accents_tfidf.ftr"
    pattern_search = NgramSimilarity(inpath)
    pattern_search.extract_candidate_ngrams("Lord McDonald's (reel)", n=6, mode='idx', indices=[0, 1])
    pattern_search.setup_test_corpus()
    pattern_search.find_similar_patterns(edit_dist_threshold=1)
    pattern_search.find_similar_tunes()

    return pattern_search.results


main()

