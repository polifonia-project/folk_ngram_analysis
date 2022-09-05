"""
'similarity_search.py' contains PatternSimilarity class, which reads the output of pattern_extraction.py, a pkl
table containing:

[rows] All unique n-gram patterns for a given musical feature and range of pattern lengths,
which occur at least once in a given feature sequence format music cre_corpus.

[cols]: tf-idf values for each piece of music in the cre_corpus; plus a single cre_corpus-level idf column.

EG: All accent-level 'pitch_class' feature patterns of 3-7 note-events in length which occur at least once, extracted
from the MIDI Ceol Rince na hEireann cre_corpus at ./cre_corpus/MIDI.

Such a table allows us read tf-idf value per n-gram per piece of music, and, from this input data,
PatternSimilarity can perform the following tasks:

-Extract most frequent n-gram(s) (as ranked by tf-idf) from a single piece of music.
-Find all similar patterns across the cre_corpus via Damerau-Levenshtein distance.
-[Work-in-progress] Identify pieces of music which contain the highest number of similar patterns to the candidate.
"""

import os

from fastDamerauLevenshtein import damerauLevenshtein
import pandas as pd
from tqdm import tqdm

from utils import calculate_tune_lengths

pd.options.mode.chained_assignment = None


class PatternSimilarity:

    """
    An PatternSimilarity object is instantiated by passing a single argument, 'inpath', which is the path to a
    'pattern cre_corpus' table of n-gram patterns, tf-idf, and idf values held in a pickle file
    (as outputted by 'pattern_extraction.py').

    Attributes:

        pattern_corpus -- pandas Dataframe file containing a sparse matrix of n-gram patterns read from a pkl file as
        outputted by ngram_tfidf-tools.py.

        title -- Empty attribute to hold title of candidate tune tune under investigation. Can be assigned in main().

        candidate_patterns -- empty attribute, to hold a dict of the most frequent ngram pattern(s) of length n for a
        selected musical feature in a candidate tune.
        Attr is populated via PatternSimilarity.extract_candidate_patterns() method.

        n -- length of candidate pattern(s) extracted from tune. Can be assigned in main()

        similar_patterns -- empty attribute to hold a table derived from pattern_corpus Dataframe, retaining
        similar n-gram patterns to the candidate pattern(s)as measured via Damerau-Levenshtein edit
        distance. Attr is populated via PatternSimilarity.find_similar_patterns() method.

        similar_tunes -- empty attribute to hold a table derived from pattern_corpus Dataframe, retaining tunes in which
        similar patterns stored in 'similar_patterns' attr occur at least once.

        results -- [work-in-progress] empty attribute to hold a Dataframe containing two columns outputted by 
        PatternSimilarity.find_similar_tunes() method:
        1. 'title: titles of all tunes in 'similar_tunes' table.
        2. 'count': the number of similar n-gram patterns occurring in each tune.

        results_path -- empty attribute to hold path to directory where tune similarity results outputted by 
        PatternSimilarity.find_similar_tunes() can be written to pkl.

         Current work-in-progress involves evaluating the viability of this table, sorted by count, as a simple
         indicator of melodic similarity. Formal evaluation and additional metrics will be added as work continues.
    """

    def __init__(self, inpath):

        """
        Initializes PatternSimilarity class object.

        Args:
            inpath -- path to pickled 'pattern cre_corpus' Dataframe outputted by 'pattern_extraction.py'.
        """

        raw_data = pd.read_pickle(inpath)
        self.pattern_corpus = raw_data
        self.title = None
        self.candidate_patterns = None
        self.n = None
        self.similar_patterns = None
        self.pattern_results_path = None
        self.similar_tunes = None
        self.pattern_presence_results = None
        self.presence_results_path = None
        self.freq_count_results_results = None
        self.freq_count_results_path = None
        self.tune_lengths = None

    def extract_candidate_patterns(self, title, n=None, mode=None, indices=None):

        # TODO: Refactor for performance

        """
        Extracts frequent n-gram pattern(s) of length 'n' items for a target piece of music.
        Stores results in self.candidate_patterns dict, formatted per: tune_title [key]: candidate_patterns [value].

        Args:
            
        title -- title of the candidate tune under investigation. This is equivalent to the filename of the feature
        sequence csv representation of the tune, with the '.csv' suffix removed.

        n -- length of candidate pattern(s).
        EG: n = 6: extracts the most frequent pattern(s) of 6 items in length from the piece of music named in 'title'.

        mode -- either 'max' or 'idx'.
            If mode=='max' the method will return all ngram pattern(s) in the target piece of music with the maximum
            tf-idf value.
            If mode=='idx' the method returns n-gram patterns for a specific range of indices, as ranked by tf-idf.
            The index values are passed as a list to 'indices' parameter.

        indices -- A list of integer column index values. EG: for 'indices'=[0, 1], the top two n-gram patterns as
        ranked by tf-idf will be extracted..

        NOTE: This method will be augmented with n-gram pattern clustering functionality in future work.
        """

        # read and filter cre_corpus-level pattern data:
        print("Reading input data...")
        print(self.pattern_corpus.head())
        print("Filtering input data...")
        # set parameters from args
        self.title = title
        self.n = n
        filtered_corpus = self.pattern_corpus[self.pattern_corpus['ngram'].apply(lambda x: len(x) == self.n)]
        print(filtered_corpus.head())

        if 'freq' and 'doc_freq' in filtered_corpus.columns:
            filtered_corpus.drop(['freq', 'doc_freq'], axis=1, inplace=True)
        print(filtered_corpus.info())
        print(filtered_corpus.head())

        # find col corresponding to target tune; assign col name to 'title' attr:
        for col_name in tqdm(filtered_corpus.columns, desc='Locating candidate tune in pattern cre_corpus...'):
            if col_name == self.title:
                print(f'Target tune detected: {col_name}')
                target_col = col_name

        # 'max' mode:
        if mode == 'max':
            print("'max' mode selected -- extracting max pattern(s) as ranked by TF-IDF...")
            # convert target col from sparse to dense dtype --
            # this reformatting is necessary for input into pandas' max() function below:
            filtered_corpus[target_col] = filtered_corpus[target_col].to_numpy(dtype='int16')
            # extract n-gram patterns(s) with max TF-IDF value from target tune:
            freq_ngrams = filtered_corpus[['ngram', target_col]][filtered_corpus[target_col] ==
                                                                 filtered_corpus[target_col].max()]
        # 'idx' mode:
        elif mode == 'idx':
            print("'idx' mode selected -- extracting TF-IDF pattern(s) by index...")
            # sort by tfidf in descending order and re-index:
            filtered_corpus.sort_values(by=[target_col], ascending=False, inplace=True)
            filtered_corpus.reset_index(inplace=True, drop=True)
            # extract patterns(s) from target tune by their indices, as passed to 'indices' arg above:
            freq_ngrams = filtered_corpus[filtered_corpus.index.isin(indices)]

        # store and print results:
        self.candidate_patterns = freq_ngrams
        print(f"\nFrequent n-gram pattern(s) extracted from {title}:")
        for pattern in self.candidate_patterns['ngram'].tolist():
            print(pattern)
        return self.candidate_patterns

    def find_similar_patterns(self, edit_dist_threshold=None):

        """
        This method runs Robert Grigoriou's fastDamerauLevenshtein implementation of the Damerau Levenshtein edit
         distance algorithm, with default weights (a penalty value of 1 for replacement, deletion, swapping or insertion
         of items in sequence)

        Pairwise similarity is calculated for each of the search term n-gram pattern(s) held in
        PatternSimilarity.candidate_patterns list vs each of the n-grams in PatternSimilarity.patterns_corpus Dataframe.

        The results are filtered according to an integer value, as passed to 'edit_dist_threshold' arg.
        Only n-grams with an edit distance less than or equal to the threshold value are retained, and they are returned
        to PatternSimilarity.similar_patterns Dataframe.
        """

        # list search term patterns:
        search_term_patterns = self.candidate_patterns['ngram'].tolist()
        # set up new 1-column Dataframe containing all patterns in cre_corpus. The search terms will be tested for
        # similarity with every  pattern in the cre_corpus, and results will be appended to the Dataframe:
        pattern_similarity_table = self.pattern_corpus[['ngram']]

        # Calculate Damerau-Levenshtein distances between search term patterns and all patterns in cre_corpus:
        for search_term in search_term_patterns:
            edit_dists = [
                damerauLevenshtein(search_term, pattern, similarity=False) for pattern in tqdm
                (
                    pattern_similarity_table['ngram'], desc="Calculating pattern similarity..."
                )
            ]
            # Add results for each pattern to Dataframe as a column:
            pattern_similarity_table[f'{search_term}'] = edit_dists

        pattern_similarity_table.set_index('ngram', inplace=True)
        # Filter table to remove less similar patterns via edit distance threshold:
        retained = pattern_similarity_table[(pattern_similarity_table <= edit_dist_threshold).any(1)]
        retained.reset_index(inplace=True)
        # Print & return results:
        print(f"\n{len(retained)} Similar patterns detected:")
        retained.rename(columns={retained.columns[1]: 'DL distance'}, inplace=True)
        retained.sort_values(by='DL distance', inplace=True)
        print(retained.head())
        self.write_results_to_file(retained, self.pattern_results_path, label='patterns')
        self.similar_patterns = retained
        return self.similar_patterns

    def find_similar_tunes(self):

        """
        This method filters PatternSimilarity.test_corpus Dataframe, firstly retaining only rows for n-grams matching
        those in PatternSimilarity.similar_patterns. It then filters the Dataframe columns, retaining only those
        containing non-zero values.

        After filtering, only pieces of music (i.e.: columns) which themselves contain at least one of the similar
        patterns held in PatternSimilarity.similar_patterns are retained.

        The number of similar patterns contained by each piece of music is counted, and the results are sorted by count.
        Work-in-progress involves evaluation of the effectiveness of this count as an indicator of similarity between
        the tunes in the filtered table and the candidate tune.
        """
        print("\nSearching cre_corpus for similar tunes...\n")
        # filter self.pattern_corpus, retaining only rows for similar n-grams:
        lookup_df = self.pattern_corpus[self.pattern_corpus['ngram'].isin(self.similar_patterns['ngram'])]
        # drop 'ngram' & 'idf' columns
        lookup_df.drop(['ngram', 'idf'], axis=1, inplace=True)
        # drop all zero-value columns (i.e.: all tunes in which no similar patterns occur):
        self.similar_tunes = lookup_df.loc[:, (lookup_df != 0).any(axis=0)]
        self.similar_tunes.drop(labels=['freq', 'doc_freq'], axis=1, inplace=True)
        return self.similar_tunes

    def calculate_tune_lengths(self, feat_seq_path):

        # TODO: Test

        self.tune_lengths = calculate_tune_lengths(feat_seq_path)

    def calc_pattern_presence_results(self, normalize=True):

        # TODO: test new normalization block

        """Counts the number of non-zero values in each column of 'similar_tunes' Dataframe outputted by
        PatternSimilarity.find_similar_tunes() above (i.e.: counts the number of tunes in which a pattern appears)

        # NOTE: The frequency of occurrences of each similar n-gram per tune is not yet included in calculations."""

        print("Compiling pattern presence data...")
        # convert 'similar_tunes' to Boolean type:
        similar_tunes = self.similar_tunes.astype(bool)
        # add summing row
        presence = similar_tunes.sum(axis=0)
        presence.name = 'pattern presence'
        similar_tunes = similar_tunes.append(presence)

        if normalize is True:
            print("Normalising presence data...")
            tune_lengths = self.tune_lengths
            similar_tunes = pd.concat([similar_tunes, tune_lengths], axis=0)
            # TODO: Check how rows are identifed here -- this approach is for cols
            results = (similar_tunes['pattern_presence'] / similar_tunes['length']).to_frame().reset_index()
            results.columns = ['title', 'normalized pattern presence']
        else:
            results = presence.to_frame().reset_index()
            results.columns = ['title', 'pattern presence']
        # Reformat / rename cols and sort:
        results.sort_values(by=results.columns[1], ascending=False, inplace=True)
        results.reset_index(inplace=True, drop=True)

        # Print and save results:
        print(f"Similarity results for {self.title} (pattern presence):")
        print(results.head())
        self.write_results_to_file(results, self.presence_results_path, label='presence')
        self.pattern_presence_results = results

    def calc_freq_count_results(self, normalize=True):

        """
        Calculates sum for each column of 'similar_tunes' Dataframe (i.e.: counts the number of times a pattern appears)
        """

        # TODO: test new normalization block

        # Note: for use on freq.pkl or tf.pkl inputs (will run on tfidf.pkl but results will not be valid as tfidf
        # values can be summed but the summed output is not interpretable.
        similar_tunes = self.similar_tunes
        # add summing row:
        count = similar_tunes.sum(axis=0)
        count.name = 'sum'
        similar_tunes = similar_tunes.append(count)

        if normalize is True:
            # TODO: check the below process for logical errors.
            # read tune lengths:
            tune_lengths = self.tune_lengths
            similar_tunes = pd.concat([similar_tunes, tune_lengths], axis=0)
            results = (similar_tunes['sum'] / similar_tunes['length']).to_frame().reset_index()
            results.columns = ['title', 'freq']
        else:
            results = count.to_frame().reset_index()
            results.columns = ['title', 'count']

        # sort
        results.sort_values(by=results.columns[1], ascending=False, inplace=True)

        # Print and save results:
        print(f"Similarity results for {self.title}")
        print("(frequency)" if "freq" in results.columns else "(count)")
        print(results.head())
        self.write_results_to_file(results, self.freq_count_results_path, label='freq' if normalize is True else 'count')

        return self.freq_count_results_results

    def write_results_to_file(self, results, path, label):
        if not os.path.isdir(path):
            os.makedirs(path)
        results.to_csv(f"{path}/{self.title}_{label}.csv")

    def extract_structural_patterns(self):
        pass

    def extract_k_medoids_patterns(self):
        pass

    def validate_results(self):
        # TODO: move to separate file?
        pass


def main():
    """
    Initializes PatternSimilarity object by passing the path to a data table of n-gram patterns, tf-idf, and idf values,
    in feather format, as outputted by setup_ngrams_tfidf.main().

    With an PatternSimilarity object set up, we call:

    PatternSimilarity.extract_candidate_patterns()
    Extracts n-gram patterns from candidate tune under investigation.
    
    Args: 
    
    'title' -- title of candidate piece of music per the filename of the original MIDI file.
    'n' -- pattern length.
    
    'mode' -- If 'tfidf', this method call will extract the most frequent n-gram pattern of length n from the 
    candidate tune, as ranked by tf-idf. If 'idx', the cal will extract the n-grams at indices listed in 
    'indices' arg.
    
    indices -- list of integers corresponding to PatternSimilarity.test_corpus dataframe indices.

    PatternSimilarity.setup_test_corpus()
    Filters cre_corpus Dataframe to include patterns of length n±1.

    PatternSimilarity.find_similar_patterns()
    Finds similar patterns to candidate pattern via Damerau-Levenshtein edit distance.
    arg: 'edit_dist_threshold' = maximum number of edits permitted in Damerau-Levenshtein algorithm.
    NOTE: default edit_dist_threshold=2, but for pieces of music with multiple frequent patterns, reducing this value to 
    1 can prove more effective.

    PatternSimilarity.find_similar_tunes()
    Filters the cre_corpus for pieces of music which include similar patterns to candidate.
    Counts occurrences of similar patterns per piece of music and returns ranked table.
    Saves table to csv.
    
    :returns: PatternSimilarity.results
    
    This is an initial work-in-progress metric of similarity between the search candidate and other pieces of music in
    the cre_corpus. Evaluation and work on additional metrics is ongoing.
    """

    basepath = "./cre_corpus"
    f_in = basepath + "/pattern_corpus/tfidf.pkl"
    res_path = basepath + "/results"
    pattern_search = PatternSimilarity(f_in)
    pattern_search.results_path = res_path
    pattern_search.extract_candidate_patterns("LordMcDonaldsreel", n=6, mode='idx', indices=[])
    pattern_search.find_similar_patterns(edit_dist_threshold=1)
    pattern_search.find_similar_tunes()
    pattern_search.compile_results_table()


if __name__ == "__main__":
    main()
