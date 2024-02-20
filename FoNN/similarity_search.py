# TODO: format demo notebook
# TODO: docstrings and comments
# TODO: ordering within inits
# TODO: misc line items below.

from abc import ABC, abstractmethod
import os

from Levenshtein import distance
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack
from scipy.stats import median_abs_deviation
from sklearn.metrics import pairwise_distances
import swifter
import weighted_levenshtein

import FoNN.pattern_extraction
from FoNN._edit_dist_weights import substitution, insertion_ham, deletion_ham

# globals:
pd.options.mode.chained_assignment = None
FEATURES = FoNN.pattern_extraction.NgramPatternCorpus.FEATURES
LEVELS = FoNN.pattern_extraction.NgramPatternCorpus.LEVELS


class TuneSimilarity(ABC):

    def __init__(
            self,
            corpus_path=None,
            level=None,
            query_tune=None,
            feature=None,
            *params
    ):

        # properties (derived from args above)
        self.query_tune = query_tune

        # attrs
        self.corpus_path = corpus_path
        self.level = level
        assert feature in FEATURES
        self.feature = feature
        _titles_path = f"{self.corpus_path}/pattern_corpus/{self.level}/titles.npy"
        print("Loading titles...")
        self._titles = np.load(_titles_path)
        print("Done.")
        _patterns_path = f"{self.corpus_path}/pattern_corpus/{self.level}/patterns.npy"
        print("Loading patterns...")
        self._patterns = pd.Series(np.load(_patterns_path, allow_pickle=True).tolist())
        print("Done.")
        self._out_dir = f"{corpus_path}/similarity_results/{level}"
        self.initialize_params(*params)
        # create out_dir if it does not already exist
        if not os.path.isdir(self._out_dir):
            os.makedirs(self._out_dir)

    @property
    def query_tune(self):
        """Define 'query_tune' property, corresponding to title of tune under investigation."""
        return self._query_tune

    @query_tune.setter
    def query_tune(self, val):
        """Set 'query_tune' value. Note: must match the filename of a tune in the corpus (excluding filetype suffix)."""
        if val:
            assert val in self._titles
            self._query_tune = val
        else:
            self._query_tune = None

    def initialize_params(self, *params):
        pass

    @abstractmethod
    def _setup_precomputed_tfidf_similarity_matrix(self):
        pass

    def _lookup_precomputed_results(self, data, ascending=False):

        """
        Look up precalculated tune similarity matrix and returns results for query tune.

        Args:
            data -- input matrix
            ascending -- Boolean flag: 'True' sorts results in ascending order (for distance matrix inputs);
                         'False' sorts in descending order (for similarity matrix inputs).
        """

        titles = self._titles
        query_tune = self.query_tune
        # find query tune in titles array, print title and index
        query_tune_idx = int(np.where(titles == query_tune)[0])
        # lookup similarity matrix via index of query tune; store results in DataFrame
        results_raw = np.array(data[query_tune_idx], dtype='float16')
        similarity_results = pd.DataFrame(results_raw, index=titles, columns=[query_tune], dtype='float16')
        # sort results and return top 500
        similarity_results.sort_values(axis=0, ascending=ascending, by=query_tune, inplace=True)
        return similarity_results[:500]

    @abstractmethod
    def _read_precomputed_tfidf_vector_similarity_results(self):
        pass

    @abstractmethod
    def _extract_incipits_and_cadences(self):
        pass

    @abstractmethod
    def _load_sparse_matrix(self):
        pass

    @abstractmethod
    def _create_incipit_and_cadence_hamming_dist_matrix(self):
        pass

    @abstractmethod
    def _calculate_incipit_and_cadence_edit_distance(self):
        pass

    @abstractmethod
    def _incipit_and_cadence_flow_control(self):
        pass

    @abstractmethod
    def _extract_search_term_motifs_from_query_tune(self):
        pass

    @abstractmethod
    def _run_motif_edit_distance_calculations(self):
        pass

    @abstractmethod
    def _calculate_custom_weighted_motif_similarity_score(self):
        pass

    @abstractmethod
    def run_similarity_search(self):
        pass


class TFIDFSimilarity(TuneSimilarity):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._precomputed_tfidf_similarity_matrix_path = \
            f"{self.corpus_path}/pattern_corpus/{self.level}/tfidf_vector_cos_sim.mm"
        # setup input data for 'tfidf' similarity method
        print("Loading TF-IDF Cosine similarity matrix...")
        self._precomputed_tfidf_similarity_matrix = self._setup_precomputed_tfidf_similarity_matrix()
        print("Done.")

    def _setup_precomputed_tfidf_similarity_matrix(self):
        """Load TF-IDF Cosine similarity matrix, store as PatternSimilarity._precomputed_tfidf_similarity_matrix."""
        titles = self._titles
        matrix_path = self._precomputed_tfidf_similarity_matrix_path
        x = y = len(titles)
        return np.memmap(matrix_path, dtype='float16', mode='r', shape=(x, y))

    def _read_precomputed_tfidf_vector_similarity_results(self):
        """Apply _lookup_precomputed_results() to TF-IDF vector Cosine similarity matrix and write output to disc."""

        tfidf_similarity_matrix = self._precomputed_tfidf_similarity_matrix
        # read tfidf similarity matrix via _lookup_precomputed_results()
        tfidf_results = self._lookup_precomputed_results(tfidf_similarity_matrix, ascending=False)
        # format and print output
        tfidf_results = tfidf_results.rename(columns={f"{self.query_tune}": "Cosine similarity"})

        # drop query tune from results
        tfidf_results = tfidf_results[tfidf_results.index != self.query_tune]

        print(tfidf_results.head())

        # save & return output
        tfidf_results_dir = f"{self._out_dir}/{self.query_tune}"
        if not os.path.isdir(tfidf_results_dir):
            os.makedirs(tfidf_results_dir)
        tfidf_results_path = f"{tfidf_results_dir}/{self.query_tune}_tfidf_results.csv"
        tfidf_results.to_csv(tfidf_results_path)

    def run_similarity_search(self):
        """run 'tfidf' similarity method; format, save, and print output."""
        print(f'Running TF-IDF similarity search...')
        self._read_precomputed_tfidf_vector_similarity_results()

    def _extract_incipits_and_cadences(self):
        pass

    def _create_incipit_and_cadence_hamming_dist_matrix(self):
        pass

    def _calculate_incipit_and_cadence_edit_distance(self):
        pass

    def _incipit_and_cadence_flow_control(self):
        pass

    def _load_sparse_matrix(self):
        pass

    def _extract_search_term_motifs_from_query_tune(self):
        pass

    def _run_motif_edit_distance_calculations(self):
        pass

    def _calculate_custom_weighted_motif_similarity_score(self):
        pass


class IncipitAndCadenceSimilarity(TuneSimilarity):

    EDIT_DIST_METRICS = {
        'levenshtein': 'Levenshtein distance',
        'hamming': 'Hamming distance',
        'weighted_hamming': 'custom-weighted Hamming distance'
    }

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # setup input path and load data for 'incipit_and_cadence' similarity method
        self._feat_seq_data_path = f"{self.corpus_path}/feature_sequence_data/{self.level}"
        print("Extracting incipit and cadence subsequences from feature sequence data...")
        # extract incipit and cadence subsequences from feature sequence data
        self._incipits_and_cadences = self._extract_incipits_and_cadences()
        print("Done.")

    def _extract_incipits_and_cadences(self):
        """Extract incipit and cadence sequences from all tunes and store output in single corpus-level DataFrame."""

        in_dir = self._feat_seq_data_path
        feature = self.feature

        # define incipit bar numbers. The first 4 bars are taken to represent the incipit.
        incipit_bars = [1, 2, 3, 4]
        # Bars 7 and 8 are taken to represent the first part-ending cadence.
        # define cadence bars and add to list
        cadence_bars = [7, 8]
        incipit_and_cadence_bars = incipit_bars + cadence_bars

        # setup output DataFrame and populate by slicing incipit and cadence subsequences from corpus feature sequence
        # csv files
        incipits_and_cadences = pd.DataFrame(dtype='float16')
        for file_name in os.listdir(in_dir):
            if file_name.endswith('.csv'):
                tune_data = pd.read_csv(f"{in_dir}/{file_name}", index_col=0)
                # filter empty csvs
                if tune_data.empty or len(tune_data) < 2:
                    print("Empty file:")
                    print(file_name)
                    pass
                else:
                    # slice relevant data from csv file by filtering 'bar_num' column by bar_nums
                    filtered_data = tune_data[tune_data['bar_num'].isin(incipit_and_cadence_bars)]
                    incipits_and_cadences[file_name[:-4]] = filtered_data[feature]

        incipits_and_cadences.fillna(0, inplace=True)
        incipits_and_cadences = incipits_and_cadences.astype('float16').reset_index(drop=True).T
        return incipits_and_cadences

    def _create_incipit_and_cadence_hamming_dist_matrix(self):

        """
        Calculate matrix of Hamming distance between all incipit and cadence sequences by applying
        sklearn.metrics.pairwise_distances().
        """

        incipits_and_cadences = self._incipits_and_cadences
        # calculate distance matrix
        incipits_and_cadences_hamming_dist = pairwise_distances(
            incipits_and_cadences, metric='hamming'
        ).astype('float16')

        return incipits_and_cadences_hamming_dist

    def _calculate_incipit_and_cadence_edit_distance(self, edit_dist_metric):

        """
        Apply selected edit distance method to incipit and cadence input data.

        Args:
            metric -- select similarity of distance metric by name. Value can be 'levenshtein'
                      (Levenshtein distance); 'hamming' (Hamming distance); or 'weighted_hamming'
                      (custom-weighted Hamming distance).
        """

        incipits_and_cadences = self._incipits_and_cadences
        test_terms = incipits_and_cadences.to_numpy().tolist()
        # convert all incipit and cadence sequences to as required by external Levensthein.distance() method.
        formatted_test_terms = [''.join([str(int(i)) for i in t]) for t in test_terms]
        # setup results object
        results = None

        if edit_dist_metric == 'levenshtein':
            query_tune = self.query_tune
            # look up query tune in incipits_and_cadences table.
            search_term = incipits_and_cadences.loc[query_tune]
            # compare against all other incipit and cadence sequences via Levensthein.distance()
            lev_results = incipits_and_cadences.swifter.apply(lambda row: distance(row, search_term), axis=1)
            # Store in DataFrame; format and slice, retaining top 500 results
            lev_results = pd.DataFrame(lev_results.sort_values(), dtype='int16')
            lev_results.columns = ['Levenshtein distance']
            results = lev_results.head(500)

        if edit_dist_metric == 'hamming':
            # look up Hamming distance matrix calculated via _lookup_precomputed_results()
            hamming_dist_matrix = self._create_incipit_and_cadence_hamming_dist_matrix()
            hamming_results = self._lookup_precomputed_results(hamming_dist_matrix, ascending=True)
            hamming_results.columns = ['Hamming distance']
            results = hamming_results.head(500)

        if edit_dist_metric == 'weighted_hamming':
            query_tune = self.query_tune
            # convert all incipit and cadence sequences to as required by weighted_levenshtein.levenshtein()
            search_term = ''.join(str(int(i)) for i in incipits_and_cadences.loc[query_tune].tolist())

            # compare against all other incipit and cadence sequences via weighted_levenshtein.levenshtein():
            # by only allowing substitution of elements, this Levenshtein implementation now functions as a Hamming
            # distance.
            # weighted_levenshtein.levenshtein() takes a custom substitution matrix, which allows musically consonant
            # substitutions to be penalised less heavily than dissonant substitutions. This matrix is defined in
            # FoNN.edit_dist_weights.py and passed to the function call below via substitute_costs keyword arg.
            weighted_hamming_dists = [weighted_levenshtein.levenshtein(
                search_term,
                t,
                substitute_costs=substitution
            ) for t in formatted_test_terms]
            # Store results in DataFrame; format and slice, retaining top 500 results
            weighted_hamming_results = pd.DataFrame(
                weighted_hamming_dists,
                index=incipits_and_cadences.index,
                columns=['Custom-weighted Hamming distance'],
                dtype='float16'
            )
            weighted_hamming_results.sort_values(by='Custom-weighted Hamming distance', inplace=True)
            results = weighted_hamming_results.head(500)

        return results

    def _incipit_and_cadence_flow_control(self, edit_dist_metric='levenshtein'):

        """
        Flow control: run 'incipit and cadence' similarity method and write results to disc.

        Args:
            edit_dist_metric -- select edit distance metric. Value can be 'levenshtein' (Levenshtein distance);
                                'hamming' (Hamming distance); or
                                'weighted_hamming' (custom-weighted Hamming distance), as defined in
                                self.EDIT_DIST_METRICS class constant.
        """

        # apply selected edit distance metric to calculate pairwise distance between incipit and cadence of query tune
        # vs all other tunes.
        incipit_and_cadence_results = self._calculate_incipit_and_cadence_edit_distance(edit_dist_metric)
        # drop query tune from results
        incipit_and_cadence_results = incipit_and_cadence_results[incipit_and_cadence_results.index != self.query_tune]
        print(incipit_and_cadence_results.head())

        # format output filenames and write results to file
        incipit_and_cadence_results_dir = f"{self._out_dir}/{self.query_tune}"
        if not os.path.isdir(incipit_and_cadence_results_dir):
            os.makedirs(incipit_and_cadence_results_dir)
        incipit_and_cadence_results_path = \
            f"{incipit_and_cadence_results_dir}/{self.query_tune}_incipit_and_cadence_results_{edit_dist_metric}.csv"
        incipit_and_cadence_results.to_csv(incipit_and_cadence_results_path)

    def run_similarity_search(self, edit_dist_metric='levenshtein'):

        # run incipit and cadence similarity method
        assert edit_dist_metric in self.EDIT_DIST_METRICS.keys()
        formatted_edit_dist_metric = ' '.join(edit_dist_metric.split('_')).title()
        print(f"Running 'incipit and cadence' similarity search, using {formatted_edit_dist_metric} distance metric...")
        self._incipit_and_cadence_flow_control(edit_dist_metric=edit_dist_metric)

    def _setup_precomputed_tfidf_similarity_matrix(self):
        pass

    def _read_precomputed_tfidf_vector_similarity_results(self):
        pass

    def _load_sparse_matrix(self):
        pass

    def _extract_search_term_motifs_from_query_tune(self):
        pass

    def _run_motif_edit_distance_calculations(self):
        pass

    def _calculate_custom_weighted_motif_similarity_score(self):
        pass


class MotifSimilarity(TuneSimilarity):

    def __init__(self, n, *args, **kwargs):

        # properties
        self.n = n  # TODO: document

        # attrs
        super().__init__(*args, **kwargs)
        # setup input data for 'motif' similarity method
        _pattern_frequency_matrix_path = f"{self.corpus_path}/pattern_corpus/{self.level}/{self.n}grams_freq_matrix.npz"
        print("Loading pattern occurrences matrix...")
        self._pattern_frequency_matrix = self._load_sparse_matrix(_pattern_frequency_matrix_path)
        print("Done.")
        _pattern_tfidf_matrix_path = f"{self.corpus_path}/pattern_corpus/{self.level}/{self.n}grams_tfidf_matrix.npz"
        print("Loading pattern TF-IDF matrix...")
        self._pattern_tfidf_matrix = self._load_sparse_matrix(_pattern_tfidf_matrix_path)
        print("Done.")
        self._search_terms = None  # TODO: document this attr

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):

        """
        Set 'n' property representing length of search term pattern(s) to be extracted from query tune when using
        'motif' similarity mode. 'n' can be any integer value between 3 and 12.
        """

        if val:
            assert 3 <= val <= 16
            self._n = val
        else:
            self._n = None

    def _load_sparse_matrix(self, in_dir):

        """
        Load sparse matrix from file.

        Args:
            in_dir -- directory containing sparse matrix file
        """

        data = load_npz(in_dir).transpose(copy=False)
        return data

    def _extract_search_term_motifs_from_query_tune(self):

        """
        Extract representative motif(s) from query tune using automatically-calculated TF-IDF value threshold; patterns
        with TF-IDF values above the threshold are taken as prominent within the query tune and returned for use as
        search terms in self._run_motif_edit_distance_calculations().
        """

        title = self.query_tune
        data = self._pattern_tfidf_matrix
        title_idx = np.where(self._titles == title)[0]
        # read query tune TF-IDF data from matrix and flatten into array
        query_tune_data = np.squeeze(np.asarray(data[:, title_idx].todense()))
        # drop 0s
        query_tune_data_nonzero = query_tune_data[query_tune_data != 0]

        # calculate median, max and MAD of TF-IDF values
        _median = np.median(query_tune_data_nonzero)
        print(f"Median: {_median}")
        _max = np.max(query_tune_data_nonzero)
        print(f"Max: {_max}")
        m_a_d = median_abs_deviation(query_tune_data_nonzero)
        print(f"Median absolute deviation: {m_a_d}")
        # set threshold TF-IDF value; store patterns with TF-IDF vals above this threshold for use as pattern similarity
        # search terms.
        threshold = _median + m_a_d
        print(f"Threshold: {threshold}")
        search_term_tfidf_vals = query_tune_data[query_tune_data >= threshold]
        print(f"{len(search_term_tfidf_vals)} search term patterns extracted:")
        search_term_pattern_ids = np.asarray(np.where(query_tune_data >= threshold)).flatten()
        # search_term_patterns = patterns[search_term_pattern_ids].tolist()
        # print('Search term patterns:')

        # store search term pattern ids & sort by their corresponding TF-IDF values
        search_terms = np.vstack((search_term_pattern_ids, search_term_tfidf_vals)).T
        self._search_terms = search_terms[search_terms[:, 1].argsort()[::-1]]

    def _run_motif_edit_distance_calculations(self):

        """
        Calculate custom-weighted Hamming distance between search term motif(s) and all corpus patterns stored in
        self._patterns attr. Detect patterns within a distance threshold of 1 and write them to file.
        For each of our query patterns, these patterns are taken as similar.
        Return indices for all similar patterns (which correspond to row identifiers in
        PatternSimilarity._pattern_freq_matrix).
        """

        # look up input search term patterns and TF-IDF values
        self._extract_search_term_motifs_from_query_tune()
        search_term_pattern_ids = self._search_terms.T[0].astype(int)
        search_term_pattern_tfidf_vals = self._search_terms.T[1]

        patterns = self._patterns
        # convert patterns held in self._patterns attr from numeric arrays to strings
        # This is for compatibility with weighted_levenshtein.levenshtein() function call within local
        # custom_hamming_dist() function below.
        reformatted_patterns = np.asarray([''.join([str(int(i)) for i in p]) for p in patterns])
        # lookup search terms and slice from reformatted patterns array
        reformatted_search_terms = reformatted_patterns[search_term_pattern_ids]

        def custom_hamming_dist(x, y):

            """
            Run custom-weighted Hamming distance calculations. This local function uses the external
            weighted_levenshtein package, which is supplemented with a custom substitution matrix based on diatonic
            consonance.
            """

            return weighted_levenshtein.levenshtein(
                x,
                y,
                substitute_costs=substitution,
                insert_costs=insertion_ham,
                delete_costs=deletion_ham
            )

        # use numpy.vectorize() to apply the above function to all search term patterns in a single call
        vectorized_edit_dist = np.vectorize(custom_hamming_dist)
        # call custom Hamming distance function and format output in array
        edit_distance_results = vectorized_edit_dist(reformatted_patterns[:, np.newaxis], reformatted_search_terms).T

        pattern_similarity_results = []
        # iterate the similar pattern results generated for each search term pattern
        for idx, pattern_results_array in enumerate(edit_distance_results):
            pattern_id = search_term_pattern_ids[idx]
            # convert from NumPy to Pandas to allow use of index label cols
            similar_patterns = pd.Series(pattern_results_array, name=pattern_id)
            similar_patterns = similar_patterns[similar_patterns <= 1].sort_values()
            similar_patterns_df = pd.DataFrame(similar_patterns)
            # take the TF-IDF value of the search term patter from the original query tune and append as new scalar col.
            tfidf_val = search_term_pattern_tfidf_vals[idx]
            similar_patterns_df.insert(0, 'tfidf_weighting_factor', [tfidf_val] * len(similar_patterns))
            # store the DataFrames generated for each search term pattern in a list
            pattern_similarity_results.append(similar_patterns_df)

        # concat into a single DataFrame and format indices
        pattern_similarity_data_table = pd.concat(pattern_similarity_results).astype(pd.SparseDtype("float", np.nan))
        pattern_ids = pattern_similarity_data_table.index.tolist()
        detected_patterns = patterns[pattern_ids].tolist()
        pattern_similarity_data_table.insert(0, 'pattern', detected_patterns)
        cols_labels = [int(i) for i in pattern_similarity_data_table.columns[2:]]
        # line below expects array, currently is provided with a list -- does this also break for The Session?
        cols_formatted = [str(i) for i in patterns[cols_labels]]
        new_cols = pattern_similarity_data_table.columns[:2].tolist() + cols_formatted
        pattern_similarity_data_table.columns = new_cols

        # return the formatted DataFrame
        return pattern_similarity_data_table

    def _calculate_custom_weighted_motif_similarity_score(self, weighting='single'):

        """
        First, find all similar patterns to the 'motif' search term(s).
        Occurrences of the similar patterns returned are looked up in self_pattern_frequency_matrix via their indices.
        Matrix rows containing their occurrences per each tune in the corpus are weighted using 'consonance' and
        'prominence' factors. 'Consonance' weighting factor is applied by default. 'Prominence' factor
        can be enabled by the user if desired.
        These weighted frequency counts of similar pattern occurrences across the corpus are summed on a tune-by-tune
        basis and normalised, outputting both raw and normalised similarity scores.

        Args:
            weighting -- value can be either 'single' (default; 'consonance' weighting factor applied) or
                         'double' (both 'consonance' and 'prominence' weighting factors applied).
        """

        assert weighting == 'single' or 'double'

        # run pattern similarity search via self._run_motif_edit_distance_calculations()
        edit_distance_results = self._run_motif_edit_distance_calculations()
        # slice cols for re-use later in this method
        sliced_cols = edit_distance_results[['pattern', 'tfidf_weighting_factor']]
        sliced_cols.rename(columns={'tfidf_weighting_factor': 'prominence'}, inplace=True)
        # although retained and renamed for later use in the sliced_cols variable, 'tfidf_weighting_factor' col is
        # dropped from the main edit_distance_results DataFrame.
        edit_distance_results.drop(['tfidf_weighting_factor'], axis=1, inplace=True)
        # Conditionally slice the content of edit_distance_results DataFrame based on edit distance values for each
        # pattern / row:
        # Exact matches (i.e. detection on the search term pattern) have a distance value of 0
        exact_pattern_ids = edit_distance_results[(edit_distance_results == 0).any(1)].index.tolist()
        # Very similar patterns (i.e. those containing a single consonant diatonic substitution)
        # have a distance value of 0.5
        very_similar_pattern_ids = edit_distance_results[(edit_distance_results == 0.5).any(1)].index.tolist()
        # Similar patterns (i.e. those containing either two consonant diatonic substitutions or one dissonant diatonic
        # substitution) have a distance value of 0.5.
        similar_pattern_ids = edit_distance_results[(edit_distance_results == 1).any(1)].index.tolist()
        print(f"{len(exact_pattern_ids)} exact matches detected")
        print(f"{len(very_similar_pattern_ids)} very similar patterns detected.")
        print(f"{len(similar_pattern_ids)} similar patterns detected.")

        # look up the rows corresponding to the three pattern subsets above in
        # PatternSimilarity._pattern_frequency_matrix (corpus-level sparse matrix of pattern frequencies per tune)
        freq_matrix = self._pattern_frequency_matrix
        # weigh the frequency count values appropriately
        # exact matches: multiply frequency counts by 'consonance' weighting factor of 2
        exact_pattern_matches = freq_matrix[exact_pattern_ids].multiply(2)
        # very similar patterns: multiply frequency counts by 'consonance' weighting factor of 1.5
        very_similar_pattern_matches = freq_matrix[very_similar_pattern_ids].multiply(1.5)
        # similar patterns: no weighting factor applied
        similar_pattern_matches = freq_matrix[similar_pattern_ids]

        # concatenate the three weighted matrix slices generated above
        combined_weighted_pattern_count = vstack(
            [exact_pattern_matches, very_similar_pattern_matches, similar_pattern_matches], format='csr')
        # transpose and remove zeros
        transposed = combined_weighted_pattern_count.tocsc()
        cols_idx = np.unique(transposed.nonzero()[1])
        transposed_nonzero = transposed[:, np.unique(transposed.nonzero()[1])]
        # densify / convert to DataFrame
        weighted_pattern_frequencies = pd.DataFrame.sparse.from_spmatrix(transposed_nonzero).astype(
            "Sparse[float16, nan]")
        weighted_pattern_frequencies = weighted_pattern_frequencies.sparse.to_dense().astype('float16')
        # add indices
        weighted_pattern_frequencies.columns = cols_idx
        rows_idx = exact_pattern_ids + very_similar_pattern_ids + similar_pattern_ids
        weighted_pattern_frequencies.insert(0, 'pattern_id', rows_idx)
        # groupby to remove duplicate rows
        weighted_pattern_frequencies = weighted_pattern_frequencies.groupby(
            'pattern_id', as_index=True).agg('first').astype('float16')
        # append 'sliced_cols' data removed at start of this method
        weighted_pattern_frequencies = weighted_pattern_frequencies.join(sliced_cols, how='left')

        if weighting == 'double':
            # remove the non-numeric 'pattern' col to allow DataFrame-level application of 'prominence' weighting factor
            patterns = weighted_pattern_frequencies.pop('pattern')
            # also remove the 'prominence' col, which is to be the multiplier in these calculations
            prominence_factor = weighted_pattern_frequencies.pop('prominence')
            # apply 'prominence' weighting factor
            weighted_pattern_frequencies = weighted_pattern_frequencies.multiply(prominence_factor, axis=0)
            # re-append the removed columns
            weighted_pattern_frequencies.insert(0, 'prominence', prominence_factor)
            weighted_pattern_frequencies.insert(0, 'pattern', patterns)

        # print DataFrame of weighted patten frequencies
        # print("Weighted pattern count per tune:")
        # print(weighted_pattern_frequencies.head())
        # print(weighted_pattern_frequencies.info())

        # In the previous version we wrote this table to file to facilitate analysis of intermediate outputs,
        # but it proved to be a performance bottleneck and has been removed in this version.

        # drop patterns which are no longer necessary for the remaining workflow
        weighted_pattern_frequencies.drop(['pattern', 'prominence'], axis=1, inplace=True)
        # sum and sort the weighted frequency counts per tune
        score = weighted_pattern_frequencies.sum(axis=0)
        score.sort_values(ascending=False, inplace=True)
        # add human-readable tune titles as index
        labels = self._titles[score.index.tolist()]
        motif_results = pd.DataFrame(labels, columns=['title'])
        motif_results['score'] = score.tolist()
        # remove query tune from results
        motif_results = motif_results[motif_results['title'] != self.query_tune]

        # format final results table, normalise similarity score and append as additional column
        motif_results['score'] = motif_results['score'].round(decimals=3)
        rescaled = (
                           motif_results['score'] - motif_results['score'].min()) / (
                motif_results['score'].max() - motif_results['score'].min()
        )
        motif_results['normalized_score'] = rescaled.round(decimals=3)
        motif_results.set_index('title', drop=True, inplace=True)

        # print, save, and return results
        print(f"Final {weighting}-weighted 'motif' results:")
        print(motif_results)
        motif_results_dir = f"{self._out_dir}/{self.query_tune}"
        if not os.path.isdir(motif_results_dir):
            os.makedirs(motif_results_dir)
        motif_results_path = \
            f"{motif_results_dir}/{self.query_tune}_motif_results_{self.n}grams_{weighting}_weighted.csv"
        motif_results.to_csv(motif_results_path)
        return motif_results

    def run_similarity_search(self, weighting):
        """Run 'motif' similarity method."""
        print(f"Running {weighting}-weighted 'motif' similarity search...")
        return self._calculate_custom_weighted_motif_similarity_score(weighting)

    def _setup_precomputed_tfidf_similarity_matrix(self):
        pass

    def _lookup_precomputed_results(self):
        pass

    def _read_precomputed_tfidf_vector_similarity_results(self):
        pass

    def _extract_incipits_and_cadences(self):
        pass

    def _create_incipit_and_cadence_hamming_dist_matrix(self):
        pass

    def _calculate_incipit_and_cadence_edit_distance(self):
        pass

    def _incipit_and_cadence_flow_control(self):
        pass














