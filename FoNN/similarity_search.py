"""
Similarity_search.py reads outputs of pattern_extraction.py and takes them as inputs to three pattern-based
tune similarity methodologies: 'motif', 'incipit_and_cadence' and 'tfidf'.

'motif' is a novel multi-step similarity method:
first a representative pattern is extracted from a user-selected query tune via maximal tfidf.
All similar patterns to this search term pattern which occur in the corpus are detected via edit distance.
The number of similar patterns per tune in the corpus is calculated and returned as an indicator of tune-level
similarity.

'incipit and cadence' is an extended version of a traditional musicological incipit search.
Structurally-important subsequences incipit and cadence sequences are extracted from all tunes in the corpus and
compared via edit distance.

For both these methods, various edit distances (Levenshtein, Hamming, custom-weighted Hamming),
distance thresholds and custom substitution penalty matrices can be applied.

The final method, 'tfidf' is the Cosine similarity between TFIDF vectors of all tunes in the corpus. This similarity
matrix is calculated in pattern_extraction.py but this module contains methods to read the results and write excerpts to
 disc.
 """

# TODO: Simplify structure
# TODO: Implement local and/or global alignment methods

import os

from Levenshtein import distance
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.spatial.distance import hamming
from sklearn.metrics import pairwise_distances
import swifter
from tqdm import tqdm
import weighted_levenshtein

from FoNN._edit_dist_weights import substitution

# globals:
pd.options.mode.chained_assignment = None


class PatternSimilarity:

    """Reads input data, applies all three similarity methodologies and writes results to disc.

    Attributes:
        n -- length of representative pattern to extract from query tune for 'motif' method.
        _tune_lengths -- list the lengths of all tunes in corpus, used in 'motif' method results normalisation.
        feat_seq_path -- path to corpus feature sequence csv files outputted by feature_sequence_extraction_tools.py
        titles --  array listing titles of all tunes in corpus, outputted via pattern_extraction.py.
        tfidf_matrix -- sparse matrix storing tfidf values of all patterns (index) in all tunes (columns), outputted by
        pattern_extraction.py.
        patterns -- array of all unique local patterns extracted from corpus, outputted by pattern_extraction.py.
        query_tune -- title of query tune for input into similarity search.
        input_filter -- filters input sparse matrices by pattern length (n) for 'motif' method.
        Set input_filter = 'broad' if using Levenshtein distance, input_filter='narrow' if using Hamming distance.
        freq_matrix -- sparse matrix storing occurrences of all patterns (index) in all tunes (columns), outputted by
        pattern_extraction.py.
        out_dir -- top level directory from which to write similarity results and create appropriate subdirectories.
        feature -- input musical feature name. Must correspond to feature as listed in
        FoNN.pattern_extraction.NgramPatternCorpus.FEATURES and as explained in
        FoNN.feature_sequence_extraction_tools.Tune docstring.
        ngram_patterns_path -- path to write intermediate similar pattern output of 'motif' method, for reference in
        results analysis.
        tfidf_vector_cos_similarity_matrix -- matrix storing Cosine similarity between TFIDF vectors of all tunes in the
         corpus, outputted by pattern_extraction.py.
         cadences -- Boolean flag: if cadences == 'y', include both incipit and cadence subsequences in 'incipit and
         cadence' method inputs. If cadences == 'n', include incipit only.
         _incipits_and_cadences -- table holding incipit and cadence sequences for all tunes in corpus.
         _incipits_and_cadences_hamming_dist_matrix -- Hamming distance matrix between all incipit and cadence sequences
         _reformatted_patterns -- array holding string versions of 'motif' method patterns.
         weighted_hamming_threshold -- user-defined distance threshold value for custom-weighted Hamming distance, as
         applied in 'motif' method. Default value is 1.
    """

    def __init__(
            self,
            titles_path=None,
            patterns_path=None,
            n=None,
            query_tune=None,
            feat_seq_path=None,
            cadences=None,
            feature=None,
            input_filter=None
            ):

        """Initialize class instance. For explanation of args please see class instance attributes above."""

        self.n = n
        self._tune_lengths = self._calculate_tune_lengths(feat_seq_path)
        self.feat_seq_path = feat_seq_path
        self.titles = np.load(titles_path)
        self.tfidf_matrix = None
        self.patterns = pd.Series(np.load(patterns_path, allow_pickle=True))
        self.query_tune = query_tune
        self.input_filter = input_filter
        self.freq_matrix = None
        self.out_dir = None
        self.feature = feature
        self.ngram_patterns_path = None
        self.tfidf_vector_cos_similarity_matrix = None
        # TODO: allow 'cadences' value be altered without necessitating creation of a new object instance.
        self.cadences = cadences
        self._incipits_and_cadences = self._extract_incipits_and_cadences_from_all_tunes() if self.cadences else None
        self._incipits_and_cadences_hamming_dist_matrix = self._create_incipit_and_cadence_hamming_dist_matrix()
        self.weighted_hamming_threshold = 1

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):
        self._n = val

    @property
    def query_tune(self):
        return self._query_tune

    @query_tune.setter
    def query_tune(self, val):
        self._query_tune = val

    @property
    def tfidf_matrix(self):
        return self._tfidf_matrix

    @tfidf_matrix.setter
    def tfidf_matrix(self, path=None):

        """
        Loads and filters pattern TF-IDF matrix, stores data as _tfidf_matrix property.

        Args:
            path -- filepath to pattern TF-IDF matrix.
        """

        if path:
            # _filter_matrix() filtration mode here defaults to 'narrow' rather than 'broad'
            # 'broad' is only used in Levenshtein similarity inputs
            # In all other cases use 'narrow'
            self._tfidf_matrix = self._filter_matrix(self._load_matrix(path), mode='narrow')
        else:
            self._tfidf_matrix = None

    @property
    def freq_matrix(self):
        return self._freq_matrix

    @freq_matrix.setter
    def freq_matrix(self, path=None):

        """
        Loads and filters pattern occurrences matrix, stores data as _tfidf_matrix property.

        Args:
            path -- filepath to pattern occurrences matrix.
        """

        input_filter = self.input_filter
        if path and input_filter:
            assert input_filter == 'broad' or input_filter == 'narrow'
            self._freq_matrix = self._filter_matrix(self._load_matrix(path), mode=input_filter)
        else:
            self._freq_matrix = None

    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, path):

        """
        Sets out_path to write similarity results

        Args:
            path -- user-assigned top-level results directory.
        """

        if path:
            if not os.path.isdir(path):
                os.makedirs(path)
            self._out_dir = path

    @property
    def tfidf_vector_cos_similarity_matrix(self):
        return self._tfidf_vector_cos_similarity_matrix

    @tfidf_vector_cos_similarity_matrix.setter
    def tfidf_vector_cos_similarity_matrix(self, path):

        """
        Loads and tune TF-IDF Cosine similarity matrix, stores data as _tfidf_vector_cos_similarity_matrix property.

        Args:
            path -- filepath to Cosine similarity matrix.
        """

        if path:
            titles = self.titles
            x = y = len(titles)
            self._tfidf_vector_cos_similarity_matrix = np.memmap(path, dtype='float16', mode='r', shape=(x, y))

    def _load_matrix(self, in_dir):

        """
        Private method. Loads sparse matrix from file

        Args:
            in_dir -- directory containing sparse matrix file
        """

        titles = self.titles
        data_in = load_npz(in_dir).transpose(copy=False)
        data_out = pd.DataFrame.sparse.from_spmatrix(data_in, columns=titles).astype("Sparse[float16, nan]")
        # data_out.reset_index(inplace=True)
        # print(data_out.head(), data_out.info())
        return data_out

    def _filter_matrix(self, data, mode=None):

        """Private method. Filters sparse matrix via user-selectable mode.

        Args:
             data -- input sparse matrix
             mode -- 'broad' if using Levenshtein distance, 'narrow' if using Hamming distance.
        """

        assert mode == 'broad' or mode == 'narrow'
        n = self.n
        patterns = self.patterns
        # create object which will store output
        filtered_data = None
        # add helper column to input matrix
        data['pattern_len'] = patterns.swifter.apply(lambda x: len(x)).astype("int16")
        if mode == 'broad':
            # 'broad' mode filters input matrix by pattern length using Boolean mask, retaining patterns of length n+-1.
            data['bool_mask'] = (
                np.logical_and(
                    data['pattern_len'] <= n + 1, data['pattern_len'] >= n - 1)).astype("Sparse[float16, nan]")
            filtered_data = data[data['bool_mask'] == 1].astype("Sparse[float16, nan]")
            # drop Boolean column after filtering
            filtered_data.drop(columns=['bool_mask'], inplace=True)
        elif mode == 'narrow':
            # 'narrow' mode: retains patterns of length n only.
            filtered_data = data[data['pattern_len'] == n].astype("Sparse[float16, nan]")
        return filtered_data

    def _lookup_precomputed_results(self, data, ascending=False):

        """
        Private method. Looks up precalculated tune similarity matrix and returns top results for row 
        corresponding to query tune.
        
        Args:
            data -- input matrix
            ascending -- Boolean flag: 'True' sorts results in ascending order (for distance matrix inputs); 
            'False' sorts in descending order (for similarity matrix inputs).
        """
        
        titles = self.titles
        query_tune = self.query_tune
        # find query tune in titles array, print title and index
        query_tune_idx = int(np.where(titles == query_tune)[0])
        print(f"Query tune: {query_tune}. Index: {query_tune_idx}")
        # lookup similarity matrix via index of query tune; store results in DataFrame  
        results_raw = np.array(data[query_tune_idx], dtype='float16')
        similarity_results = pd.DataFrame(results_raw, index=titles, columns=[query_tune], dtype='float16')
        # sort results and return top 500
        similarity_results.sort_values(axis=0, ascending=ascending, by=query_tune, inplace=True)
        return similarity_results[:500]

    def _read_tfidf_vector_similarity_results(self):
        
        """
        Private method. Applies _lookup_precomputed_results() to TF-IDF vector Cosine similarity matrix and writes 
        to disc.
        """
        
        # read similarity matrix via _lookup_precomputed_results()
        tfidf_similarity_matrix = self.tfidf_vector_cos_similarity_matrix
        tfidf_results = self._lookup_precomputed_results(tfidf_similarity_matrix, ascending=False)
        # setup out paths, create subdirs if they do not already exist
        base_path = self.out_dir
        tfidf_results_path = f"{base_path}/tfidf_results"
        if not os.path.isdir(tfidf_results_path):
            os.makedirs(tfidf_results_path)
        # write output
        tfidf_results.to_csv(f"{tfidf_results_path}/tfidf_vector_cos_similarity.csv")
        return None

    def _extract_incipits_and_cadences_from_all_tunes(self):
        
        """
        Private method. Extracts incipit and cadence sequences from all tunes and stores output in DataFrame.
        Note: cadences attr can be used to select whether to include cadences or not:
        If cadences == 'y', both incipits and cadences are included; if == 'n' only incipits are included.
        """

        in_dir = self.feat_seq_path
        feature = self.feature

        # define incipit bar numbers. In Irish dance tunes, the first 4 bars can be assumed to represent the incipit. 
        incipit_bars = [1, 2, 3, 4]
        assert self.cadences == 'y' or self.cadences == 'n'
        # define cadence bars and add to list if cadences flag == 'y'
        cadence_bars = [7, 8] if self.cadences == 'y' else []
        bar_nums = incipit_bars + cadence_bars
        
        # setup output DataFrame and populate it by slicing incipits and cadences from corpus feature sequence csv files 
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
                    # slice relevant data from csv file by filtering bar_count column according to bar_nums
                    filtered_data = tune_data[tune_data['bar_count'].isin(bar_nums)]
                    incipits_and_cadences[file_name[:-4]] = filtered_data[feature]

        incipits_and_cadences.fillna(0, inplace=True)
        incipits_and_cadences = incipits_and_cadences.astype('float16').reset_index(drop=True).T
        return incipits_and_cadences
        
    def _create_incipit_and_cadence_hamming_dist_matrix(self):

        """
        Private method. Calculates Hamming distance matrix between all incipit and cadence sequences by applying 
        sklearn.metrics.pairwise_distances().
        """

        incipits_and_cadences = self._incipits_and_cadences
        # calculate distance matrix, convert to triangular format and force type to conserve memory
        incipits_and_cadences_hamming_dist = pairwise_distances(
            incipits_and_cadences, metric='hamming'
        ).astype('float16')
        
        return incipits_and_cadences_hamming_dist if incipits_and_cadences else None

    def _compute_incipit_and_cadence_results(self, metric):
        
        """Calculates and/or reads results of selected similarity/distance metric as applied to  incipit and cadence 
        input data.
        
        Args:
            metric -- select similarity of distance metric by name. Value can be 'levenshtein' (Levenshtein distance); 
            'hamming' (Hamming distance); or 'weighted hamming' (custom-weighted Hamming distance).
        """
        
        incipits_and_cadences = self._incipits_and_cadences
        hamming_dist_matrix = self._incipits_and_cadences_hamming_dist_matrix
        test_terms = incipits_and_cadences.to_numpy().tolist()
        # convert all incipit and cadence sequences to as required by Levensthein.distance()
        formatted_test_terms = [''.join([str(int(i)) for i in t]) for t in test_terms]
        # setup results object
        results = None

        if metric == 'levenshtein':
            query_tune = self.query_tune
            # look up query tune in incipits_and_cadences table.
            search_term = incipits_and_cadences.loc[query_tune]
            # compare against all other incipit and cadence sequences via Levensthein.distance()
            lev_results = incipits_and_cadences.swifter.apply(lambda row: distance(row, search_term), axis=1)
            # Store in DataFrame; format and slice, retaining top 500 results
            lev_results = pd.DataFrame(lev_results.sort_values(), dtype='int16')
            lev_results.columns = ['Levenshtein distance']
            results = lev_results.head(500)

        if metric == 'hamming':
            # look up Hamming distance matrix calculated via _lookup_precomputed_results()
            hamming_results = self._lookup_precomputed_results(hamming_dist_matrix, ascending=True)
            results = hamming_results

        if metric == 'weighted hamming':
            query_tune = self.query_tune
            # convert all incipit and cadence sequences to as required by weighted_levenshtein.levenshtein()
            search_term = ''.join(str(int(i)) for i in incipits_and_cadences.loc[query_tune].tolist())
            
            # compare against all other incipit and cadence sequences via weighted_levenshtein.levenshtein():
            # by only allowing substitution of elements, this Levenshtein implementation now functions as a Hamming 
            # distance.
            # weighted_levenshtein.levenshtein() takes a custom substitution matrix, which allows musically consonant 
            # substitutions to be penalised less heavily than dissonant substitutions. This matrix is defined in 
            # FoNN.edit_dist_weights.py and passed to the function call below as 'substitution' object.
            weighted_hamming_results = [weighted_levenshtein.levenshtein(
                search_term, t, substitute_costs=substitution
            ) for t in formatted_test_terms]
            # Store results in DataFrame; format and slice, retaining top 500 results
            results = pd.DataFrame(
                weighted_hamming_results, 
                index=incipits_and_cadences.index, 
                columns=['Hamming distance (weighted)'], 
                dtype='float16'
            )
            results.sort_values(by='Hamming distance (weighted)', inplace=True)
            results = results.head(500)

        return results

    def _incipit_and_cadence_flow_control(self):

        """Flow control to run all incipit and cadence metrics and write results to disc."""

        # setup paths
        base_path = self.out_dir
        out_path = f"{base_path}/structural_results"
        # run incipit and cadence metrics
        lev_results = self._compute_incipit_and_cadence_results('levenshtein')
        hamming_results = self._compute_incipit_and_cadence_results('hamming')
        weighted_hamming_results = self._compute_incipit_and_cadence_results('weighted hamming')

        # format output filenames to reflect whether cadences were included in input sequences
        cadence_flag = 'cadence' if self.cadences == 'y' else ''
        # write results to file
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        lev_results.to_csv(out_path + f"/incipit_{cadence_flag}_lev.csv")
        hamming_results.to_csv(out_path + f"/incipit_{cadence_flag}_ham.csv")
        weighted_hamming_results.to_csv(out_path + f"/incipit_{cadence_flag}_ham_w2.csv")
        return None

    def _extract_search_term_motifs_from_query_tune(self):

        """
        Extract representative motif(s) from query tune by maximal TF-IDF
         for use as search term(s) in 'motif' method.
         """

        title = self.query_tune
        data = self.tfidf_matrix
        n = self.n
        patterns = self.patterns

        # read query tune col from tfidf matrix to numpy array, retain only non-zero values.
        query_tune_data = data[title].dropna()
        content = query_tune_data.to_numpy()
        # save the content array index as an array
        idx = query_tune_data.index.to_numpy()
        # combine into DataFrame
        query_tune_data_densified = pd.DataFrame(index=idx, data=content, columns=[title], dtype='float16')
        # lookup index or indices corresponding to max tfidf value(s) in DataFrame
        max_tfidf_indices = query_tune_data_densified[query_tune_data_densified[title] ==
                                                      query_tune_data_densified[title].max()].index
        search_term_indices = [int(i) for i in max_tfidf_indices]
        # look up the patterns corresponding to these indices in the patterns array, return as search terms
        search_terms = patterns[search_term_indices]
        return search_terms

    def _filter_patterns(self):

        """
        Filter 'patterns' array per index of 'freq' pattern frequency matrix for input into motif detection
        methods below.
        """

        data = self.freq_matrix
        patterns = self.patterns
        # extract indices of 'freq' matrix.
        filter_indices = data.index.tolist()
        # slice patterns array according to these indices, return matching subset
        filtered_patterns = patterns[filter_indices]
        self.patterns = filtered_patterns
        return self.patterns

    def _find_similar_motifs(self, mode=None):

        """
        Calculates edit distance between search term motif and all candidate motif patterns in corpus.
        Detects similar patterns, writes them to file as intermediate results.

        Args:
            mode -- choice of edit distance to be applied, can be either 'levenshtein' (Levenshtein distance);
            'hamming' (Hamming distance) or 'weighted hamming' (custom-weighted Hamming distance)
        """

        search_terms = self._extract_search_term_motifs_from_query_tune()
        patterns = self.patterns

        assert mode == 'levenshtein' or mode == 'hamming' or mode == 'weighted hamming'

        if mode == 'levenshtein':
            # apply pairwise Levenshtein.distance() between search term pattern(s) and all other patterns in the (filtered)
            # patterns array
            results = pd.DataFrame([patterns.swifter.apply(distance, args=[term]) for term in search_terms]).T
            results.columns = [np.array2string(term) for term in search_terms]
            # filter results by edit distance threshold of 1. This retains only patterns within one element substitution
            # , deletion or insertion of the search term pattern.
            filtered = results[(results <= 1).any(1)]

        if mode == 'hamming':
            # apply pairwise scipy.spatial.distance.hamming() between search term pattern(s) and all other patterns in
            # the (filtered) patterns array
            results = pd.DataFrame([patterns.swifter.apply(hamming, args=[term]) for term in search_terms]).T
            results.columns = [np.array2string(term) for term in search_terms]
            # automatically set threshold to allow only one element substitution between search term and each candidate
            # pattern and filter results accordingly
            thresh = 1 / self.n
            filtered = results[(results <= thresh).any(1)]

        elif mode == 'weighted hamming':
            # convert search term and candidate patterns to string type for compatibility with
            # weighted_levenshtein.levenshtein()
            reformatted_search_terms = [''.join([str(int(i)) for i in t]) for t in search_terms]
            reformatted_patterns = self.reformatted_patterns
            # apply pairwise weighted_levenshtein.levenshtein() between search term pattern(s) and all other patterns in
            # the (filtered) patterns array. Note: as per incipit and cadence method, here we use the same custom
            # substitution penalty matrix defined in _edit_dist_weights.py.
            results = []
            for search_term in reformatted_search_terms:
                weighted_hamming_results = [weighted_levenshtein.levenshtein(
                    search_term,
                    pattern,
                    substitute_costs=substitution
                ) for pattern in reformatted_patterns]
                results.append(weighted_hamming_results)

            # create results DataFrame and format axes
            results = pd.DataFrame(results).T
            col_names = [np.array2string(term) for term in search_terms]
            results.columns = col_names
            results.index = patterns.index

            # set edit distance filter threshold to 1 and filter results accordingly
            thresh = self.weighted_hamming_threshold
            filtered = results[(results <= thresh).any(1)]

        # retain the indices of the similar patterns discovered-- these will be used to lookup and count their
        # occurrences per tune in the 'freq' matrix, in the final step of the 'motif' method.
        similar_pattern_indices = filtered.index.to_list()
        similar_patterns = patterns[similar_pattern_indices]
        filtered['patterns'] = similar_patterns
        # setup paths and write similar patterns to csv
        base_path = self.out_dir
        out_path = f"{base_path}/{mode}_patterns"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        filtered.to_csv(f"{out_path}/{mode}_patterns.csv")
        return similar_pattern_indices

    @staticmethod
    def _calculate_tune_lengths(target_dir):

        """
        Helper function to calculate the length of all feature sequences in the input corpus.
        this data is used in normalisation of 'motif' results.
        """

        results = {}
        for file_name in os.listdir(target_dir):
            if file_name.endswith('.csv'):
                file_path = f"{target_dir}/{file_name}"
                tune_title = file_name[:-4]
                with open(file_path) as content:
                    counter = len(content.readlines()) - 1
                results[tune_title] = counter

        file_length = pd.DataFrame()
        file_length['title'] = results.keys()
        file_length['length'] = results.values()
        file_length.set_index('title', drop=True, inplace=True)

        results = file_length.T
        results = results.rename_axis(None, axis=1)

        return results

    def _find_tunes_containing_similar_motifs(self, mode):

        # TODO: docstring; comments

        similar_pattern_indices = self._find_similar_motifs(mode=mode)
        freq_matrix = self.freq_matrix
        # filter freq matrix
        filtered = freq_matrix.loc[similar_pattern_indices].drop(columns=['pattern_len']).dropna(axis=1, how='all')
        filtered.fillna(0, inplace=True)
        return filtered

    def _count_similar_motif_occurrences_per_tune(self, data, normalize):

        # TODO: docstring; comments

        if normalize:
            normalization_table = self._tune_lengths
            normalized = normalization_table[normalization_table.columns.intersection(data.columns)].squeeze()
            results = (data.sum(axis=0) / normalized).round(decimals=3).to_frame().reset_index()
            results.columns = ['title', f'normalized_count']
            results = results.head(500)
        else:
            results = data.sum(axis=0).to_frame().reset_index()
            results.columns = ['title', 'count']
            results = results.head(500)

        return results

    def _run_motif_similarity_metrics(self, data, normalize=True, mode=None):

        # TODO: docstring; comments

        assert mode == 'levenshtein' or mode == 'hamming' or mode == 'weighted hamming'
        results = self._count_similar_motif_occurrences_per_tune(data, normalize=normalize)
        # Reformat / rename cols and sort:
        results.sort_values(by=results.columns[1], ascending=False, inplace=True)
        results.reset_index(inplace=True, drop=True)

        norm = '_normalized' if normalize else ''
        base_path = self.out_dir
        out_path = f"{base_path}/{mode}_results"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        csv_out_path = f"{out_path}/{mode}{norm}.csv"
        results.to_csv(csv_out_path)

        return None

    def run_similarity_search(self, mode=None):

        # TODO: docstring; comments

        if mode == 'levenshtein' or mode == 'weighted hamming' or mode == 'hamming':
            print(mode)
            self._filter_patterns()
            similar_tunes = self._find_tunes_containing_similar_motifs(mode)
            self._run_motif_similarity_metrics(similar_tunes, normalize=True, mode=mode)
            self._run_motif_similarity_metrics(similar_tunes, normalize=False, mode=mode)

        elif mode == 'incipit':
            print(mode)
            self._incipit_and_cadence_flow_control()

        elif mode == 'tfidf':
            print(mode)
            self._read_tfidf_vector_similarity_results()

        return None

