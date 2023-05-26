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

# TODO: Update docstrings; test
# TODO: Implement local and/or global alignment methods

import os

from Levenshtein import distance
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import pairwise_distances
import swifter
from tqdm import tqdm
import weighted_levenshtein

import FoNN.pattern_extraction
from FoNN._edit_dist_weights import substitution

# globals:
pd.options.mode.chained_assignment = None
FEATURES = FoNN.pattern_extraction.NgramPatternCorpus.FEATURES
LEVELS = FoNN.pattern_extraction.NgramPatternCorpus.LEVELS


class PatternSimilarity:

    """Reads input data, applies all three similarity methodologies and writes results to disc.

    Attributes:

        query_tune -- title of query tune for input into similarity search.
        feature -- input musical feature name. Must correspond to feature as listed in
                   FoNN.pattern_extraction.NgramPatternCorpus.FEATURES and as explained in
                   FoNN.feature_sequence_extraction_tools.Tune docstring.
        n -- length of representative pattern to extract from query tune for 'motif' method.
        _titles --  array listing titles of all tunes in corpus, outputted via pattern_extraction.py.
        _patterns -- array of all unique local patterns extracted from corpus, outputted by pattern_extraction.py.
        freq_matrix -- sparse matrix storing occurrences of all patterns (index) in all tunes (columns), outputted by
                       pattern_extraction.py.
        _feat_seq_data_path -- path to corpus feature sequence csv files outputted by 
                               feature_sequence_extraction_tools.py
        _tfidf_matrix_path -- path to sparse matrix storing tfidf values of all patterns (index) in all tunes (columns),
                              outputted by pattern_extraction.py.
        _tfidf_matrix -- data content of file at _tfidf_matrix_path.
        _tfidf_vector_cos_similarity_matrix_path -- path to matrix file storing Cosine similarity between TFIDF vectors 
                                                    of all tunes in the corpus, as outputted via 
                                                    FoNN.pattern_extraction.NgramPatternCorpus.
        _tfidf_vector_cos_similarity_matrix -- data content of file at _tfidf_vector_cos_similarity_matrix_path.
        _tune_lengths -- list the lengths of all tunes in corpus, used in 'motif' method results normalisation.
        _reformatted_patterns -- array holding string versions of 'motif' method patterns.
        _out_dir -- top level directory from which to write similarity results and create appropriate subdirectories.
    """

    def __init__(
            self,
            corpus_path=None,
            level=None,
            n=None,
            query_tune=None,
            feature=None
            ):

        """Initialize PatternSimilarity class instance.

        Args:
            corpus_path -- root dir of music corpus under investigation.
            level -- Level of granularity of corpus data. Must correspond to one of the three levels
                    at which FoNN can extract feature sequence and pattern data, per global LEVELS constant:
                    1. 'note' (note-level)
                    2. 'accent' (accent-level)
                    3. 'duration_weighted' (duration-weighted note-level)
            query_tune -- Title of query tune. Must correspond to MIDI filename of a tune in the corpus.
            feature -- Musical feature under investigation. Must correspond to a feature name listed in global LEVELS
                       constant (FoNN.pattern_extraction.NgramPatternCorpus.LEVELS).
            """

        # set paths to private files containing corpus input data
        _titles_path = f"{corpus_path}/pattern_corpus/{level}/titles.npy"
        _tfidf_matrix_path = f"{corpus_path}/pattern_corpus/{level}/tfidf_matrix.npz"
        _patterns_path = f"{corpus_path}/pattern_corpus/{level}/patterns.npy"
        _freq_matrix_path = f"{corpus_path}/pattern_corpus/{level}/freq_matrix.npz"
        _feat_seq_data_path = f"{corpus_path}/feature_sequence_data/{level}"
        _tfidf_vector_cos_similarity_matrix_path = f"{corpus_path}/pattern_corpus/{level}/tfidf_vector_cos_sim.mm"
        # attrs
        self._titles = np.load(_titles_path)
        self.query_tune = query_tune    # property
        self.n = n                      # property
        self.feature = feature          # property
        self._patterns = pd.Series(np.load(_patterns_path, allow_pickle=True).tolist())
        self._freq_matrix = self._filter_matrix(self._load_sparse_matrix(_freq_matrix_path), mode='broad')
        self._feat_seq_data_path = _feat_seq_data_path
        self._tfidf_matrix_path = _tfidf_matrix_path
        self._tfidf_matrix = self._filter_matrix(self._load_sparse_matrix(_tfidf_matrix_path), mode='narrow')
        self._tfidf_vector_cos_similarity_matrix_path = _tfidf_vector_cos_similarity_matrix_path
        self._tfidf_vector_cos_similarity_matrix = self._setup_tfidf_vector_cos_similarity_matrix()
        self._tune_lengths = self._calculate_tune_lengths()
        self._reformatted_patterns = None
        self._out_dir = f"{corpus_path}/similarity_results/{level}"
        # create out_dir if it does not already exist
        if not os.path.isdir(self._out_dir):
            os.makedirs(self._out_dir)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):

        """
        Set 'n' property representing length of search term pattern(s) to be extracted from query tune when using
        'motif' similarity mode.
        """

        # check value is within the max and min pattern lengths allowable
        assert 3 <= val <= 12
        self._n = val

    @property
    def query_tune(self):
        return self._query_tune

    @query_tune.setter
    def query_tune(self, val):
        """Set 'query_tune' property representing title of query tune."""
        # query tune name must match the title of a tune in the corpus.
        assert val in self._titles
        self._query_tune = val

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, val):
        """Set 'feature' property, the musical feature for which pattern data has been extracted"""
        # feature name must match one of the 15 features provided by FoNN, as listed in global FEATURES constant.
        assert val in FEATURES
        self._feature = val

    def _setup_tfidf_vector_cos_similarity_matrix(self):
        """Load TF-IDF Cosine similarity matrix, store data as _tfidf_vector_cos_similarity_matrix property"""
        titles = self._titles
        matrix_path = self._tfidf_vector_cos_similarity_matrix_path
        x = y = len(titles)
        return np.memmap(matrix_path, dtype='float16', mode='r', shape=(x, y))

    def _load_sparse_matrix(self, in_dir):

        """
        Loads sparse matrix from file

        Args:
            in_dir -- directory containing sparse matrix file
        """

        titles = self._titles
        data_in = load_npz(in_dir).transpose(copy=False)
        data_out = pd.DataFrame.sparse.from_spmatrix(data_in, columns=titles).astype("Sparse[float16, nan]")
        # data_out.reset_index(inplace=True)
        # print(data_out.head(), data_out.info())
        return data_out

    def _filter_matrix(self, data, mode=None):

        """Filters sparse matrix via user-selectable mode.

        Args:
             data -- input sparse matrix
             mode -- 'broad' filters input matrix to retain rows for all patterns of length n+-1;
                     'narrow' filters input matrix to retain rows for all patterns of length n.
        """

        assert mode == 'broad' or mode == 'narrow'
        n = self.n
        patterns = self._patterns
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
        Looks up precalculated tune similarity matrix and returns top results for row 
        corresponding to query tune.
        
        Args:
            data -- input matrix
            ascending -- Boolean flag: 'True' sorts results in ascending order (for distance matrix inputs); 
            'False' sorts in descending order (for similarity matrix inputs).
        """
        
        titles = self._titles
        query_tune = self.query_tune
        # find query tune in titles array, print title and index
        query_tune_idx = int(np.where(titles == query_tune)[0])
        print(f"Query tune: {query_tune}.")
        # lookup similarity matrix via index of query tune; store results in DataFrame  
        results_raw = np.array(data[query_tune_idx], dtype='float16')
        similarity_results = pd.DataFrame(results_raw, index=titles, columns=[query_tune], dtype='float16')
        # sort results and return top 500
        similarity_results.sort_values(axis=0, ascending=ascending, by=query_tune, inplace=True)
        return similarity_results[:500]

    def _read_precomputed_tfidf_vector_similarity_results(self):
        
        """
        Applies _lookup_precomputed_results() to TF-IDF vector Cosine similarity matrix and writes 
        to disc.
        """

        tfidf_similarity_matrix = self._tfidf_vector_cos_similarity_matrix
        # read tfidf similarity matrix via _lookup_precomputed_results()
        tfidf_results = self._lookup_precomputed_results(tfidf_similarity_matrix, ascending=False)
        # setup out paths, create subdirs if they do not already exist
        tfidf_results_path = f"{self._out_dir}/tfidf_results/{self.query_tune}"
        if not os.path.isdir(tfidf_results_path):
            os.makedirs(tfidf_results_path)
        # write output
        tfidf_results.to_csv(f"{tfidf_results_path}/tfidf_vector_cos_similarity.csv")
        return tfidf_results

    def _extract_incipits_and_cadences(self, cadences=True):
        
        """
        Extracts incipit and cadence sequences from all tunes and stores output in DataFrame.

        Args:
            cadences -- Boolean flag used to select whether to include cadences or not:
            If True, both incipits and cadences are included in input data; if False only incipits are included.
        """

        in_dir = self._feat_seq_data_path
        feature = self.feature

        # define incipit bar numbers. In Irish dance tunes, the first 4 bars can be assumed to represent the incipit. 
        incipit_bars = [1, 2, 3, 4]
        # define cadence bars and add to list if cadences flag is True
        cadence_bars = [7, 8] if cadences else []
        bar_nums = incipit_bars + cadence_bars
        
        # setup output DataFrame and populate by slicing incipit (and optional cadence) subsequences from corpus feature
        # sequence csv files
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
                    # slice relevant data from csv file by filtering 'bar_count' column by bar_nums
                    filtered_data = tune_data[tune_data['bar_count'].isin(bar_nums)]
                    incipits_and_cadences[file_name[:-4]] = filtered_data[feature]

        incipits_and_cadences.fillna(0, inplace=True)
        incipits_and_cadences = incipits_and_cadences.astype('float16').reset_index(drop=True).T
        return incipits_and_cadences
        
    def _create_incipit_and_cadence_hamming_dist_matrix(self):

        """
        Calculates Hamming distance matrix between all incipit and cadence sequences by applying 
        sklearn.metrics.pairwise_distances().
        """

        incipits_and_cadences = self._incipits_and_cadences
        # calculate distance matrix, convert to triangular format and force type to conserve memory
        incipits_and_cadences_hamming_dist = pairwise_distances(
            incipits_and_cadences, metric='hamming'
        ).astype('float16')
        
        return incipits_and_cadences_hamming_dist if incipits_and_cadences else None

    def _calculate_incipit_and_cadence_edit_distance(self, metric):
        
        """Calculates and/or reads results of selected similarity/distance metric as applied to  incipit and cadence 
        input data.
        
        Args:
            metric -- select similarity of distance metric by name. Value can be 'levenshtein' (Levenshtein distance); 
            'hamming' (Hamming distance); or 'weighted hamming' (custom-weighted Hamming distance).
        """
        
        incipits_and_cadences = self._incipits_and_cadences
        test_terms = incipits_and_cadences.to_numpy().tolist()
        # convert all incipit and cadence sequences to as required by external Levensthein.distance() method.
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
            hamming_dist_matrix = self._create_incipit_and_cadence_hamming_dist_matrix()
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
            # FoNN.edit_dist_weights.py and passed to the function call below via substitute_costs keyword arg.
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

    def _incipit_and_cadence_flow_control(self, cadences=True):

        """Flow control to run all incipit and cadence metrics and write results to disc.

        Args:
            cadences -- Boolean flag: if cadences == True, include both incipit and cadence subsequences in 'incipit and
            cadence' method inputs. If cadences == False, include incipit only.
        """

        # setup paths
        out_path = f"{self._out_dir}/{self.query_tune}/structural_results"
        # extract incipit and cadence subsequences from feature sequence data
        self.cadences = cadences
        self._incipits_and_cadences = self._extract_incipits_and_cadences(cadences=cadences)
        # run all three incipit and cadence edit distance metrics: Levenshtein distance, Hamming distance, and custom-
        # weighted Hamming distance.
        lev_results = self._calculate_incipit_and_cadence_edit_distance('levenshtein')
        hamming_results = self._calculate_incipit_and_cadence_edit_distance('hamming')
        weighted_hamming_results = self._calculate_incipit_and_cadence_edit_distance('weighted hamming')

        # format output filenames to reflect whether cadences were included in input sequences
        cadence_flag = '_cadence' if self.cadences else ''
        # write results to file
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        lev_results.to_csv(out_path + f"/incipit{cadence_flag}_levenshtein.csv")
        hamming_results.to_csv(out_path + f"/incipit{cadence_flag}_hamming.csv")
        weighted_hamming_results.to_csv(out_path + f"/incipit{cadence_flag}_custom_weighted_hamming.csv")
        return None

    def _extract_search_term_motifs_from_query_tune(self):

        """
        Extract representative motif(s) from query tune by maximal TF-IDF
        for use as search term(s) in 'motif' method.
         """

        title = self.query_tune
        data = self._tfidf_matrix
        patterns = self._patterns

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
       self.freq pattern occurrences matrix is filtered on initialisation via self._filter_matrix().
       This method applies corresponding filtration to self._patterns array.
        """

        data = self._freq_matrix
        patterns = self._patterns
        # extract indices of 'freq' matrix.
        filter_indices = data.index.tolist()
        # slice patterns array according to these indices, return matching subset
        filtered_patterns = patterns[filter_indices]
        self._patterns = filtered_patterns
        return self._patterns

    def _run_motif_edit_distance_calculations(self):

        """
        Calculates edit distance between search term motif(s) and all corpus patterns stored in self._patterns.
        Detect patterns within a Levenshtein distance threshold of 1, write them to file as intermediate results,
        and return their indices.

        Args:
            mode -- choice of edit distance to be applied, can be either 'levenshtein' (Levenshtein distance);
            'hamming' (Hamming distance) or 'weighted hamming' (custom-weighted Hamming distance)"""

        search_terms = self._extract_search_term_motifs_from_query_tune()
        patterns = self._patterns

        # apply pairwise Levenshtein.distance() between search term pattern(s) and all other patterns in the (filtered)
        # patterns array
        results = pd.DataFrame([patterns.swifter.apply(distance, args=[term]) for term in search_terms]).T
        results.columns = [np.array2string(term) for term in search_terms]
        # filter results by edit distance threshold of 1. This retains only patterns within one element substitution
        # , deletion or insertion of the search term pattern.
        filtered = results[(results <= 1).any(1)]
        # retain the indices of the similar patterns discovered-- these will be used to lookup the 'freq' matrix and
        # count similar pattern occurrences per tune
        similar_pattern_indices = filtered.index.to_list()
        similar_patterns = patterns[similar_pattern_indices]
        filtered['patterns'] = similar_patterns
        # setup paths and write similar patterns to csv

        out_path = f"{self._out_dir}/motif_similar_patterns"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        filtered.to_csv(f"{out_path}/motif_similar_patterns.csv")
        return similar_pattern_indices

    def _calculate_tune_lengths(self):

        """
        Helper function to calculate the length of all feature sequences in the input corpus.
        this data is used in normalisation of 'motif' results.

        Args:
            target_dir -- directory holding a corpus of feature sequence csv files as outputted by
            feature_sequence_extraction_tools.py.
        """

        target_dir = self._feat_seq_data_path
        results = {}
        # read all csv files in target dir
        for file_name in os.listdir(target_dir):
            if file_name.endswith('.csv'):
                file_path = f"{target_dir}/{file_name}"
                tune_title = file_name[:-4]
                # calculate length of each file
                with open(file_path) as content:
                    counter = len(content.readlines()) - 1
                results[tune_title] = counter
        # store file lengths in DataFrame and format output
        file_length = pd.DataFrame()
        file_length['title'] = results.keys()
        file_length['length'] = results.values()
        file_length.set_index('title', drop=True, inplace=True)
        results = file_length.T
        results = results.rename_axis(None, axis=1)
        return results

    def _find_similar_patterns_and_their_occurrences(self):

        """
        Calls _run_edit_distance_calculations() method to find all similar patterns to the search term(s)
        . Occurrences of these patterns are the looked up in self.freq matrix and returned.
        """

        # find all patterns in self._patterns which are an edit distance of 1 or less from the search term pattern(s)
        # extracted from the query tune.
        similar_pattern_indices = self._run_motif_edit_distance_calculations()
        freq_matrix = self._freq_matrix
        # look up the rows corresponding to these patterns in self.freq pattern occurrences matrix
        filtered = freq_matrix.loc[similar_pattern_indices].drop(columns=['pattern_len']).dropna(axis=1, how='all')
        filtered.fillna(0, inplace=True)
        return filtered

    def _count_similar_pattern_occurrences_per_tune(self, data, normalize=True):

        """
        Sums the occurrences of each pattern in the filtered occurrences matrix outputted by
        self._read_similar_pattern_occurrences_from_matrix()

        Args:
            data -- pattern occurrence matrix object as outputted by
            self._find_similar_patterns_and_their_occurrences()
            normalize -- Boolean flag: if True, output pattern occurrence counts for each tune are normalized by the
            length of the tune as stored in self._tune_lengths.
        """

        if normalize:
            # normalization process
            # read self._tune_lengths
            normalization_table = self._tune_lengths
            # map tune lengths to corresponding columns in matrix
            normalized = normalization_table[normalization_table.columns.intersection(data.columns)].squeeze()
            # normalize
            results = (data.sum(axis=0) / normalized).round(decimals=3).to_frame().reset_index()
            results.columns = ['title', f'normalized_count']
            results = results.head(500)
        else:
            # if normalization is not desired, raw pattern occurrence counts are returned
            results = data.sum(axis=0).to_frame().reset_index()
            results.columns = ['title', 'count']
            results = results.head(500)

        return results

    def _calculate_motif_similarity(self, data, normalize=True):

        """
        Flow control for motif similarity method.

        Args:
            data -- pattern occurrence matrix object, passes to self._count_similar_pattern_occurrences_per_tune() call.
            normalize -- Boolean flag: if True, putput pattern occurrence counts are normalized by the length of the
            tune. Passes to self._count_similar_pattern_occurrences_per_tune().
        """

        results = self._count_similar_pattern_occurrences_per_tune(data, normalize=normalize)
        # Reformat / rename cols and sort:
        results.sort_values(by=results.columns[1], ascending=False, inplace=True)
        results.reset_index(inplace=True, drop=True)
        # format filenames and write outputs to file
        norm = 'normalized_' if normalize else ''
        out_path = f"{self._out_dir}/motif_results"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        csv_out_path = f"{out_path}/{norm}motif_results.csv"
        results.to_csv(csv_out_path)

        return None

    def run_similarity_search(self, mode=None):

        """Top-level flow control to select and run FoNN's three similarity search methods: 'motif',
        'incipit_and_cadence' [and cadence], and 'tfidf'.

        Args:
             mode -- selects between three modes above. If 'incipit_and_cadence' is selected, 6 methodological
             variations will run in parallel, generating similarity results via Levenshtein, Hamming and custom-weighted
              Hamming distance on incipit-only and incipit & cadence inputs.
        """

        if mode == 'motif':
            # run 'motif' similarity methods
            print(f'Similarity search mode: {mode}')
            self._filter_patterns()
            self._reformatted_patterns = [''.join([str(int(i)) for i in t]) for t in self._patterns]
            similar_tunes = self._find_similar_patterns_and_their_occurrences()
            self._calculate_motif_similarity(similar_tunes, normalize=True)

        elif mode == 'incipit_and_cadence':
            # run incipit and cadence similarity methods
            print(f'Similarity search mode: {mode}')
            self._incipit_and_cadence_flow_control()

        elif mode == 'tfidf':
            # run 'tfidf' similarity methods; format and print output
            print(f'Similarity search mode: {mode.upper()}')
            tfidf_res = self._read_precomputed_tfidf_vector_similarity_results()
            tfidf_res = tfidf_res.rename(columns={f"{self.query_tune}": "Cosine similarity"})
            print(tfidf_res.head())

        return None
