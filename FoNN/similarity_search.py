"""
similarity_search.py contains PatternSimilarity class which implements three pattern-based tune similarity
methodologies: 'motif', 'incipit_and_cadence' and 'tfidf'.
For a user-selectable query tune, users can apply these methods to search the corpus for similar tunes.

Input data must fist be processed through FoNN's feature sequence and pattern extraction pipeline
(via feature_sequence_extraction_tools.py and pattern_extraction.py) to generate and populate the required 'pattern
corpus' data stored in '../[corpus]/pattern_corpus' dir.

Similarity methodologies applied in PatternSimilarity:
1. 'motif':
First a representative pattern is extracted from a user-selected query tune via maximal tfidf.
All similar patterns to this search term pattern which occur in the corpus are detected via edit distance.
The number of similar patterns per tune in the corpus is calculated, normalised by tune length, and returned as a
tune-similarity metric.

2. 'incipit and_cadence':
An extended version of a traditional musicological incipit search.
Structurally-important subsequences incipit and cadence subsequences are extracted from all tunes in the corpus and
compared via pairwise edit distance against the query tune. Users can select from three available edit distance metrics:
Levenshtein distance; Hamming distance; and a custom-weighted Hamming distance in which musically-consonant
substitutions are penalised less than dissonant substitutions. The edit distance output is taken as a
tune-dissimilarity metric.

3. 'tfidf':
The Cosine similarity between TFIDF vectors of all tunes in the corpus. This similarity
matrix is calculated via pattern_extraction.py but this module contains methods to read and format the results and write
to disc.
 """

# TODO: (long-term) Implement local and/or global alignment methods


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

import FoNN.pattern_extraction
from FoNN._edit_dist_weights import substitution, insertion, deletion

# globals:
pd.options.mode.chained_assignment = None
swifter.set_defaults(progress_bar=False)
FEATURES = FoNN.pattern_extraction.NgramPatternCorpus.FEATURES
LEVELS = FoNN.pattern_extraction.NgramPatternCorpus.LEVELS


class PatternSimilarity:

    """
    Read input 'pattern corpus' data, apply selected similarity method(s) to query tune and write results to disc.

    Attributes:

        _titles --  array listing titles of all tunes in corpus, outputted via pattern_extraction.py.
        query_tune -- title of query tune for input into similarity search.
        n -- length of representative pattern to extract from query tune for 'motif' method.
        feature -- input musical feature name. Must correspond to feature name as listed and explained in
                   FoNN.feature_sequence_extraction_tools.Tune docstring & FoNN README.md.
        _patterns -- array of all unique local patterns extracted from corpus, outputted by pattern_extraction.py.
        _pattern_occurrences_matrix -- sparse matrix storing occurrences of all patterns (index) in all tunes (columns),
                                       outputted by FoNN.pattern_extraction.NgramPatternCorpus.
        _feat_seq_data_path -- path to corpus feature sequence csv files outputted by 
                               FoNN.feature_sequence_extraction_tools.Corpus
        _tfidf_matrix_path -- path to sparse matrix storing tfidf values of all patterns (index) in all tunes (columns),
                              outputted by FoNN.pattern_extraction.NgramPatternCorpus.
        _tfidf_matrix -- data content of file at _tfidf_matrix_path.
        _tfidf_vector_cos_similarity_matrix_path -- path to matrix file storing Cosine similarity between TFIDF vectors 
                                                    of all tunes in the corpus, as outputted via 
                                                    FoNN.pattern_extraction.NgramPatternCorpus.
        _tfidf_vector_cos_similarity_matrix -- data content of file at _tfidf_vector_cos_similarity_matrix_path.
        _tune_lengths -- list the lengths of all tunes in corpus, used in 'motif' method results normalisation.
        _out_dir -- top level directory from which to write similarity results and create appropriate subdirectories.
    """
    
    EDIT_DIST_METRICS = {
        'levenshtein': 'Levenshtein distance',
        'hamming': 'Hamming distance',
        'custom_weighted_hamming': 'custom-weighted Hamming distance',
        'custom_weighted_levenshtein': 'custom-weighted Levenshtein distance'
    }

    def __init__(
            self,
            corpus_path=None,
            level=None,
            n=None,
            query_tune=None,
            feature=None
            ):

        """
        Initialize PatternSimilarity class instance.

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
        _pattern_occurrences_matrix_path = f"{corpus_path}/pattern_corpus/{level}/freq_matrix.npz"
        _feat_seq_data_path = f"{corpus_path}/feature_sequence_data/{level}"
        _tfidf_vector_cos_similarity_matrix_path = f"{corpus_path}/pattern_corpus/{level}/tfidf_vector_cos_sim.mm"
        # attrs
        self._titles = np.load(_titles_path)
        self.query_tune = query_tune  # property
        self.n = n  # property
        self.feature = feature  # property
        self._patterns = pd.Series(np.load(_patterns_path, allow_pickle=True).tolist())
        self._pattern_occurrences_matrix = self._filter_matrix(
            data=self._load_sparse_matrix(_pattern_occurrences_matrix_path),
            mode='broad'
        )
        self._feat_seq_data_path = _feat_seq_data_path
        self._tfidf_matrix_path = _tfidf_matrix_path
        self._tfidf_matrix = self._filter_matrix(self._load_sparse_matrix(_tfidf_matrix_path), mode='narrow')
        self._tfidf_vector_cos_similarity_matrix_path = _tfidf_vector_cos_similarity_matrix_path
        self._tfidf_vector_cos_similarity_matrix = self._setup_tfidf_vector_cos_similarity_matrix()
        self._tune_lengths = self._calculate_tune_lengths()
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
        'motif' similarity mode. 'n' can be any integer value between 3 and 12.
        """

        assert 3 <= val <= 12
        self._n = val

    @property
    def query_tune(self):
        return self._query_tune

    @query_tune.setter
    def query_tune(self, val):
        """Set 'query_tune' title. Note: must match the filename of a tune in the corpus (excluding filetype suffix)."""
        assert val in self._titles
        self._query_tune = val

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, val):

        """
        Set 'feature' property, the musical feature under investigation. Default is 'diatonic_scale_degree'
        Note: feature name must match one of the 15 features provided by FoNN, as listed in global FEATURES constant.
        """

        assert val in FEATURES
        self._feature = val

    def _setup_tfidf_vector_cos_similarity_matrix(self):
        """Load TF-IDF Cosine similarity matrix, store as PatternSimilarity._tfidf_vector_cos_similarity_matrix."""
        titles = self._titles
        matrix_path = self._tfidf_vector_cos_similarity_matrix_path
        x = y = len(titles)
        return np.memmap(matrix_path, dtype='float16', mode='r', shape=(x, y))

    def _load_sparse_matrix(self, in_dir):

        """
        Load sparse matrix from file

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

        """
        Filter sparse matrix via user-selectable mode.

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

    def _read_precomputed_tfidf_vector_similarity_results(self):

        """Apply _lookup_precomputed_results() to TF-IDF vector Cosine similarity matrix and write output to disc."""

        tfidf_similarity_matrix = self._tfidf_vector_cos_similarity_matrix
        # read tfidf similarity matrix via _lookup_precomputed_results()
        tfidf_results = self._lookup_precomputed_results(tfidf_similarity_matrix, ascending=False)
        # format and print output
        tfidf_results = tfidf_results.rename(columns={f"{self.query_tune}": "Cosine similarity"})
        print(tfidf_results.head())
        # setup out paths, create subdirs if they do not already exist
        tfidf_results_path = f"{self._out_dir}/{self.query_tune}/tfidf_results"
        if not os.path.isdir(tfidf_results_path):
            os.makedirs(tfidf_results_path)
        # write output
        tfidf_results.to_csv(f"{tfidf_results_path}/tfidf_vector_cos_similarity.csv")
        return tfidf_results

    def _extract_incipits_and_cadences(self):
        
        """
        Extract incipit and cadence sequences from all tunes and store output in single corpus-level DataFrame.

        Args:
            cadences -- Boolean flag used to select whether to include cadences or not:
            If True, both incipits and cadences are included in input data; if False only incipits are included.
        """

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
            (Levenshtein distance); 'hamming' (Hamming distance); or 'custom_weighted_hamming' 
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
            results = hamming_results

        if edit_dist_metric == 'custom_weighted_hamming':
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
                search_term,
                t,
                substitute_costs=substitution
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

    def _incipit_and_cadence_flow_control(self, edit_dist_metric='levenshtein'):

        """
        Flow control: run 'incipit and cadence' similarity method and write results to disc.

        Args:
            edit_dist_metric -- select edit distance metric. Value can be 'levenshtein' (Levenshtein distance); 
            'hamming' (Hamming distance); or 'custom_weighted_hamming' (custom-weighted Hamming distance), as defined in 
            PatternSimilarity.EDIT_DIST_METRICS.
        """

        # setup paths
        out_path = f"{self._out_dir}/{self.query_tune}/incipit_and_cadence_results"
        # extract incipit and cadence subsequences from feature sequence data
        self._incipits_and_cadences = self._extract_incipits_and_cadences()
        # apply selected edit distance metric to calculate pairwise distance between incipit and cadence of query tune 
        # vs all other tunes.
        results = self._calculate_incipit_and_cadence_edit_distance(edit_dist_metric)
        print(results.head())
        # format output filenames and write results to file
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        results.to_csv(out_path + f"/incipit_and_cadence_{edit_dist_metric}.csv")
        return None

    def _extract_search_term_motifs_from_query_tune(self):

        """
        Extract representative motif(s) from query tune by maximal TF-IDF for use as search term(s) in 'motif' method.
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
       Filter self._patterns array, corresponding to filtration of self._pattern_occurrences_matrix, which is
       automatically applied on initialization via self._filter_matrix().
        """

        data = self._pattern_occurrences_matrix
        patterns = self._patterns
        # extract indices of 'freq' matrix.
        filter_indices = data.index.tolist()
        # slice patterns array according to these indices, return matching subset
        filtered_patterns = patterns[filter_indices]
        self._patterns = filtered_patterns
        return self._patterns

    def _run_motif_edit_distance_calculations(self):

        """
        Calculate Levenshtein distance between search term motif(s) and all corpus patterns stored in self._patterns.
        Detect patterns within a distance threshold of 1 and write them to file.
        Return pattern indices (which correspond to row identifiers in PatternSimilarity._pattern_freq_matrix).

        Args:
            mode -- choice of edit distance to be applied, can be either 'levenshtein' (Levenshtein distance);
            'hamming' (Hamming distance) or 'custom_weighted_hamming' (custom-weighted Hamming distance)"""

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

        out_path = f"{self._out_dir}/{self.query_tune}/{self.n}gram_similar_patterns"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        filtered.to_csv(f"{out_path}/{self.n}gram_similar_patterns.csv")
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
        Call _run_edit_distance_calculations() method to find all similar patterns to the 'motif' search term(s)
        .Occurrences of these patterns are the looked up in self_pattern_occurrences_matrix vi their indices and
        matrix rows containing their occurrences per tune are returned.
        """

        # find all patterns in self._patterns which are an edit distance of 1 or less from the search term pattern(s)
        # extracted from the query tune.
        similar_pattern_indices = self._run_motif_edit_distance_calculations()
        freq_matrix = self._pattern_occurrences_matrix
        # look up the rows corresponding to these patterns in self_pattern_occurrences_matrix.
        filtered = freq_matrix.loc[similar_pattern_indices].drop(columns=['pattern_len']).dropna(axis=1, how='all')
        filtered.fillna(0, inplace=True)
        return filtered

    def _count_similar_pattern_occurrences_per_tune(self, normalize=True):

        """
        Sums corpus-level occurrences for all similar patterns to the search term pattern(s) as outputted by
        self._find_similar_patterns_and_their_occurrences().

        Args:
            normalize -- Boolean flag: if True, output pattern occurrence counts for each tune are normalized by the
            length of the tune as stored in self._tune_lengths.
        """

        # generate input data
        data = self._find_similar_patterns_and_their_occurrences()

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

    def _calculate_motif_similarity(self, normalize=True):

        """
        Runs 'motif' similarity method via call to PatternSimilarity._count_similar_pattern_occurrences_per_tune();
        formats and writes results to file.

        Args:
            normalize -- Boolean flag: if True, putput pattern occurrence counts are normalized by the length of the
            tune. Passes to self._count_similar_pattern_occurrences_per_tune().
        """

        results = self._count_similar_pattern_occurrences_per_tune(normalize=normalize)
        # Reformat / rename cols and sort:
        results.sort_values(by=results.columns[1], ascending=False, inplace=True)
        results.reset_index(inplace=True, drop=True)
        print(results.head())
        # format filenames and write outputs to file
        norm = 'normalized_' if normalize else ''
        out_path = f"{self._out_dir}/{self.query_tune}/motif_results"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        csv_out_path = f"{out_path}/{norm}{self.n}gram_results.csv"
        results.to_csv(csv_out_path)

        return None

    def run_similarity_search(self, mode=None, motif_norm=True, edit_dist_metric='levenshtein'):

        """
        Top-level flow control to select and run FoNN's three similarity search methods: 'motif',
        'incipit_and_cadence', and 'tfidf'.
        Output is automatically written to disc in csv format at '../[corpus]/similarity_results' dir.

        Args:
             mode -- selects between three modes above.
             motif_norm -- Boolean flag. Selects whether to normalize output of 'motif' method.
             edit_dist_metric -- selects between three edit distance metrics available in 'incipit_and_cadence' mode:
                                 1. Levenshtein distance ('levenshtein'); 2. Hamming distance ('hamming'); or
                                 3. Custom-weighted Hamming distance ('custom_weighted_hamming').
        """

        print(f"Query tune: {self.query_tune}.")

        if mode == 'motif':
            # run 'motif' similarity methods
            print(f"Search mode: {mode}{' (normalized)' if motif_norm else ''}")
            self._filter_patterns()
            self._calculate_motif_similarity(normalize=motif_norm)

        elif mode == 'incipit_and_cadence':
            # run incipit and cadence similarity methods
            assert edit_dist_metric in self.EDIT_DIST_METRICS.keys()
            formatted_mode = ' '.join(mode.split('_'))
            formatted_edit_dist_metric = ' '.join(edit_dist_metric.split('_')).title()
            print(f"Similarity search mode: {formatted_mode} ({formatted_edit_dist_metric} distance)")
            self._incipit_and_cadence_flow_control(edit_dist_metric=edit_dist_metric)

        elif mode == 'tfidf':
            # run 'tfidf' similarity methods; format and print output
            print(f'Similarity search mode: {mode.upper()}')
            self._read_precomputed_tfidf_vector_similarity_results()

        return None


class PatternSimilarityDev(PatternSimilarity):

    MOTIF_MODES = ('exact', 'composite', 'edit_distance')

    def __init__(self, *args, **kwargs):

        super(PatternSimilarityDev, self).__init__(*args, **kwargs)
        self.motif_edit_distance_filter_range = None
        self.include_query_tune_in_results = False
        self.motif_count_weighting_factor = None

    @property
    def motif_edit_distance_filter_range(self):
        return self._motif_edit_distance_filter_range

    @motif_edit_distance_filter_range.setter
    def motif_edit_distance_filter_range(self, rng=None):

        if rng:
            assert isinstance(rng, tuple) and len(rng) == 2 and rng[0] < rng[1]
        self._motif_edit_distance_filter_range = rng

    @property
    def include_query_tune_in_results(self):
        return self._include_query_tune_in_results

    @include_query_tune_in_results.setter
    def include_query_tune_in_results(self, flag):

        assert isinstance(flag, bool)
        self._include_query_tune_in_results = flag

    @property
    def motif_count_weighting_factor(self):
        return self._motif_count_weighting_factor

    @motif_count_weighting_factor.setter
    def motif_count_weighting_factor(self, factor):
        if factor:
            assert isinstance(factor, (int, float))
        self._motif_count_weighting_factor = factor

    def _read_precomputed_tfidf_vector_similarity_results(self):

        """Apply _lookup_precomputed_results() to TF-IDF vector Cosine similarity matrix and write output to disc."""

        tfidf_similarity_matrix = self._tfidf_vector_cos_similarity_matrix
        # read tfidf similarity matrix via _lookup_precomputed_results()
        tfidf_results = self._lookup_precomputed_results(tfidf_similarity_matrix, ascending=False)
        # format and print output
        tfidf_results = tfidf_results.rename(columns={f"{self.query_tune}": "Cosine similarity"})
        print(tfidf_results.head())

        # drop query tune from results
        tfidf_results = tfidf_results[tfidf_results.index != self.query_tune]

        # setup out paths, create subdirs if they do not already exist
        tfidf_results_path = f"{self._out_dir}/{self.query_tune}/tfidf_results"
        if not os.path.isdir(tfidf_results_path):
            os.makedirs(tfidf_results_path)
        # write output
        tfidf_results.to_csv(f"{tfidf_results_path}/tfidf_vector_cos_similarity.csv")
        return tfidf_results

    def _incipit_and_cadence_flow_control(self, edit_dist_metric='levenshtein'):

        """
        Flow control: run 'incipit and cadence' similarity method and write results to disc.

        Args:
            edit_dist_metric -- select edit distance metric. Value can be 'levenshtein' (Levenshtein distance);
            'hamming' (Hamming distance); or 'custom_weighted_hamming' (custom-weighted Hamming distance), as defined in
            PatternSimilarity.EDIT_DIST_METRICS.
        """

        # setup paths
        out_path = f"{self._out_dir}/{self.query_tune}/incipit_and_cadence_results"
        # extract incipit and cadence subsequences from feature sequence data
        self._incipits_and_cadences = self._extract_incipits_and_cadences()
        # apply selected edit distance metric to calculate pairwise distance between incipit and cadence of query tune
        # vs all other tunes.
        results = self._calculate_incipit_and_cadence_edit_distance(edit_dist_metric)
        print(results.head())
        # drop query tune from results
        if not self.include_query_tune_in_results:
            results = results[results['title'] != self.query_tune]

        # format output filenames and write results to file
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        results.to_csv(out_path + f"/incipit_and_cadence_{edit_dist_metric}.csv")
        return None

    def _extract_search_term_motifs_from_query_tune(self):

        """
        Extract representative motif(s) from query tune by maximal TF-IDF for use as search term(s) in 'motif' method.
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

        # # lookup index or indices corresponding to max tfidf value(s) in DataFrame
        # max_tfidf_indices = query_tune_data_densified[query_tune_data_densified[title] ==
        #                                               query_tune_data_densified[title].max()].index
        # search_term_indices = [int(i) for i in max_tfidf_indices]

        # lookup indices of top three patterns in query tune as ranked by TF-IDF
        max_tfidf_indices = query_tune_data_densified[title].sort_values(ascending=False).index
        search_term_indices = max_tfidf_indices[:2]

        # look up the patterns corresponding to these indices in the patterns array, return as search terms
        search_terms = patterns[search_term_indices]
        return search_terms

    def _extract_top_3_motifs_from_query_tune(self):

        """
        Extract representative motif(s) from query tune by maximal TF-IDF for use as search term(s) in 'motif' method.
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

        # # lookup index or indices corresponding to max tfidf value(s) in DataFrame
        # max_tfidf_indices = query_tune_data_densified[query_tune_data_densified[title] ==
        #                                               query_tune_data_densified[title].max()].index
        # search_term_indices = [int(i) for i in max_tfidf_indices]

        # lookup indices of top three patterns in query tune as ranked by TF-IDF
        tfidf_indices = query_tune_data_densified[title].sort_values(ascending=False).index
        search_term_indices = tfidf_indices[:3]

        # Look up search terms patterns and write to file
        search_terms = patterns[search_term_indices]
        search_terms.name = 'patterns'
        out_path = f"{self._out_dir}/{self.query_tune}/motif_results/{self.n}gram_results/exact"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        search_terms.to_csv(f"{out_path}/{self.n}gram_exact_patterns.csv")

        return search_term_indices

    def _find_exact_pattern_matches_and_their_occurrences(self):

        """
        Call _run_edit_distance_calculations() method to find all similar patterns to the 'motif' search term(s)
        .Occurrences of these patterns are the looked up in self_pattern_occurrences_matrix vi their indices and
        matrix rows containing their occurrences per tune are returned.
        """

        # find indices of all patterns in self._patterns which are exact matches to the search term pattern(s)
        # extracted from the query tune.
        search_term_indices = self._extract_top_3_motifs_from_query_tune()

        freq_matrix = self._pattern_occurrences_matrix
        # look up the rows corresponding to these patterns in self_pattern_occurrences_matrix.
        filtered = freq_matrix.loc[search_term_indices].drop(columns=['pattern_len']).dropna(axis=1, how='all')
        filtered.fillna(0, inplace=True)
        return filtered

    def _count_search_term_pattern_occurrences_per_tune(self, normalize=True):

        """
        Sums corpus-level occurrences of the search term pattern(s) per tune as outputted by
        self._find_exact_pattern_matches_and_their_occurrences().

        Args:
            normalize -- Boolean flag: if True, output pattern occurrence counts for each tune are normalized by the
            length of the tune as stored in self._tune_lengths.
        """

        weighting = self.motif_count_weighting_factor

        # generate input data
        data = self._find_exact_pattern_matches_and_their_occurrences()
        # If weighting, count pattern occurrences per tune and multiply totals by 2
        # This modification to the pattern counts will boost counts of exact-matching patterns vs similar patterns in
        # final results
        count = data.sum(axis=0)
        ranking = count * weighting if weighting else count

        if normalize:
            # normalization process
            # read self._tune_lengths
            normalization_table = self._tune_lengths
            # map tune lengths to corresponding columns in matrix
            normalized = normalization_table[normalization_table.columns.intersection(data.columns)].squeeze()
            # normalize
            results = (ranking / normalized).round(decimals=3).to_frame().reset_index()
            results.columns = ['title', f'normalized_count']
        else:
            # if normalization is not desired, return ranking
            results = ranking.to_frame().reset_index()
            results.columns = ['title', 'count']

        # drop query tune from results
        if not self.include_query_tune_in_results:
            results = results[results['title'] != self.query_tune]

        return results

    def _run_motif_edit_distance_calculations(self, dist_metric='levenshtein', motif_mode=None):

        """
        Calculate Levenshtein distance between search term motif(s) and all corpus patterns stored in self._patterns.
        Detect patterns within a distance threshold of 1 and write them to file.
        Return pattern indices (which correspond to row identifiers in PatternSimilarity._pattern_freq_matrix).

        Args:
            dist_metric -- choice of edit distance to be applied, can be either 'levenshtein' (Levenshtein distance);
            'hamming' (Hamming distance), 'custom_weighted_hamming' (custom-weighted Hamming distance),
            custom-weighted Levenshtein distance ('custom_weighted_levenshtein')"""

        search_term_pattern_indices = self._extract_top_3_motifs_from_query_tune()
        patterns = self._patterns
        results = None
        search_terms = patterns[search_term_pattern_indices]

        if dist_metric == 'custom_weighted_hamming' or 'custom_weighted_levenshtein':
            # convert self._patterns data to str
            reformatted_patterns = pd.Series([''.join([str(int(i)) for i in p]) for p in patterns])
            # reindex to match original patterns Series
            reformatted_patterns.index = patterns.index
            # extract search terms
            reformatted_search_terms = reformatted_patterns[search_term_pattern_indices]

        if dist_metric == 'levenshtein':
            # apply pairwise Levenshtein.distance() between search term pattern(s) and all other patterns in the
            # (filtered) patterns array
            results = pd.DataFrame([patterns.swifter.apply(distance, args=[t]) for t in search_terms]).T
        if dist_metric == 'hamming':
            # calculate Hamming distance between search term pattern(s) and all other patterns in the
            # (filtered) patterns array
            results = pd.DataFrame([patterns.swifter.apply(hamming, args=[t]) for t in search_terms]).T
            # reformat scipy's Hamming dist values from decimal to simple count
            results = (results * self.n).astype(int)
        if dist_metric == 'custom_weighted_hamming':
            results = pd.DataFrame([reformatted_patterns.swifter.apply(
                weighted_levenshtein.levenshtein,
                args=[t],
                substitute_costs=substitution
            )
                for t in reformatted_search_terms]).T
        if dist_metric == 'custom_weighted_levenshtein':
            results = pd.DataFrame([reformatted_patterns.swifter.apply(
                weighted_levenshtein.levenshtein,
                args=[t],
                substitute_costs=substitution,
                insert_costs=insertion,
                delete_costs=deletion
            )
                for t in reformatted_search_terms]).T

        if dist_metric == 'custom_weighted_hamming' or 'custom_weighted_levenshtein':
            results.columns = [term for term in reformatted_search_terms]
        else:
            results.columns = [np.array2string(term) for term in search_terms]
        # filter results by min and max edit distance threshold values set via self.motif_edit_distance_filter_range
        # property. This retains only patterns within a user-defined distance range of the query tune.
        rng = self.motif_edit_distance_filter_range
        filtered = results[((rng[0] <= results) & (results <= rng[1])).any(1)]
        # retain the indices of the similar patterns discovered-- these will be used to lookup the 'freq' matrix and
        # count similar pattern occurrences per tune
        similar_pattern_indices = filtered.index.to_list()
        similar_patterns = patterns[similar_pattern_indices]
        filtered['patterns'] = similar_patterns
        # setup paths and write similar patterns to csv

        out_path = f"{self._out_dir}/{self.query_tune}/motif_results/{self.n}gram_results/{motif_mode}"
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        filtered.to_csv(f"{out_path}/{self.n}gram_{dist_metric}_patterns.csv")

        return similar_pattern_indices

    def _find_similar_patterns_and_their_occurrences(self, metric=None, motif_mode=None):

        """
        Call _run_edit_distance_calculations() method to find all similar patterns to the 'motif' search term(s)
        .Occurrences of these patterns are the looked up in self_pattern_occurrences_matrix vi their indices and
        matrix rows containing their occurrences per tune are returned.
        """

        # find all patterns in self._patterns which are an edit distance of 1 or less from the search term pattern(s)
        # extracted from the query tune, excluding the query patterns.
        similar_pattern_indices = self._run_motif_edit_distance_calculations(dist_metric=metric, motif_mode=motif_mode)
        freq_matrix = self._pattern_occurrences_matrix
        # look up the rows corresponding to these patterns in self_pattern_occurrences_matrix.
        filtered = freq_matrix.loc[similar_pattern_indices].drop(columns=['pattern_len']).dropna(axis=1, how='all')
        filtered.fillna(0, inplace=True)
        return filtered

    def _count_similar_pattern_occurrences_per_tune(self, normalize=True, metric=None, motif_mode=None):

        """
        Sums corpus-level occurrences for all similar patterns to the search term pattern(s) as outputted by
        self._find_similar_patterns_and_their_occurrences().

        Args:
            normalize -- Boolean flag: if True, output pattern occurrence counts for each tune are normalized by the
            length of the tune as stored in self._tune_lengths.
        """

        # generate input data
        data = self._find_similar_patterns_and_their_occurrences(metric=metric, motif_mode=motif_mode)

        if normalize:
            # normalization process
            # read self._tune_lengths
            normalization_table = self._tune_lengths
            # map tune lengths to corresponding columns in matrix
            normalized = normalization_table[normalization_table.columns.intersection(data.columns)].squeeze()
            # normalize
            results = (data.sum(axis=0) / normalized).round(decimals=3).to_frame().reset_index()
            results.columns = ['title', f'normalized_count']
        else:
            # if normalization is not desired, raw pattern occurrence counts are returned
            results = data.sum(axis=0).to_frame().reset_index()
            results.columns = ['title', 'count']

        # drop query tune from results
        if not self.include_query_tune_in_results:
            results = results[results['title'] != self.query_tune]

        return results

    def _combine_exact_and_similar_pattern_results(self, normalize, metric, motif_mode):

        exact = self._count_search_term_pattern_occurrences_per_tune(normalize=normalize)
        similar = self._count_similar_pattern_occurrences_per_tune(
            normalize=normalize,
            metric=metric,
            motif_mode=motif_mode
        )
        # combine dfs and sum counts of common tunes between the two results tables
        results = pd.concat([exact, similar]).groupby('title')['count'].sum().reset_index()

        return results

    def _calculate_motif_similarity(self, normalize=True, motif_mode=None, metric=None):

        """
        Runs 'motif' similarity method via call to PatternSimilarity._count_similar_pattern_occurrences_per_tune();
        formats and writes results to file.

        Args:
            normalize -- Boolean flag: if True, putput pattern occurrence counts are normalized by the length of the
            tune. Passes to self._count_similar_pattern_occurrences_per_tune().
        """

        assert motif_mode in self.MOTIF_MODES
        results = None

        if motif_mode == 'exact':
            results = self._count_search_term_pattern_occurrences_per_tune(normalize=normalize)
        if motif_mode == 'composite':
            assert metric in self.EDIT_DIST_METRICS
            results = self._combine_exact_and_similar_pattern_results(normalize=normalize, metric=metric, motif_mode=motif_mode)
        if motif_mode == 'edit_distance':
            results = self._count_similar_pattern_occurrences_per_tune(normalize=normalize, metric=metric, motif_mode=motif_mode)

        # Reformat / rename cols and sort:
        results.sort_values(by=results.columns[1], ascending=False, inplace=True)
        results.reset_index(inplace=True, drop=True)
        print(results.head())
        # format filenames and write outputs to file
        norm = 'normalized_' if normalize else ''
        base_path = f"{self._out_dir}/{self.query_tune}/motif_results/{self.n}gram_results/{motif_mode}"
        if not os.path.isdir(base_path):
            os.makedirs(base_path)
        if motif_mode == 'exact':
            csv_out_path = f"{base_path}/{norm}{self.n}gram_results_exact.csv"
        else:
            csv_out_path = f"{base_path}/{norm}{self.n}gram_results_{motif_mode}_{metric}.csv"

        results.to_csv(csv_out_path)

        return None

    def run_similarity_search(self, method=None, **kwargs):

        """
        Top-level flow control to select and run FoNN's three similarity search methods: 'motif',
        'incipit_and_cadence', and 'tfidf'.
        Output is automatically written to disc in csv format at '../[corpus]/similarity_results' dir.

        Args:
             method -- selects between three methods above.
             motif_norm -- Boolean flag. Selects whether to normalize output of 'motif' method.
             motif_mode -- selects between 'exact', 'composite' and 'edit_distance' modes.
             edit_dist_metric -- selects between three edit distance metrics available in 'incipit_and_cadence' method:
                                 1. Levenshtein distance ('levenshtein'); 2. Hamming distance ('hamming'); or
                                 3. Custom-weighted Hamming distance ('custom_weighted_hamming').
                                 4. Custom-weighted Levenshtein distance ('custom_weighted_levenshtein')
        """

        motif_norm = kwargs.pop('motif_norm', True)
        motif_mode = kwargs.pop('motif_mode', 'edit_distance')
        edit_dist_metric = kwargs.pop('edit_dist_metric', 'levenshtein')

        assert isinstance(motif_norm, bool)
        assert motif_mode in self.MOTIF_MODES
        assert edit_dist_metric in self.EDIT_DIST_METRICS

        print(f"Query tune: {self.query_tune}.")

        if method == 'motif':
            assert edit_dist_metric in self.EDIT_DIST_METRICS.keys()
            # run 'motif' similarity methods
            print(f"Search method: {method}{' (normalized)' if motif_norm else ''}\n"
                  f"{'Edit distance metric: ' if edit_dist_metric else ''}{edit_dist_metric}")
            if 'hamming' in edit_dist_metric:
                self._pattern_occurrences_matrix = self._filter_matrix(
                    data=self._pattern_occurrences_matrix,
                    mode='narrow'
                )
            self._filter_patterns()

            self._calculate_motif_similarity(normalize=motif_norm, motif_mode=motif_mode, metric=edit_dist_metric)

        elif method == 'incipit_and_cadence':
            # run incipit and cadence similarity methods
            assert edit_dist_metric in self.EDIT_DIST_METRICS.keys()
            formatted_method = ' '.join(method.split('_'))
            formatted_edit_dist_metric = ' '.join(edit_dist_metric.split('_')).title()
            print(f"Similarity search method: {formatted_method} ({formatted_edit_dist_metric} distance)")
            self._incipit_and_cadence_flow_control(edit_dist_metric=edit_dist_metric)

        elif method == 'tfidf':
            # run 'tfidf' similarity methods; format and print output
            print(f'Similarity search method: {method.upper()}')
            self._read_precomputed_tfidf_vector_similarity_results()

        return None

    # TODO: make self.n an attr and data attrs properties

    # TODO: Test -- make new test dataset

    # TODO: Check that outputs save in appropriate folders

    # TODO: run comparison of exact / exact plus weighted Hamming / exact plus Levenshein / Levenshtein / weighted
    #  Hamming.

    # TODO: Apply some of above motif options to Incipit and Cadence?








