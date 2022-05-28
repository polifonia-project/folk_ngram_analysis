# work-in-progress on a set of functions to find and visualize n-gram pattern occurrences
# in feature sequence data.


import matplotlib.pyplot as plt
import pandas as pd


def read_csv(inpath):
    # read csv file with 'n-gram' column as tuples (rather than strings)
    # NOTE: This reads a csv feature sequence file representing an individual tune rather than a table of patterns
    df = pd.read_csv(inpath, converters={'n-gram': eval})
    return df


def extract_frequent_ngrams(df):
    # function opens accent-level n-grams dataframe for a single tune in the corpus
    # it sorts the table rows by tf-idf. n-gram column values are extracted for all rows containing the maximum tf-idf
    # value in the table.
    df.sort_values(by=['tf_idf'], axis=0, ascending=False, inplace=True)
    filtered_df = df[df['tf_idf'] == df['tf_idf'].max()]
    extracted_ngrams = filtered_df['n-gram'].tolist()
    print(extracted_ngrams)
    return extracted_ngrams


def extract_feature_sequence(df, feature='pitch_class', threshold=81):
    # function to extract a single accent-level feature sequence from a csv file containing multiple feature sequences
    # for a given tune.
    feature_sequence = df[feature][df['velocity'] > threshold].tolist()
    feat_seq_df = pd.DataFrame()
    feat_seq_df[feature] = feature_sequence

    return feat_seq_df


def find_ngram_occurrences(ngrams, df, feature='pitch_class'):
    # function to compare ngram(s) extracted by extract_frequent_ngrams()
    # against the given feature sequence and check for matching patterns
    # outputs a dataframe with feature sequence extracted by extract_feature_sequence() in one column;
    # ngram occurrences detected within that sequence in another column (sparse / padded with zeroes)
    feature_sequence = df[feature].tolist()
    for n in ngrams:
        df[n] = [None] * len(feature_sequence)
        for idx, ngram in enumerate(feature_sequence):
            test_slice = feature_sequence[idx:idx + len(n)]
            if list(n) == test_slice:
                df[n][idx:idx + len(n)] = test_slice

    print(df)
    return df


def visualize_results(df, outpath, title='tune title here', n_val=6, level='Accent', ngrams=True):
    # function to generate a line chart visualizing the feature sequence for a given tune, along with any
    # n-gram patterns detected within the sequence (which are displayed as red-highlighted sections of the line).
    if ngrams:
        df.plot.line(
            title=f"{title}:\n{level}-level {df.columns[0]} sequence with frequent {n_val}-grams",
            y=df.columns,
            xlabel=f'{level} sequence',
            ylabel=f'{df.columns[0]}',
            figsize=(10, 6)
        )
    else:
        df.plot.line(
            title=f"{title}:\n{level}-level {df.columns[0]} sequence",
            y=df.columns,
            xlabel=f'{level} sequence',
            ylabel=f'{df.columns[0]}',
            figsize=(10, 6)
        )

    plt.legend(loc="lower right").get_texts()[0].set_text(f'{df.columns[0]} seq.')
    plt.savefig(fname=outpath, bbox_inches="tight", dpi=600)
    plt.show()
    plt.close()

    return None


# -----------------------------------------------------------------------------------------------------------------------

# Function calls:

# Read n-gram table for candidate melody and extract most frequent ngram(s):
# (in this case, looking at accent-level pitch class 6-grams for Lord McDonald's Reel)
ngrams_inpath = '/Users/dannydiamond/NUIG/Polifonia/CRE testbed/tf_idf/pitch_class_ngrams_TFIDF/' \
         'pitch_class_n_6_TFIDF/pitch_class_n_6_TFIDF/'
ngrams_filename = "Lord McDonald's (reel)_pitch_class_n_6_ACC_SEQ_TFIDF.csv"
ngrams_df = read_csv(ngrams_inpath + ngrams_filename)
candidate_ngrams = extract_frequent_ngrams(ngrams_df)
# Read feature sequence data for same melody and extract feature of interest:
sequence_inpath = "/Users/dannydiamond/NUIG/Polifonia/CRE testbed/numeric_representation_csv/Lord McDonald's (reel).csv"
feat_seq_df = read_csv(sequence_inpath)
pitch_class_accents_sequence = extract_feature_sequence(feat_seq_df)
# Visualize accent-level pitch class sequence for Lord McDonald's:
pitch_class_accents_seq_outpath = '/Users/dannydiamond/NUIG/Polifonia/internal_presentation_sept2021/' \
                                  'figures/pitch_class_accent_seq_plot'
visualize_results(pitch_class_accents_sequence,
                  pitch_class_accents_seq_outpath,
                  title="Lord McDonald's (reel)",
                  ngrams=False
                  )
results = find_ngram_occurrences(candidate_ngrams, pitch_class_accents_sequence)
# Visualize accent-level pitch class sequence with frequent 6-grams for Lord McDonald's:
ngrams_outpath = '/Users/dannydiamond/NUIG/Polifonia/internal_presentation_sept2021/figures/freq_ngram_plot'
visualize_results(results, ngrams_outpath, title="Lord McDonald's (reel)")

# Visualize accent-level pitch sequence for Lord McDonald's:
pitch_accents_sequence = extract_feature_sequence(feat_seq_df, feature='pitch', threshold=81)
pitch_accents_sequence_outpath = '/Users/dannydiamond/NUIG/Polifonia/internal_presentation_sept2021/' \
                                  'figures/pitch_accent_seq_plot'
visualize_results(pitch_accents_sequence,
                  pitch_accents_sequence_outpath,
                  title="Lord McDonald's (reel)",
                  level='Accent',
                  ngrams=False
                  )

# # Visualizing note-level pitch sequence for Lord McDonald's:
# pitch_note_sequence = extract_feature_sequence(feat_seq_df, feature='pitch', threshold=0)
# pitch_note_sequence_outpath = '/Users/dannydiamond/NUIG/Polifonia/internal_presentation_sept2021/' \
#                                   'figures/pitch_note_seq_plot'
# visualize_results(pitch_note_sequence,
#                   pitch_note_sequence_outpath,
#                   title="Lord McDonald's (reel)",
#                   level='Note',
#                   ngrams=False
#                   )