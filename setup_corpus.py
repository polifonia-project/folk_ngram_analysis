"""
Flow control / corpus setup script, running tools from corpus_processing_tools.py module.

This module's main() function extracts primary and secondary feature sequence data from a corpus
of monophonic MIDI files, and saves the results for each to csv. It also extract root note detection metrics for input
into 'Root Note Detection' component.
For further information, please see docstrings below.
"""

from corpus_processing_tools import Corpus


def calc_music21_root_metrics(inpath, outpath):

    """
    Reads MIDI corpus to music21 stream format and calculates an initial set of root note detection metrics for each
    tune, which are saved to csv.

    Further root detection metrics, which are dependant on secondary feature sequence data, can be calculated and
    appended via calc_feat_seqs() below.

    Root note detection metrics outputted by calc_music21_root_metrics():

    - 'as transcribed': pitch class of root note read from music21 key signature.
    - 'final_note': pitch class of final note of the tune.
    - 'krumhansl_schmuckler': pitch class of root as outputted by Krumhansl-Schmukler algorithm
    - simple_weights': pitch class of root as outputted using Craig Sapp's 'simple weights'
    - arden_essen': pitch class of root as outputted by Aarden Essen algorithm
    - 'bellman_budge': pitch class of root as outputted by Bellman-Budge algorithm
    - 'temperley_kostka_payne': pitch class of root as outputted by Temperley-Kostka-Payne algorithm

    Args:
        inpath -- path to directory containing MIDI corpus.
        outpath -- path to write root note detection metrics to csv.
    """

    cre_roots = Corpus(inpath)
    cre_roots.filter_empty_scores()
    cre_roots.roots_path = outpath
    cre_roots.calculate_root_metrics()
    cre_roots.convert_note_names_to_pitch_classes()
    cre_roots.save_roots_table()


def calc_feat_seqs(inpath, roots_path, outpath):

    """
    Calls all methods required to calculate primary, secondary, and duration-weighted feature sequences from MIDI
    corpus. Writes three csv files for each tune in corpus, respectively containing:
    1. Note-level feature sequence data.
    2. Accent-level feature sequence data
    3. Duration-weighted note-level feature sequence data for selected features
    (per below, the defaults are 'midi_note', 'relative_pitch_class' and 'velocity')

    Each outputted file contains sequences of the following musical features:
    - 'midi_note': MIDI note number
    - 'onset': note onset (1/4 notes)
    - 'duration': note (1/4 notes)
    - 'velocity': MIDI velocity
    - 'onset': rescaled note onset (1/8 notes)
    - 'duration': rescaled duration (1/8 notes)
    - 'interval': chromatic interval between successive notes.
    - 'root': scalar key-invariant pitch class value representing the root or tonal centre of a tune
    - 'relative_pitch': key-invariant chromatic pitch.
    - 'relative_pitch_class': key-invariant chromatic pitch class.
    - 'parsons_code': simple melodic contour. Please see Tune.calc_parsons_codes() docstring for detailed explanation.
    - 'parsons_cumsum': cumulative Parsons code values (convenient for graphic representation of contour).


    Also calculates root note detection metrics from feature sequence data:

    - 'freq note': pitch class of the most frequently-occurring tone in each tune as a pitch class.
    - 'freq acc': pitch class of the most frequently-occurring accented tone in each tune.
    - 'freq weighted note': pitch class of the most frequently-occurring duration-weighted tone in each tune.
    - 'freq weighted acc': pitch class of the most frequently-occurring accented duration-weighted tone in each tune.

    These values are appended to Corpus.roots Dataframe in columns named per the above, and saved to csv.

    Args:
        inpath -- path to directory containing MIDI corpus.
        roots_path -- path to root note data table in either csv or pkl format.
        csv_outpath -- path under which corpus feature sequence data is written to csv.
    """

    cre = Corpus(inpath)
    cre.filter_empty_scores()
    cre.roots_path = roots_path
    cre.calculate_feat_seqs()
    cre.calc_pitch_class_seqs()
    cre.calc_intervals()
    cre.read_root_data()
    cre.convert_roots_to_midi_note_nums()
    cre.assign_roots()
    cre.calc_relative_pitch_seqs()
    cre.calc_relative_pitch_class_seqs()
    cre.filter_accents()
    cre.calc_parsons_codes()
    cre.calc_parsons_cumsum()
    cre.calc_duration_weighted_feat_seqs(['midi_note', 'relative_pitch_class', 'velocity'])
    cre.calc_duration_weighted_accent_seqs()
    cre.csv_outpath = outpath
    cre.save_feat_seq_data_to_csv()
    cre.find_most_freq_notes()
    cre.reformat_roots_table()
    cre.save_roots_table()


def main():

    """
    Main function. Converts a corpus of monophonic MIDI files to feature sequence representation via calc_feat_seqs()
    above.

    Also calculates root note metric inputs for 'Root Note Detection' component at ./root_note_detection/ via
    calls in both calc_music21_root_metrics() and calc_feat_seqs() functions.

    'inpath' variable below points to local location of MIDI corpus directory;
    'roots_path' variable points to local location of 'roots.csv' file as described in corpus_processing_tools.py
    docstrings;
    'csv_outpath' variable points to directory at which output files will be written.
    """

    basepath = "./corpus"
    inpath = basepath + "/MIDI"
    roots_path = basepath + "/root_note_detection/roots.csv"
    csv_outpath = basepath + '/feat_seq_corpus'
    calc_music21_root_metrics(inpath, roots_path)
    calc_feat_seqs(inpath, roots_path, csv_outpath)


if __name__ == "__main__":
    main()

