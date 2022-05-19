"""
Flow control / corpus setup script, running tools from corpus_processing_tools.py module.

This module's main() function extracts primary and secondary feature sequence data from a corpus
of monophonic MIDI files, and saves the results for each to csv. For further information, please see the main()
function docstring.
"""

from corpus_processing_tools import Corpus


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

    Args:
        in_dir -- path to directory containing MIDI corpus.
        roots_path -- path to root note data table in either csv or pkl format.
        csv_outpath -- path under which corpus feature sequence data is written to csv.
    """

    thesession = Corpus(inpath)
    thesession.filter_empty_scores()
    thesession.roots_path = roots_path
    thesession.calculate_feat_seqs()
    thesession.calc_pitch_class_seqs()
    thesession.calc_intervals()
    thesession.read_root_data()
    thesession.convert_roots_to_midi_note_nums()
    thesession.assign_roots()
    thesession.calc_relative_pitch_seqs()
    thesession.calc_relative_pitch_class_seqs()
    thesession.filter_accents()
    thesession.calc_parsons_codes()
    thesession.calc_parsons_cumsum()
    thesession.calc_duration_weighted_feat_seqs(['midi_note', 'relative_pitch_class', 'velocity'])
    thesession.calc_duration_weighted_accent_seqs()
    thesession.csv_outpath = outpath
    thesession.save_feat_seq_data_to_csv()


def main():

    """
    Main function. Converts a corpus of monophonic MIDI files to feature sequence representation via calc_feat_seqs()
    function above.

    'inpath' variable below points to local location of MIDI corpus directory;
    'roots_path' variable points to local location of 'roots.csv' file as described in corpus_processing_tools.py
    docstrings;
    'csv_outpath' variable points to directory at which output files will be written.
    """

    basepath = "./corpus"
    inpath = basepath + "/MIDI"
    roots_path = basepath + "/roots.csv"
    csv_outpath = basepath + '/feat_seq_corpus'
    calc_feat_seqs(inpath, roots_path, csv_outpath)


if __name__ == "__main__":
    main()

