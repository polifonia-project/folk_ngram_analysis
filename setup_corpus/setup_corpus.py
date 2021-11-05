"""

Flow control / corpus setup script, running tools from corpus_processing_tools.py module.

This module's main() function extracts primary and secondary feature sequence data from a corpus
of monophonic MIDI files, and saves the results for each MIDI to csv. For further information, please see the main()
function docstring.

"""

from corpus_processing_tools import Music21Corpus, MusicDataCorpus


class SetupCorpus:

    """
    SetupCorpus class sets up Music21Corpus and MusicDataCorpus objects, and runs their methods: creating tables of
    primary and secondary feature sequence data at note event- and accent-level for every melody in an input corpus of
    monophonic MIDI files. It allows generation of additional duration-weighted feature sequences, via
    SetupCorpus.run_duration_weighted_sequence_calculations() method.
    """

    def __init__(self, m21_corpus):
        self.corpus = m21_corpus

    def generate_primary_feat_seqs(self):
        """Derives primary feature sequence data for all melodies in corpus via Music21Corpus methods."""
        self.corpus.read_midi_files_to_streams()
        self.corpus.derive_feat_seqs()
        self.corpus.combine_feat_seqs_and_titles()
        return self.corpus

    def setup_music_data_corpus(self):
        """Converts self.corpus attribute from Music21Corpus to MusicDataCorpus object."""
        self.corpus = MusicDataCorpus(self.corpus)
        return self.corpus

    def run_simple_secondary_feature_sequence_calculations(self):

        """
        Calculates simple secondary feature sequences at note event- and accent-level for all melodies
        in Music21Corpus object.
        """

        self.corpus.rescale_corpus_durations()
        self.corpus.rescale_corpus_onsets()
        self.corpus.calc_corpus_intervals()
        self.corpus.calc_corpus_parsons()
        self.corpus.calc_corpus_parsons_cumsum()
        return self.corpus

    def run_key_invariant_sequence_calulations(self, roots_path):

        """
        Reads root data; converts root values to MIDI note numbers via lookup table; derives key-invariant pitch
        and key invariant pitch class sequences. Runs at at note event- and accent-level for all
        melodies in Music21Corpus object.
        """

        self.corpus.read_root_data(roots_path)
        self.corpus.convert_roots_to_midi_note_nums()
        self.corpus.assign_roots()
        self.corpus.calc_key_invariant_pitches()
        self.corpus.calc_pitch_classes()
        return self.corpus

    def run_duration_weighted_sequence_calculations(self, features):

        """
        Generates duration-weighted sequence for selected features / column names, which are passed in list
        to 'features' arg. Runs on note event-level feature sequence data for all melodies in Music21Corpus object.
        """

        self.corpus.calc_duration_weighted_feat_seqs(features)
        return self.corpus

    def save_corpus(self, feat_seq_path, accents_path, duration_weighted_path):

        """
        Saves output corpus data using Music21Corpus.save_corpus_data_to_csv().

        feat_seq_path -- path to directory for note-level feature sequence data files for all melodies in corpus
        accents_path -- path to directory for accent-level feature sequence data files for all melodies in corpus
        duration_weighted_path -- path to directory for duration-weighted data for all melodies in corpus
        """

        self.corpus.save_corpus_data_to_csv(feat_seq_path, accents_path, duration_weighted_path)
        pass


def main():

    """
    Main function for setting up Ceol Rince na hEireann (CRE) test corpus.

    To run, first download the test dataset from:
    https://drive.google.com/drive/folders/1DTROUZeKHSs_Bqe0lsn2UXdfrW2IgBYi?usp=sharing;

    Next, point 'inpath' variable (below) to local location of 'midi' directory,
    and 'roots_path' variable (below) to local location of 'roots.csv' file.

    By default, this function will generate sequences of the following primary musical features:
    - 'MIDI_note': MIDI note number
    - 'Onset': note-event onset (eighth notes)
    - 'Duration': note-event duration (eighth notes)
    - 'velocity': MIDI velocity per note-event
    - 'interval': chromatic interval (relative to previous tone)
    - 'parsons_code': simple contour (Parsons code)
    - 'Parsons_cumsum': cumulative Parsons code contour
    - 'chromatic_root': root (chromatic pitch class of )
    - 'pitch': key-invariant pitch (relative to 4th octave MIDI numbers)
    - 'pitch_class': key-invariant chromatic pitch class

    This data is outputted at two levels for every melody in the corpus: per note-event and per accented note-event.
    Duration-weighted sequences can also be derived for selected features,
    with feature names passed as arguments to SetupCorpus.run_duration_weighted_sequence_calculations():
    Per below, the defaults are 'pitch' and 'pitch_class'
    """

    inpath = "/Users/dannydiamond/NUIG/Polifonia/CRE_clean/MIDI"
    m21_corpus = Music21Corpus(inpath)
    corpus = SetupCorpus(m21_corpus)
    corpus.generate_primary_feat_seqs()
    corpus.setup_music_data_corpus()
    corpus.run_simple_secondary_feature_sequence_calculations()
    roots_path = "/Users/dannydiamond/NUIG/Polifonia/CRE_clean/testing/roots.csv"
    corpus.run_key_invariant_sequence_calulations(roots_path)
    corpus.run_duration_weighted_sequence_calculations(['pitch', 'pitch_class'])
    corpus.save_corpus(
        feat_seq_path="/Users/dannydiamond/NUIG/Polifonia/CRE_clean/feat_seq_data/note",
        accents_path="/Users/dannydiamond/NUIG/Polifonia/CRE_clean/feat_seq_data/accent",
        duration_weighted_path="/Users/dannydiamond/NUIG/Polifonia/CRE_clean/feat_seq_data/duration_weighted"
    )
    return corpus


main()
