"""Flow control / corpus setup script, running tools from corpus_processing_tools.py module."""

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
    # TODO: Add ClI?
    # TODO: Target online rather than local corpus data
    """Main function for setting up Ceol Rince na hEireann (CRE) test corpus"""
    cre_inpath = "/Users/dannydiamond/NUIG/Polifonia/CRE_clean/testing/MIDI"
    cre_m21_corpus = Music21Corpus(cre_inpath)
    cre_corpus = SetupCorpus(cre_m21_corpus)
    cre_corpus.generate_primary_feat_seqs()
    cre_corpus.setup_music_data_corpus()
    cre_corpus.run_simple_secondary_feature_sequence_calculations()
    cre_roots_path = "/Users/dannydiamond/NUIG/Polifonia/CRE_clean/testing/roots_a.csv"
    cre_corpus.run_key_invariant_sequence_calulations(cre_roots_path)
    cre_corpus.run_duration_weighted_sequence_calculations(['pitch', 'pitch_class'])
    cre_corpus.save_corpus(
        feat_seq_path="/Users/dannydiamond/NUIG/Polifonia/CRE_clean/testing/feat_seq_data/note",
        accents_path="/Users/dannydiamond/NUIG/Polifonia/CRE_clean/testing/feat_seq_data/accent",
        duration_weighted_path="/Users/dannydiamond/NUIG/Polifonia/CRE_clean/testing/feat_seq_data/duration_weighted"
    )
    return cre_corpus


main()
