"""
Music21Corpus class allows extraction of primary feature sequence data from corpus MIDI files via the Music21 library:
- MIDI note number
- Onset (1/4 notes)
- Duration (1/4 notes)
- MIDI velocity
MusicData class objects represent a single melody in feature sequence format;
 MusicDataCorpus class objects represent feature sequence data for all melodies in a corpus:
Once primary feature sequence data has been extracted, additional secondary feature sequences can be added and derived
via these classes:
- Root (a single MIDI note number per melody, read from external data table)
- Rescaled onset (eighth notes)
- Rescaled duration (eighth notes)
- Key-invariant pitch (relative melodic movement; integer scale; derived from MIDI note number and Root data)
- Key-invariant pitch class (chromatic scale: 0-11)
These classes also allow filtering of the feature sequences at accent-level
(retaining data only for note events which occur on accented beats);
and generation of duration-weighted pitch class sequences (combining pitch class and duration in a single
sequence, which represents pitch class value per eighth note rather than per note event).
'"""

import music21
import numpy as np
import os
import pandas as pd
import traceback

import setup_corpus.constants as constants
import utils

pd.options.mode.chained_assignment = None


class Music21Corpus:

    """A Music21Corpus object is instantiated with a single argument, 'inpath', which is the path to a directory
    containing a corpus of monophonic MIDI files.

    Attributes:
        inpath -- per above.
        filenames -- list of all filenames in 'inpath' directory.
        titles -- reformatted 'filenames' list, omitting non-MIDI files and removing filetype suffixes.
        roots_df -- will hold dataframe of root detection metrics.
        melodies -- will hold list of all melodies in corpus after conversion to Music21 stream objects.
        feat_seqs -- Primary feature sequence data outputted by Music21Corpus.derive_feat_seqs().
        corpus -- will hold dict of titles (keys) and feat_seqs (values) for all melodies in corpus.
    """

    def __init__(self, inpath):
        self.inpath = inpath
        self.filenames = os.listdir(self.inpath)
        self.titles = [path.split('/')[-1][:-4] for path in self.filenames if path.endswith('.mid')]
        self.melodies = None
        self.feat_seqs = None
        self.corpus = None
        print(f"Input corpus contains {len(self.titles)} melodies:\n")
        print(*self.titles, sep='\n')

    def read_midi_files_to_streams(self):
        """Reads all MIDI files in 'inpath' directory to Music21 streams"""
        self.melodies = []
        for file in self.filenames:
            if file.endswith('.mid') and not file.startswith('.'):
                mf = music21.midi.MidiFile()
                mf.open((f"{self.inpath}/{file}").encode())
                try:
                    mf.read()
                    mf.close()
                    melody = music21.midi.translate.midiFileToStream(mf, quantizePost=False).flat
                except Exception as exc:
                    print(traceback.format_exc())
                    print(exc)
                self.melodies.append(melody)

        print(f"\n{len(self.melodies)} melodies successfully read to Music21 stream representation.")
        return self.melodies

    def derive_feat_seqs(self):

        """Extracts primary feature sequence data from streams and stores in Pandas dataframes,
        which are sored in Music21Corpus.feat_seqs attribute"""

        self.feat_seqs = []
        for melody in self.melodies:
            melody_df = pd.DataFrame(columns=["MIDI_note", "onset", "duration", "velocity"])
            for note in melody.recurse().notes:
                # output and format musical data from music21 stream, and store in dataframe:
                note_df = pd.DataFrame([[
                    round(float(note.offset), 2),
                    round(float(note.duration.quarterLength), 2),
                    note.volume.velocity
                    ]],
                    columns=["onset", "duration", "velocity"])
                if note.isNote:
                    note_df["MIDI_note"] = int(note.pitch.ps)
                if note.isChord:
                    melody_df["MIDI_note"] = int(note.root().ps)
                melody_df = melody_df.append(note_df, ignore_index=True)
            # list all feature sequence dataframes in Music21Corpus.feat_seqs attribute:
            self.feat_seqs.append(melody_df)
        print(f"{len(self.feat_seqs)} melodies successfully converted to feature sequence representation.\n")
        return self.feat_seqs

    def run_music21_key_detection_algs(self):
        """Runs Music21's built-in key detection algorithms and stores output in Music21Corpus.roots_df dataframe."""
        self.roots_df = pd.DataFrame()
        self.roots_df['title'] = self.titles
        self.roots_df.reset_index(inplace=True, drop=True)

        krumhansl = [tune.analyze('key') for tune in self.melodies]
        self.roots_df['Krumhansl-Shmuckler'] = krumhansl

        simple = [music21.analysis.discrete.SimpleWeights(tune).getSolution(tune)
                  for tune in self.melodies]
        self.roots_df['simple weights'] = simple

        aarden = [music21.analysis.discrete.AardenEssen(tune).getSolution(tune)
                  for tune in self.melodies]
        self.roots_df['Aarden Essen'] = aarden

        bellman = [music21.analysis.discrete.BellmanBudge(tune).getSolution(tune)
                   for tune in self.melodies]
        self.roots_df['Bellman Budge'] = bellman

        temperley = [music21.analysis.discrete.TemperleyKostkaPayne(tune).getSolution(tune)
                     for tune in self.melodies]
        self.roots_df['Temperly Kostka Payne'] = temperley

        print("Music 21 key detection complete.")
        return self.roots_df

    def extract_key_signature_from_midi_files(self):
        """Extracts key signature data from MIDI input files; stores in Music21Corpus.roots_df dataframe."""
        self.roots_df['MIDI root'] = [tune.keySignature for tune in self.melodies if tune.keySignature]
        print("Key signatures extracted from MIDI files.")
        return self.roots_df

    def extract_final_note(self):
        """Extracts note name of final note from every melody in corpus; stores in Music21Corpus.roots_df dataframe."""
        final_notes = []
        for mel in self.melodies:
            last_note = mel.recurse().notes[-1]
            if last_note.isNote:
                final_notes.append(last_note.name.upper())
            elif last_note.isChord:
                final_notes.append(last_note.root().name.upper())
            else:
                final_notes.append('')

        self.roots_df['final_note'] = final_notes
        print("\nFinal note extracted.")
        return self.roots_df

    def convert_keys_to_roots(self):
        """Extracts root note names from key signatures outputted by MusicDataCorpus.run_music21_key_detection_algs()"""
        roots = self.roots_df.copy()
        roots.dropna(axis=1, inplace=True)
        roots.set_index('title', inplace=True)
        print(roots.dtypes)
        self.roots_df = roots.applymap(lambda x: x.tonic.name.upper())
        return self.roots_df

    def convert_note_names_to_pitch_classes(self):

        """Converts note names, formatted per Music21 standard (i.e.: G- = G natural; G# = G sharp) to integer chromatic
         pitch class values from 0-11, via constants.music21_lookup_table."""

        roots = self.roots_df.astype('string')
        # lookup numeric root vals & map to new column:
        lookup = dict(zip(constants.music21_lookup_table['note name'], constants.music21_lookup_table['pitch class']))
        roots = roots.replace(lookup)
        roots.apply(pd.to_numeric, errors='coerce')
        print("\nRoot note names data table:")
        print(self.roots_df.head())
        print(f'Checksum: {len(roots)}')
        self.roots_df = roots
        return self.roots_df

    def combine_feat_seqs_and_titles(self):
        """Stores title-stream pairs for each melody in corpus in a dictionary"""
        self.corpus = dict(zip(self.titles, self.feat_seqs))
        return self.corpus

    def save_to_csv(self, output_dir):
        """Save primary feature sequence data to csv file"""
        for title, feat_seq in self.corpus.items():
            feat_seq.to_csv(f"{output_dir}/{title}.csv")


class MusicData:
    """A MusicData object represents a single melody: it holds the melody name (MusicData.title)
    and numeric data sequences representing various features of the melody; it is instantiated with a single
    argument, 'feat_seq', which must be a two-tuple formatted per:
    (melody title [str], primary feature sequence data [Pandas dataframe])

    Attributes:
        title -- melody title (MIDI filename)

        music_data -- Pandas dataframe containing primary feature sequence data for melody

        music_data_accents --  filtered version of music_data dataframe, retaining only data for note events
        with MIDI velocity values above a threshold value of 80.

        weighted_music_data -- empty attribute to hold duration-weighted feature sequence dataframe as returned by
        MusicData.generate_duration_weighted_music_data() method.

        weighted_music_data_accents -- empty attribute to hold accent-level duration-weighted feature sequence data,
        as returned by MusicData.extract_duration_weighted_music_data_accents() method.

    """

    def __init__(self, feat_seq):
        self.title = feat_seq[0]
        self.music_data = feat_seq[1]
        self.music_data_accents = utils.filter_dataframe(self.music_data, seq='velocity')
        self.weighted_music_data = None
        self.weighted_music_data_accents = None

    def __getitem__(self, item):
        return getattr(self, item)

    def rescale_durations(self):

        """
        Rescales duration column in MusicData.music_data & MusicData.music_data_accents dataframes
        from quarter notes to eighth notes.
        """

        for df in self.music_data, self.music_data_accents:
            df['duration'] = df['duration'] * 2
        return self.music_data, self.music_data_accents

    def rescale_onsets(self):

        """
        Rescales onset column in MusicData.music_data & MusicData.music_data_accents dataframes
        from quarter notes to eighth notes.
        """

        for df in self.music_data, self.music_data_accents:
            df['onset'] = df['onset'] * 2
        return self.music_data, self.music_data_accents

    def calc_intervals(self):
        """Calculates chromatic interval relative to previous note event for each note event in pitch sequence"""
        for df in self.music_data, self.music_data_accents:
            df['interval'] = df['MIDI_note'] - df['MIDI_note'].shift(1)
            df['interval'].fillna(0, inplace=True)
        return self.music_data, self.music_data_accents

    def calc_parsons_codes(self):

        """
        Calculates Parsons code for all note events in feature sequence. Parsons code is a simple representation of
        melodic contour, formulated by Denys Parsons:
        'u' = upward melodic movement relative to previous note
        'd' = downward melodic movement relative to previous note
        'r' = repeated tone
        '*' = opening note event / reference tone.
        In our numeric version:
        1 = upward melodic movement
        -1 = downward melodic movement
        0 = repeated tone, and the opening note / reference tone.
        """

        for df in self.music_data, self.music_data_accents:
            prev_note = df['MIDI_note'].shift(1)
            # conditionally assign Parsons codes, comparing each note event with prev note:
            df['parsons_code'] = np.where(df['MIDI_note'] == prev_note, 0, np.where(df['MIDI_note'] > prev_note, 1, -1))
            # set first Parsons code value to 0
            df.loc[0, 'parsons_code'] = 0
        return self.music_data, self.music_data_accents

    def calc_parsons_cumsum(self):
        """Calculates cumulative sum of Parsons code sequence values (useful for graphing simple melodic contour)"""
        for df in self.music_data, self.music_data_accents:
            df['Parsons_cumsum'] = df['parsons_code'].cumsum()
        return self.music_data, self.music_data_accents

    def generate_duration_weighted_music_data(self, features):

        """
        Derives duration-weighted sequence for target feature sequences and returns output to
        MusicData.weighted_music_data dataframe.
        Duration weighting re-indexes sequence data from one value per note event in MusicData.music_data to
        one value per eighth note in MusicData.weighted_music_data.
        Target feature names (i.e.: names of columns to be processed) must be passed as list to 'features' arg.
        """

        self.weighted_music_data = pd.DataFrame()
        # derive new index:
        # instead of indexing by note-event, we will now index per eighth note
        # new index of eighth notes derived from 'onsets' column:
        onsets = self.music_data['onset'].to_numpy()
        eighths = np.arange(int(onsets[0]), int(onsets[-1]) + 1)
        idx = np.searchsorted(onsets, eighths)

        for feat in features:
            feat_seq = self.music_data[feat].to_numpy()
            # target feature sequence columns rescaled to new index:
            dur_weighted_seq = [feat_seq[i] for i in idx]
            self.weighted_music_data[f'dur_weighted_{feat}'] = dur_weighted_seq

        # return results to 'weighted_music_data' attribute:
        self.weighted_music_data = self.weighted_music_data.rename_axis('eighth_note')
        return self.weighted_music_data

    def extract_duration_weighted_music_data_accents(self):

        """Filters duration-weighted feature sequence data stored at MusicDataCorpus.weighted_music_data attr,
        returns output to MusicDataCorpus.weighted_music_data_accents"""

        self.weighted_music_data_accents = utils.filter_dataframe(self.weighted_music_data, seq='dur_weighted_velocity')
        return self.weighted_music_data_accents


class MusicDataCorpus(Music21Corpus):

    """
    MusicDataCorpus class inherits from Music21Corpus class, and holds a corpus of many
    MusicData objects. Through access to MusicData methods, and supplementary Music21Corpus methods,
    the class allows flexible corpus-level calculation and storage of secondary feature sequence data.
    It is instantiated with a single argument, a Music21Corpus object.
    Attributes per Music21Corpus, plus:
        corpus -- list holding a MusicData object for every melody in a Music21Corpus.corpus dictionary.
        roots-- empty attribute to be filled with a table of root notes for all melodies in corpus, read from external
        csv file via MusicDataCorpus.read_root_data() method.
    """

    def __init__(self, m21_corpus):
        super().__init__(inpath=m21_corpus.inpath)
        self.corpus = [MusicData(melody) for melody in m21_corpus.corpus.items()]
        self.roots = None

    def rescale_corpus_durations(self):
        """Applies MusicData.rescale_duration() to all melodies in corpus"""
        for melody in self.corpus:
            melody.rescale_durations()
        print(f"\nMusicDataCorpus.rescale_corpus_durations() sample outputs for {self.titles[0]}:")
        self.print_sample_output()
        return self.corpus

    def rescale_corpus_onsets(self):
        """Applies MusicData.rescale_onsets() to all melodies in corpus"""
        for melody in self.corpus:
            melody.rescale_onsets()
        print(f"\nMusicDataCorpus.rescale_corpus_onsets(): sample outputs for {self.titles[0]}:")
        self.print_sample_output()
        return self.corpus

    def calc_corpus_intervals(self):
        """Applies MusicData.calc_intervals() to all melodies in corpus"""
        for melody in self.corpus:
            melody.calc_intervals()
        print(f"\nMusicDataCorpus.calc_corpus_intervals(): sample outputs for {self.titles[0]}:")
        self.print_sample_output()
        return self.corpus

    def calc_corpus_parsons(self):
        """Applies MusicData.calc_parsons_code() to all melodies in corpus"""
        for melody in self.corpus:
            melody.calc_parsons_codes()
        print(f"\nMusicDataCorpus.calc_corpus_parsons(): sample outputs for {self.titles[0]}:")
        self.print_sample_output()
        return self.corpus

    def calc_corpus_parsons_cumsum(self):
        """Applies MusicData.calc_parsons_cumsum() to all melodies in corpus"""
        for melody in self.corpus:
            melody.calc_parsons_cumsum()
        print(f"\nMusicDataCorpus.calc_corpus_parsons_cumsum(): sample outputs for {self.titles[0]}:")
        self.print_sample_output()
        return self.corpus

    def find_most_freq_notes_in_melodies(self):

        """For each melody in the corpus, this method finds mode values for note-level, accent-level, duration-weighted,
        and duration-weighted accent-level pitch sequences (all represented by MIDI note number). This data is saved to
        MusicDataCorpus.roots_df dataframe and is used as input for root_key_detection component."""

        # find most frequent pitch (MIDI note number) for each melody in corpus:
        freq_notes = [melody.music_data.mode()['MIDI_note'][0] for melody in self.corpus]
        # find most frequent duration-weighted pitch (MIDI note number) for each melody in corpus:
        freq_weighted_notes = [melody.weighted_music_data.mode()['dur_weighted_MIDI_note'][0]
                                    for melody in self.corpus]
        # find most frequent accented pitch (MIDI note number) for each melody in corpus:
        freq_accents = [melody.music_data_accents.mode()['MIDI_note'] for melody in self.corpus
                             if melody.music_data_accents is not None]
        # find most frequent duration-weighted pitch accent pitch (MIDI note number) for each melody in corpus:
        freq_weighted_accents = [melody.weighted_music_data_accents.mode()['dur_weighted_MIDI_note']
                                      for melody in self.corpus if melody.weighted_music_data_accents is not None]

        # Add values calculated above to self.roots_df dataframe:
        self.roots_df['freq note'] = freq_notes
        self.roots_df['freq weighted note'] = freq_weighted_notes
        self.roots_df['freq note'] = self.roots_df['freq note'] % 12
        self.roots_df['freq weighted note'] = self.roots_df['freq weighted note'] % 12

        if freq_accents:
            self.roots_df['freq acc'] = freq_accents
            self.roots_df['freq acc'] = self.roots_df['freq acc'] % 12
        if freq_weighted_accents:
            self.roots_df['freq weighted acc'] = freq_weighted_accents
            self.roots_df['freq weighted acc'] = self.roots_df['freq weighted acc'] % 12

        print("\nFrequent note data added to root detection metrics table:")
        print(self.roots_df.head())
        return self.roots_df

    def append_expert_assigned_root_values(self, path):

        """Reads expert-assigned root values from an external csv file, containing one root note value
        (chromatic pitch class) per melody in the corpus. This data is added to MusicDataCorpus.roots_df dataframe, and
        is used as input for root_key_detection component."""

        expert_assigned_roots = pd.read_csv(path)
        expert_assigned_roots.set_index('title', inplace=True, drop=True)
        expert_assigned_roots['expert assigned'] = pd.to_numeric(expert_assigned_roots['expert assigned'])
        expert_assigned_roots['expert assigned'] = expert_assigned_roots['expert assigned'].round()
        print("\nReading expert-assigned root data:")
        print(expert_assigned_roots.head())
        print("\nAppending expert-assigned root data to root detection metrics table:")
        self.roots_df = self.roots_df.join(expert_assigned_roots)
        print(self.roots_df.head())
        return self.roots_df

    def read_root_data(self, roots_path):

        """
        Reads root notes csv table from location specified in 'roots_path' arg.
        This table must contain a column named 'root' containing a root note value for every melody in the corpus,
        expressed as a chromatic pitch classes (C = 0 through B# = 11).
        It must also contain a 'title' column of melody titles , formatted as per MusicData.title
        (i.e.: full filename without filetype suffix).
        """

        self.roots = pd.read_csv(roots_path)
        self.roots.reset_index(inplace=True, drop=True)
        print("\nImporting root data for all melodies in corpus:")
        print(self.roots.head())
        return self.roots

    def convert_roots_to_midi_note_nums(self):

        """
        Converts root values from chromatic pitch classes (C=0 through B=11) to 4th octave MIDI note numbers,
        corresponding to the core pitch range of Irish traditional repertoire and instrumentation (C=60 through B=71)
        Appends resultant values to MusicDataCorpus.roots in new 'MIDI_root' column.
        """

        roots = self.roots.copy()
        roots['MIDI_root'] = roots['root'].map(constants.lookup_table.set_index('root num')['midi num'])
        self.roots = roots
        print("\nRoot values for all melodies in corpus:")
        print(self.roots.head())
        return self.roots

    def assign_roots(self):

        """
        Assigns a MIDI root value to every melody in the corpus from 'MIDI_root' column in MusicDataCorpus.roots
        lookup table
        """

        for melody in self.corpus:
            # lookup root value from self.roots by melody title:
            midi_root = self.roots[self.roots['title'] == melody.title]['MIDI_root']
            midi_root = int(midi_root)
            for df in melody.music_data, melody.music_data_accents:
                df['MIDI_root'] = midi_root
                df['chromatic_root'] = midi_root % 12
        print(f"\nMusicDataCorpus.assign_roots(): sample outputs for {self.titles[0]}:")
        self.print_sample_output()
        return self.corpus

    def calc_key_invariant_pitches(self):

        """Uses root value assigned by MusicDataCorpus.assign_roots() to calculate key-invariant pitch sequences
        for all melodies in corpus, relative to 4th octave MIDI note numbers (C=60 through B=71).
        """

        for melody in self.corpus:
            for df in melody.music_data, melody.music_data_accents:
                df['pitch'] = df['MIDI_note'] - df['MIDI_root']
                df.drop(columns='MIDI_root', inplace=True)
        print(f"\nMusicDataCorpus.calc_key_invariant_pitches(): sample outputs for {self.titles[0]}:")
        self.print_sample_output()
        return self.corpus

    def calc_pitch_classes(self):

        """Derives pitch class sequences from pitch sequences calculated by
        MusicDataCorpus.calc_key_invariant_pitches() for all melodies in corpus"""

        for melody in self.corpus:
            for df in melody.music_data, melody.music_data_accents:
                df['pitch_class'] = df['pitch'] % 12
        print(f"\nMusicDataCorpus.calc_pitch_classes(): sample outputs for {self.titles[0]}:")
        self.print_sample_output()
        return self.corpus

    def calc_duration_weighted_feat_seqs(self, features):

        """
        Derives duration-weighted sequence for target features across all melodies in corpus.
        Target features (i.e.: column names) must be passed as list to 'features' arg.
        Output for each individual melody is stored as dataframe at MusicData.weighted_music_data.
        """

        for melody in self.corpus:
            melody.generate_duration_weighted_music_data(features)
        print(f"\nMusicDataCorpus.calc_duration_weighted_pitch_classes(): sample outputs for {self.titles[0]}:")
        print(f"{self.corpus[0].weighted_music_data.head()}\n\n")
        return self.corpus

    def print_sample_output(self):

        print("\nSample feature sequence output (note-event level):")
        print(f"{self.corpus[0].music_data.head()}\n")
        print("Sample feature sequence output (accent-level):")
        print(f"{self.corpus[0].music_data_accents.head()}\n\n")

    def save_corpus_data_to_csv(self, feat_seq_path, accents_path, duration_weighted_path):

        """Allows automatic saving and labelling of output csv files:
        feat_seq_path -- path to directory for note-level feature sequence data files for all melodies in corpus
        accents_path -- path to directory for accent-level feature sequence data files for all melodies in corpus
        duration_weighted_path -- path to directory for duration-weighted data for all melodies in corpus
        """

        for melody in self.corpus:
            paths_data = {
                feat_seq_path: melody.music_data,
                accents_path: melody.music_data_accents,
                duration_weighted_path: melody.weighted_music_data
            }

            for pa in paths_data:
                filename = f"{melody.title}_{pa.split('/')[-1]}"
                utils.write_to_csv(paths_data[pa], pa, filename)
                print(f"Writing output: {pa}/{filename}.csv")

        return None


def main():
    print('Running corpus_processing_tools.py')


if __name__ == "__main__":
    main()

