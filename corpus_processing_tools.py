# TODO: Review approach to targeting of attrs to reduce code duplication.

# TODO: Fix duration rounding. [possibly an ABC issue?]

# TODO: Store pattern indices

# TODO: Add MIDI ingest and filtering. Mostly done but not documented or tested.

# TODO: Add absolute pitch classes, diatonic note nums / pitches, diatonic pitch classes, diatonic intervals.

# TODO: Remove roots.csv requirement -- unnecessary in The Session and can be rendered unnecessary in CRE via key
#  annotation.

"""
Tune class objects represent a single tune in feature sequence format.
Corpus class objects represent feature sequence data for all tunes in a cre_corpus.

Together these classes allow extraction of primary and secondary feature sequence data from cre_corpus MIDI files.
To calculate key-invariant secondary features, an external table of root note values for all tunes must be provided.
If such a table does not exist, these classes can calculate multiple root note detection metrics, which can be inputted
into the separate 'Root Note Detection' component to model and return a table of root notes.

Primary feature sequences calculable via Tune and Corpus classes, with feature names as used throughout the toolkit:
- 'midi_note_num': MIDI note number
- 'onset': note onset (1/4 notes)
- 'duration': note (1/4 notes)
- 'velocity': MIDI velocity

Secondary feature sequences calculable via Tune and Corpus classes:
- 'onset': rescaled note onset (1/8 notes)
- 'duration': rescaled duration (1/8 notes)
- 'interval': chromatic interval between successive notes.
- 'root': scalar MIDI note number value representing the root or tonal centre of a tune, which is read from external
data table converted to a key-invariant pitch class.
- 'relative_pitch': key-invariant chromatic pitch.
- 'relative_pitch_class': key-invariant chromatic pitch class.
- 'parsons_code': simple melodic contour. Please see Tune.calc_parsons_codes() docstring for detailed explanation.
- 'parsons_cumsum': cumulative Parsons code values (convenient for graphic representation of contour).

These classes also allow filtering of the feature sequences at accent-level
(retaining data only for note events which occur on accented beats);
and generation of duration-weighted feature sequences (combining feature and duration values in a single
sequence, which represents pitch class value per eighth note rather than per note event).
'"""

import music21
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

import constants
import utils

pd.options.mode.chained_assignment = None


class Tune:
    
    """
    Tune class objects represent a single tune in feature sequence format. A Tune object can be instantiated
    via the 'inpath' argument, which must point to a single monophonic MIDI file. In our usage, Tune objects are
    automatically created in bulk at cre_corpus-level when a Corpus object is instantiated.

    Attributes:

    inpath -- path to a monophonic MIDI file.
    title -- Extracts and stores tune title text from MIDI filename; populates automatically on instantiation.
    score -- Holds a music21 Stream representation of the MIDI file; populates automatically on instantiation.
    score_accents -- Empty attr to hold a filtered accent-level music21 Stream representation of the MIDI file; 
    populated via Corpus.filter_music21_score().
    feat_seq -- Empty attr to hold feature sequence representation of the data held in 'score' attr.
    feat_seq_no_pickups -- Empty attr to hold feature sequence representation of the data held in 'score' attr, from
    which pickup bars aka anacruses have been removed.
    feat_seq_accents -- Empty attr to hold accent-level feature sequence representation, which is a filtered version of
    'feat_seq' attr, retaining only data for rhythmically-accented notes.
    duration_weighted -- Empty attr to hold 'duration-weighted' sequences for selected features. Duration-weighting is
    explained below.
    duration_weighted_accents -- Empty attr to hold 'duration-weighted' sequences for selected features at accent-level.
    """

    def __init__(self, inpath):

        """
        Initializes Tune object.

        Args:
            inpath -- path to a monophonic MIDI file.
        """

        self.inpath = inpath
        # Extract tune title from filename:
        self.title = inpath.split('/')[-1][:-4]
        # automatically read MIDI data to music21 Stream on instantiation:
        self.score = music21.converter.parse(f"{self.inpath}")
        self.score_accents = None
        self.feat_seq = None
        self.feat_seq_accents = None
        self.duration_weighted = None
        self.duration_weighted_accents = None
        self.chromatic_root = None
        self.diatonic_root = None

    def __iter__(self):
        return self

    def _get_attrs(self):

        # TODO: docstring ; comment ; possibly refactor?

        return [val for attr, val in self.__dict__.items() if isinstance(val, pd.DataFrame)]

    def extract_root(self):

        # Parse score for music21 Key objects and extract their tonics:
        roots = [key.tonic for key in self.score.recurse().getElementsByClass(music21.key.Key)]
        # Retain first root note name from list above, or set root to NaN if MIDI file does not contain any root data.
        self.chromatic_root = int(roots[0].ps) if roots != [] else np.NaN
        self.diatonic_root = roots[0].diatonicNoteNum

    def extract_root_metrics(self):

        # TODO: Refactor to replace lookups of pitch classes etc with direct access via music21 (if possible)...

        """
        Calculates root note metrics accessible via music21 library:

        1. Pitch class of root note read from music21 key signature.
        2. Pitch class of final note of the tune.
        3. Pitch class of root as outputted by Krumhansl-Schmukler algorithm
        4. Pitch class of root as outputted using Craig Sapp's 'simple weights'
        5. Pitch class of root as outputted by Aarden Essen algorithm
        6. Pitch class of root as outputted by Bellman-Budge algorithm
        7. Pitch class of root as outputted by Temperley-Kostka-Payne algorithm
        
        Returns list of all results.
        """

        # Parse score for music21 Key objects and list their tonic (i.e.: root) note names:
        roots = [key.tonic.name.upper() for key in self.score.recurse().getElementsByClass(music21.key.Key)]
        # Retain first root note name from list above, or set root to NaN if MIDI file does not contain any root data.
        root = roots[0] if roots != [] else np.NaN

        # Extract final note of the tune:
        final_note_event = self.score.recurse().notes[-1]
        if final_note_event.isNote:
            final_note_name = final_note_event.name.upper()
        elif final_note_event.isChord:
            final_note_name = final_note_event.root().name.upper()
        else:
            final_note_name = ''

        # Run music21's default key detection algorithm (Krumhansl-Schmuckler), and return root note name:
        krumhansl = self.score.analyze('key').tonic.name.upper()
        # Run music21's built-in alternative key detection algorithms and return root note name for each:
        analyse_key = music21.analysis.discrete
        simple = analyse_key.SimpleWeights(self.score).getSolution(self.score).tonic.name.upper()
        aarden = analyse_key.AardenEssen(self.score).getSolution(self.score).tonic.name.upper()
        bellman = analyse_key.BellmanBudge(self.score).getSolution(self.score).tonic.name.upper()
        temperley = analyse_key.TemperleyKostkaPayne(self.score).getSolution(self.score).tonic.name.upper()
        root_metrics = [root, final_note_name, krumhansl, simple, aarden, bellman, temperley]

        # Manual memory clearing:
        self.score = None

        return root_metrics

    def convert_music21_streams_to_feature_sequences(self, level=None):
        
        """
        Extracts primary feature sequence data from music21 Stream representation of a tune.
        Outputs a pandas Dataframe containing MIDI note number, onset, duration, and MIDI velocity values for each note 
        in the tune.

        Args:
            attr -- name of Tune class instance attribute to be passed as input for processing.
            Can be either 'score' (default, outputs note-level feature sequence data) or 'score_accents' (outputs accent
            -level feature sequence data)
        """

        if level is 'note':
            target = self.score
        elif level is 'accent':
            target = self.score_accents

        if not isinstance(target, music21.stream.Score):
            print("Tune.convert_music21_streams_to_feature_sequences() input must be a music21 Stream object")
            pass

        # empty dict to hold feature sequence data:
        feat_seq_data = {}
        # read all notes in music21 Stream and extract their pitches represented as MIDI note numbers:
        for idx, note in enumerate(target.recurse().notes):
            if note.isNote:
                midi_note = int(note.pitch.ps)
                diatonic_note_num = int(note.pitch.diatonicNoteNum)
                pitch_class = int(note.pitch.pitchClass)
            # if chords are encountered, take their root: (this is rare in our corpora)
            elif note.isChord:
                midi_note = int(note.root().ps)
                diatonic_note_num = int(note.root().diatonicNoteNum)
                pitch_class = int(note.root().pitchClass)

            # for each note, extract primary feature data and store all feature values in a list:
            note_event_data = [
                midi_note,                                          # MIDI (chromatic) note number
                diatonic_note_num,                                  # Diatonic note number
                pitch_class,                                        # (absolute) chromatic pitch class
                round(float(note.offset) * 2, 2),                   # onset
                round(float(note.duration.quarterLength) * 2, 2),   # duration
                note.volume.velocity                                # MIDI velocity
            ]
            # Add data for each note to 'feat_seq_data' dict:
            feat_seq_data[idx] = note_event_data

        # convert to Dataframe
        output = pd.DataFrame.from_dict(feat_seq_data, orient='index',
                                               columns=["midi_note_num",
                                                        "diatonic_note_num",
                                                        "abs_chromatic_pitch_class",
                                                        "onset",
                                                        "duration",
                                                        "velocity"
                                                        ], dtype='int16')

        if level is 'note':
            self.feat_seq = output
        elif level is 'accent':
            self.feat_seq_accents = output

        return output

    def filter_score_accents(self, thresh=0.5):

        """Filters Tune.score, retaining only accented notes via an adjustable beat strength threshold.

        args:
            thresh -- threshold music21 Note.beatStrength attr value by which the score is filtered. Only Note objects
            with beatStrength >= thresh will be retained in the filtered output.

        To filter and extract heavily-accented notes (i.e.: the most prominent note in each bar) from a score,
        set thresh=1
        To filter and extract accented notes (i.e.: one note per beat) from a score, set thresh=0.5

        NOTE: Scores can also be filtered after conversion to feature sequence format via
        Corpus.filter_feat_seq_accents() method. The alternative method is faster and less memory-intensive, but will
        only filter accurately on data originating from ABC Notation files generated by abc2midi per abc_ingest.py.
        For MIDI inputs please use this method to ensure accurate output.
        """

        filtered_score = music21.stream.Score()
        for idx, note in enumerate(self.score.recurse().notes):
            if note.isNote and float(note.beatStrength) >= thresh:
                filtered_score.append(note)
        self.score_accents = filtered_score
        return self.score_accents
        
    def add_bar_count(self):

        # TODO: Docstring

        """Adds bar count column to feature sequence DataFrame stored in Tune.feat_seq attr."""

        # select all feature sequence DataFrames from Tune instance attrs:
        targets = self._get_attrs()

        for t in targets:
            # create bar counter column via MIDI velocity values:
            bar_lines = np.where(np.logical_and(t['velocity'] == 105, t.index > 0), True, False)
            t['bar_count'] = bar_lines.cumsum()
                 
    def add_abs_diatonic_pitch_class(self):

        # TODO: docstring

        targets = self._get_attrs()
        for t in targets:
            t["diatonic_pitch_class"] = t["diatonic_note_num"] % 7
            
    def add_diatonic_interval(self):

        # TODO: docstring

        targets = self._get_attrs()
        for t in targets:
            t['diatonic_interval'] = (t['diatonic_note_num'] - t['diatonic_note_num'].shift(1)).fillna(0).astype('int16')

    def add_rel_diatonic_pitch(self):

        # TODO: Docstring

        # select all feature sequence DataFrames from Tune instance attrs:
        targets = self._get_attrs()
        for t in targets:
            t['relative_diatonic_pitch'] = (t['diatonic_note_num'] - self.diatonic_root).astype('int16')

    def add_rel_diatonic_pitch_class(self):

        # TODO: Docstring

        # select all feature sequence DataFrames from Tune instance attrs:
        targets = self._get_attrs()
        for t in targets:
            t['relative_diatonic_pitch_class'] = (t['relative_diatonic_pitch'] % 12).astype('int16')

    @staticmethod
    def strip_anacruses(*args):

        # TODO: Docstring

        """Applies heuristic to check if first bar is less than 1/2 the length of second bar.
        If so, first bar is taken as a pickup and is dropped from Tune.feat_seq inplace"""

        for a in args:
            if isinstance(a, pd.DataFrame):
                # calculate duration of first and second bars
                bar1_duration = a[a['bar_count'] == 0]['duration'].sum()
                bar2_duration = a[a['bar_count'] == 1]['duration'].sum()

                # divide bar durations and remove first bar if heuristic indicates that it is a pickup
                a = a[a['bar_count'] > 0] if (bar1_duration / bar2_duration) < 0.5 else a
            else:
                print("Tune.strip_anacruses() input must be a pandas DataFrame object")

    def reindex_bars(self):

        # TODO: Test ; docstring
        targets = self._get_attrs()
        for t in targets:
            if t.loc[0, 'barcount'] == 0:
                t['bar_count'] += 1


    def filter_feat_seq_accents(self):

        """
        Filters input feature sequence DataFrame, retaining data for rhythmically-accented
        notes only. This data is stored as a separate Dataframe at Tune.feat_seq_accents attr. It does not include data
        for anacruses /pickup measures.

        In the corpora under investigation, MIDI velocity is used to indicate rhythm and structure following a
        consistent 'stress model' applied using ABC2MIDI in the preliminary conversion from ABC notation to MIDI format.
        In this model, all notes with MIDI velocity values above 80 are rhythmically-accented.
        
        Args:
            -- attr: select input DataFrame from instance attributes by name. 
            Can be either 'feat_seq' or 'feat_seq_no_pickups'.
        """

        # Filter feature sequence data via MIDI velocity threshold:
        self.feat_seq_accents = self.feat_seq[self.feat_seq['velocity'] > 80]

    @staticmethod
    def calc_chromatc_intervals(*args):

        """Calculates chromatic interval relative to previous note for all note events; adds interval values to
        Tune.feat_seq feature sequence Dataframe."""

        for a in args:
            # calculate intervals:
            a['chromatic_interval'] = (a['midi_note_num'] - a['midi_note_num'].shift(1)).fillna(0).astype('int16')

    @staticmethod
    def calc_parsons_codes(*args):

        """
        Calculates Parsons code for all notes in note- and accent-level feature sequences.
        Parsons code is a simple representation of melodic contour, formulated by Denys Parsons:
        'u' = upward melodic movement relative to previous note
        'd' = downward melodic movement relative to previous note
        'r' = repeated tone
        '*' = opening note event / reference tone.
        In our numeric version:
        1 = upward melodic movement
        -1 = downward melodic movement
        0 = repeated tone, and the opening note / reference tone.
        """

        for a in args:
            prev_note = a['midi_note_num'].shift(1)
            # conditionally assign Parsons codes, comparing each note event with prev note:
            a['parsons_code'] = np.where(a['midi_note_num'] == prev_note, 0,
                                              np.where(a['midi_note_num'] > prev_note, 1, -1))
            a['parsons_code'] = a['parsons_code'].astype('int16')
            # set first Parsons code value to 0
            a.loc[0, 'parsons_code'] = 0

    @staticmethod
    def calc_parsons_cumsum(*args):

        """Calculates cumulative sum of Parsons code sequence values at note- and accent-level
         (useful for graphing simple melodic contour)"""

        for a in args:
            # Calculate Parsons code coumsum:
            a['parsons_cumsum'] = a['parsons_code'].cumsum().astype('int16')

    def add_rel_chromatic_pitch(self):

        """
        Uses root value assigned by Corpus.assign_roots() method to calculate key-invariant pitch sequence
        for tune at note- or accent-level.
        """

        # select all feature sequence DataFrames from Tune instance attrs:
        targets = self._get_attrs()
        for t in targets:
            t['relative_chromatic_pitch'] = (t['midi_note_num'] - self.chromatic_root).astype('int16')

    def add_rel_chromatic_pitch_class(self):

        """Derives pitch class sequence from pitch sequence calculated by
        MusicDataCorpus.calc_relative_pitch()"""

        # select all feature sequence DataFrames from Tune instance attrs:
        targets = self._get_attrs()
        for t in targets:
            t['relative_chromatic_pitch_class'] = (t['relative_chromatic_pitch'] % 12).astype('int16')

    def apply_duration_weighting(self, *args, features: list = None):

        """
        Calculates duration-weighted sequence for selected feature sequences and returns output to pandas Dataframe at
        Tune.duration_weighted. Feature names can be passed to 'features' arg as a list.

        Duration weighting re-indexes sequence data, converting from feature value per note event to
        one value per eighth note.
        """

        for a in args:
            duration_weighted = pd.DataFrame()
            # new index of eighth notes from 'onsets' column:
            onsets = a['onset'].to_numpy()
            eighth_notes = np.arange(onsets[0], onsets[-1])
            idx = np.searchsorted(onsets, eighth_notes)
    
            for feat in features:
                feat_seq = a[feat].to_numpy()
                # a feature sequence columns rescaled to new index:
                dur_weighted_seq = [feat_seq[i] for i in idx]
                duration_weighted[f'dur_weighted_{feat}'] = dur_weighted_seq
                duration_weighted[f'dur_weighted_{feat}'] = duration_weighted[f'dur_weighted_{feat}']
    
            # return results to 'Tune.duration_weighted' attribute & set type as int16:
            duration_weighted = duration_weighted.rename_axis('eighth_note').astype('int16', errors='ignore')
    
            if a is self.feat_seq:
                self.duration_weighted = duration_weighted
            elif a is self.feat_seq_accents:
                self.duration_weighted_accents = duration_weighted

    def filter_duration_weighted_accents(self):

        # TODO: Currently this makes use of the abc2midi stress model and, as such, only works on input data processed
        #  via non-abc2midi.

        """
        Filters duration-weighted feature sequence data stored at Corpus.duration_weighted, retaining only
        accent-level data. Output is stored as Tune.duration_weighted_accents attr. Note: works on data with abc2midi
        'stress model' encoding but not for data from MIDI sources. For such MIDI-originating data, please use
        Tune.apply_duration_weighting() above."""

        if 'dur_weighted_velocity' in self.duration_weighted.columns:
            self.duration_weighted_accents = utils.filter_dataframe(
                self.duration_weighted, seq='dur_weighted_velocity')
        else:
            pass


class Corpus:

    """
    A Corpus object is instantiated with a single argument, 'inpath', which can be the path to either a directory
    containing a cre_corpus of monophonic MIDI files, or to an existing Corpus object in .pkl format.

    Attributes:
        tunes -- providing an 'inpath' argument automatically instantiates a Tune object for
        each tune in the cre_corpus; 'tunes' attr holds a list of these Tune objects.
        titles -- a list containing the text string titles of all Tune objects in cre_corpus. This attr is auto-populated
        when a Corpus object is instantiated.
        roots -- empty attr to hold table of root note values, which can either be calculated or read from an
        external file.
        roots_lookup -- empty attr to hold reformatted roots table, with root values converted to MIDI note numbers.
        roots_path -- empty attr to hold string path at which roots Dataframe can be saved to pkl and csv files.
        pkl_outpath -- empty attr to hold string path at which cre_corpus can be saved as a .pkl file.
        csv_outpath -- empty attr to hold string path at which cre_corpus can be saved as a .csv file. Not recommended for
        large corpora (10k + tunes).
    """

    def __init__(self, inpath):

        """
        Initializes Corpus object.

        Args:
            inpath -- path to a directory of monophonic MIDI files.
        """

        self.tunes = self.setup_corpus(inpath)
        # Please see docstring for notes on all attrs below
        self.titles = [tune.title for tune in self.tunes]
        self.roots = None
        self.roots_lookup = None
        self.roots_path = None
        self.root_metrics = None
        self.root_metrics_path = None
        self.pkl_outpath = None
        self.csv_outpath = None

    def setup_corpus(self, inpath):

        # TODO: docstring

        # if 'inpath' arg points to a directory, create self.tunes attr and populate with Tune objects for each file:
        if os.path.isdir(inpath):
            filenames = os.listdir(inpath)
            self.tunes = [Tune(f"{inpath}/{file}") for file in tqdm(
                filenames, desc="Reading cre_corpus MIDI files to Music21 streams") if file.endswith('.mid')]
        # if 'inpath' is a pre-existing pickled Corpus object, read the file:
        elif os.path.isfile(inpath) and inpath.endswith('.pkl'):
            with open(inpath, 'rb') as f_in:
                self.tunes = pickle.load(f_in)
        else:
            print("Corpus objects can only be instantiated in two ways: 1: From a directory of MIDI files, or "
                  "2. From a Corpus.tunes object stored in a pickle (.pkl) file.")

        return self.tunes

    def filter_empty_scores(self):

        """The Session cre_corpus contains some 'experimental' pieces of music which are simply blank scores.
        This method filters out such scores before feature sequence calculations.
        """

        for tune in self.tunes:
            if len(tune.score) == 0:
                print(f"No input data for {tune.title}")
                self.tunes.remove(tune)
                self.titles.remove(tune.title)

    def extract_roots(self):

        for tune in tqdm(self.tunes, desc='Extracting chromatic and diatonic roots'):
            tune.extract_root()

    def calculate_root_metrics(self):

        """Runs Tune.extract_root_metrics() for each Tune object in Corpus.tunes. The results are held in a
        cre_corpus-level Dataframe at Corpus.roots."""

        metrics_lst = []
        for tune in tqdm(self.tunes, desc='Calculating root note metrics'):
            root_metrics = tune.extract_root_metrics()
            metrics_lst.append(root_metrics)
            # Manual garbage collection:
            tune.score = None

        # output of above loop held in dict with a key-val pair for each tune formatted per: tune title: root metrics.
        # this dict is then converted to a Dataframe.
        metrics_dict = dict(zip(self.titles, metrics_lst))
        col_names = ['as transcribed', 'final note', 'Krumhansl-Schmuckler', 'simple weights', 'Aarden Essen', 'Bellman Budge',
                     'Temperley Kostka Payne']
        roots = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=col_names)
        self.root_metrics = roots

    def filter_score_accents(self, thresh=0.5):

        """
        Runs Tune.filter_music21_scores() for each Tune object in Corpus.tunes. This filters Tune.score by an
        adjustable beat strength threshold value, retaining only rhythmically-accented notes.

        args:
            thresh -- threshold music21 Note.beatStrength attr value by which the score is filtered. Only Note objects
            with beatStrength >= thresh will be retained in the filtered output.

        To filter and extract heavily-accented notes (i.e.: the most prominent note in each bar) from a score,
        set thresh=1
        To filter and extract accented notes (i.e.: one note per beat) from a score, set thresh=0.5
        """

        for tune in tqdm(self.tunes, desc=f'Filtering music21 scores: beat strength threshold = {thresh}'):
            tune.filter_score_accents(thresh)

    def convert_scores_to_feat_seqs(self, level=None):

        """Runs Tune.convert_to_feature_sequence_repr() for every Tune object in Corpus.tunes"""

        for tune in tqdm(self.tunes, desc=f'Calculating {level}-level feature sequences from music21 scores'):
            tune.convert_music21_streams_to_feature_sequences(level=level)

    def filter_feat_seq_accents(self):

        # TODO: Docstring

        for tune in tqdm(self.tunes, desc='Extracting accent-level feature sequence data'):
            tune.filter_feat_seq_accents()

    def convert_note_names_to_pitch_classes(self):

        # TODO: Evaluate whether this can be scrapped in light of potential to extract pitch classes via music21.
        #  If it must be retained, move to new class below.


        """
        If Corpus.root_metrics dataframe contains root note names formatted per music21
        standard (i.e.: "G-" = G natural; "G#" = G sharp), this method will convert them to integer chromatic pitch
        classes (values from 0-11), via constants.music21_lookup_table Dataframe.

        NOTE: Used in root note detection setup but not in cre_corpus setup.
        """

        print('\n')
        print(f"Converting note names to integer pitch classes...\n")
        # lookup numeric root values & map to new column:
        lookup = dict(zip(constants.music21_lookup_table['note name'], constants.music21_lookup_table['pitch class']))
        self.root_metrics = self.root_metrics.replace(lookup).astype('int16')
        print(self.root_metrics.head())
        print(self.root_metrics.info())
        print('\n')

    def save_root_metrics_table(self):

        # TODO: Move to new class below.

        """Writes Corpus.roots Dataframe to csv file at path specified in Corpus.root_metrics_path attr."""

        print(f"Saving cre_corpus root metrics dataframe to: {self.root_metrics_path}")
        self.root_metrics.to_csv(self.root_metrics_path)

    def calc_intervals(self):
        """Applies Tune.calc_intervals() to all Tune objects in Corpus.tunes"""
        for tune in tqdm(self.tunes, desc='Calculating interval sequences'):
            tune.calc_intervals()

    def calc_parsons_codes(self):
        """Applies Tune.calc_parsons_code() to all Tune objects in Corpus.tunes"""
        for tune in tqdm(self.tunes, desc='Calculating Parsons code sequences'):
            tune.calc_parsons_codes()

    def calc_parsons_cumsum(self):
        """Applies Tune.calc_parsons_cumsum() to all Tune objects in Corpus.tunes"""
        for tune in tqdm(self.tunes, desc='Calculating cumulative Parsons code sequences'):
            tune.calc_parsons_cumsum()

    def find_most_freq_notes(self):

        """For each tune in the cre_corpus, this method finds modal values for note-level, accent-level, duration-weighted,
        and duration-weighted accent-level pitch sequences (all represented by MIDI note number). Output is saved to
        Corpus.roots Dataframe ans can be used as input data for separate 'Root Note Detection' component."""

        # find most frequent pitch (MIDI note number) for each tune in cre_corpus:
        freq_notes = []
        for tune in tqdm(self.tunes, desc='Calculating mode of MIDI note values for each tune'):
            if tune.feat_seq is None or len(tune.feat_seq) == 0:
                print(tune.title)
                freq_notes.append(np.NaN)
            else:
                freq_notes.append(tune.feat_seq.mode()['midi_note_num'][0] % 12)

        # find most frequent duration-weighted pitch (MIDI note number) for each tune in cre_corpus:
        freq_weighted_notes = []
        for tune in tqdm(self.tunes, desc='Calculating mode of duration-weighted MIDI note values for each tune'):
            if tune.duration_weighted is None or len(tune.duration_weighted) == 0:
                print(tune.title)
                freq_weighted_notes.append(np.NaN)
            else:
                freq_weighted_notes.append(int(tune.duration_weighted.mode()['dur_weighted_midi_note'][0] % 12))

        # find most frequent accented pitch (MIDI note number) for each tune in cre_corpus:
        freq_accents = []
        for tune in tqdm(self.tunes, desc='Calculating mode of accent-level MIDI note values for each tune'):
            if tune.feat_seq_accents is None or len(tune.feat_seq_accents) == 0:
                print(tune.title)
                freq_accents.append(np.NaN)
            else:
                freq_accents.append(tune.feat_seq_accents.mode()['midi_note_num'][0] % 12)

        # find most frequent duration-weighted pitch accent pitch (MIDI note number) for each tune in cre_corpus:
        freq_weighted_accents = []
        for tune in tqdm(self.tunes,
                         desc='Calculating mode of accent-level duration-weighted MIDI note values for each tune'):
            if tune.duration_weighted_accents is None or len(tune.duration_weighted_accents) == 0:
                print(tune.title)
                freq_weighted_accents.append(np.NaN)
            else:
                freq_weighted_accents.append(tune.duration_weighted_accents.mode()['dur_weighted_midi_note'][0] % 12)

        # Add values calculated above to self.root_metrics dataframe:
        self.root_metrics = pd.read_csv(self.root_metrics_path)
        self.root_metrics['freq note'] = freq_notes
        self.root_metrics['freq acc'] = freq_accents
        self.root_metrics['freq weighted note'] = freq_weighted_notes
        self.root_metrics['freq weighted acc'] = freq_weighted_accents
        self.root_metrics = self.root_metrics.astype('int16', errors='ignore', copy='False')
        print("\nFinal root detection metrics table:\n")
        print(self.root_metrics.head())
        print(self.root_metrics.info())

    def read_external_root_data(self):

        """
        Reads table of root note values for each tune from location specified in 'roots_path' arg.
        Table can be in either csv or pkl format.
        It must contain a column named 'root' containing a root note value for every tune in the cre_corpus,
        expressed as either a chromatic pitch classes (C = 0 through B# = 11) or a music21 note name.
        It must also contain a 'title' column of tune title text strings, matching the titles listed at Corpus.titles.

        This can be used in setting up a cre_corpus, specifically for calculating secondary features, in situations where
        root data is either absent or unreliable.
        """

        print(f"\nReading roots data from: {self.roots_path}\n")
        # read csv:
        if self.roots_path.endswith('.csv'):
            self.roots = pd.read_csv(self.roots_path, index_col=0)
            # self.roots.set_index('title', inplace=True)
        # read pkl:
        elif self.roots_path.endswith('.pkl'):
            self.roots = pd.read_pickle(self.roots_path)
        else:
            print("Roots tables can only be read from either pkl or csv files, "
                  "which must contain a 'title' col containing tune titles, and a 'root' col containing root "
                  "note values as chromatic pitch classes.")

        print(self.roots.head())
        print(self.roots.info())

    def convert_roots_to_midi_note_nums(self):

        # TODO: Potentially delete this method if no longer used?

        """
        Converts root values from chromatic pitch classes (C=0 through B=11) to 4th octave MIDI note numbers,
        corresponding to the core pitch range of Irish traditional repertoire and instrumentation (C=60 through B=71)
        Appends resultant values to Corpus.roots in new 'midi_root' column.
        """

        print('\nConverting root values from pitch classes to MIDI note numbers:\n')
        roots_lookup = self.roots.copy()
        # lookup 4th octave MIDI note value for each pitch class in 'root' column;
        # Add MIDI note values to new 'midi root' column:
        roots_lookup['midi_root'] = roots_lookup['root'].map(constants.lookup_table.set_index('root num')['midi num'])
        self.roots_lookup = roots_lookup
        print(self.roots_lookup.head())
        print(self.roots_lookup.info())
        print("\n")

    def assign_roots(self):

        # TODO: Potentially delete this method if no longer used?

        """Maps MIDI root value for every Tune in Corpus.tunes from 'midi_root' column in Corpus.roots_lookup table"""

        self.roots_lookup.set_index('title', inplace=True)

        for tune in tqdm(self.tunes, desc='Mapping root values to all tunes in cre_corpus from Corpus.roots_lookup'):
            # lookup root value from self.roots_lookup by tune title:
            midi_root = self.roots_lookup[self.roots_lookup.index == tune.title]['midi_root']
            # Calculate 'MIDI_root' and 'root' values, append to Tune.feat_seq as new columns.
            tune.feat_seq['midi_root'] = int(midi_root[0])
            tune.feat_seq['midi_root'] = tune.feat_seq['midi_root'].astype('int16', errors='ignore')
            tune.feat_seq['root'] = (tune.feat_seq['midi_root'] % 12).astype('int16', errors='ignore')

    def calc_pitch_class_seqs(self):

        """Converts MIDI note numbers to pitch classes for all Tune objects in Corpus.tunes. Adds as new column to
        Tune.feat_seq Dataframe."""

        for tune in tqdm(self.tunes, desc='Calculating pitch class sequences'):
            tune.feat_seq['pitch_class'] = (tune.feat_seq['midi_note_num'] % 12).astype('int16')

    def calc_relative_chromatic_pitch_seqs(self):
        """Runs Tune.calc_relative_pitch() for all Tune objects in Corpus.tunes."""
        for tune in tqdm(self.tunes, desc='Calculating relative pitch sequences'):
            tune.add_rel_chromatic_pitch()

    def calc_relative_chromatic_pitch_class_seqs(self):
        """Runs Tune.calc_relative_pitch_class() for all Tune objects in Corpus.tunes."""
        for tune in tqdm(self.tunes, desc='Calculating relative pitch class sequences'):
            tune.add_rel_chromatic_pitch_class()

    def calc_duration_weighted_feat_seqs(self, features):

        """
        Runs Tune.generate_duration_weighted_music_data() for all Tune objects in Corpus.tunes.

        Args:
            features -- list of musical features (i.e.: column names from Tune.feat_seq Dataframe) for which
            duration-weighted sequences are to be calculated.
        """

        for tune in tqdm(self.tunes, desc='Calculating duration-weighted feature sequences'):
            tune.generate_duration_weighted_music_data(features)

    def calc_duration_weighted_accent_seqs(self):
        """Runs Tune.extract_duration_weighted_music_data_accents() for all Tune objects in Corpus.tunes."""
        for tune in tqdm(self.tunes, desc='Calculating accent-level duration-weighted feature sequences'):
            tune.extract_duration_weighted_music_data_accents()

    def pickle_corpus(self):
        """Writes Corpus object to pkl file at location specified in Corpus.tunes_outpath."""
        f_out = open(self.pkl_outpath, 'wb')
        pickle.dump(self.tunes, f_out)
        print(f"\nCorpus data saved to pickle at: {self.pkl_outpath}")

    def save_feat_seq_data_to_csv(self):
        # TODO: Refactor or split into smaller more controllable functions to test for existence of each DataFrame and
        # save if present.

        """For all Tune objects in Corpus.tunes, this method creates a subdirectory structure at the path specified in
        Corpus.csv_outpath. It then writes the Dataframes at Tune.feat_seq, Tune.feat_seq_accents,
        Tune.duration_weighted and Tune.duration_weighted_accents to csv files in the appropriate folders."""

        # create subdirectories under location set by Corpus.csv_outpath attr (if they don't already exist):
        if not os.path.isdir(self.csv_outpath):
            os.makedirs(f"{self.csv_outpath}/feat_seq/")
            os.makedirs(f"{self.csv_outpath}/feat_seq_accents/")
            # os.makedirs(f"{self.csv_outpath}/duration_weighted/")
            # os.makedirs(f"{self.csv_outpath}/duration_weighted_accents/")

        # write data to file:
        for tune in tqdm(self.tunes, desc='Saving feature sequence data to csv'):
            if tune.feat_seq is not None and tune.feat_seq_accents is not None and len(tune.feat_seq_accents) > 2:
                tune.feat_seq.to_csv(f"{self.csv_outpath}/feat_seq/{tune.title}.csv")
                tune.feat_seq_accents.to_csv(f"{self.csv_outpath}/feat_seq_accents/{tune.title}.csv")
                # tune.duration_weighted.to_csv(f"{self.csv_outpath}/duration_weighted/{tune.title}.csv")
                # tune.duration_weighted.to_csv(f"{self.csv_outpath}/duration_weighted_accents/{tune.title}.csv")
            else:
                print(f"Missing output data for {tune.title}")


class RootMetrics(Corpus):

    # TODO: Populate this class from above

    def __init__(self, inpath):
        super().__init__(inpath)


