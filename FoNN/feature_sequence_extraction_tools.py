"""
Feature sequence data represents each note in a symbolic music document via numerical feature values, such
as: (midi_note_num: 68, offset: 10, duration: 2).

Tune class objects represent a single tune in this feature sequence format, which is extracted from music document files
 (i.e. digital scores) principally via the music21 Python library. Any file type compatible with music21 can be used to
initialize a Tune object.

Corpus class objects represent collections of multiple Tune objects.
Together, these classes allow extraction of primary and secondary feature sequence data from music score files.

Primary feature sequences calculable via Tune and Corpus classes, with feature names as used throughout the FoNN
toolkit:

-- 'midi_note_num': Chromatic pitch represented as MIDI number
-- 'offset': note offset (1/8 notes)
-- 'duration': note (1/8 notes)
-- 'velocity': MIDI velocity

Secondary feature sequences calculable via Tune and Corpus classes:

-- 'diatonic_note_num': Diatonic pitch
-- 'beat_strength' -- music21 beatStrength attribute
-- 'chromatic_pitch_class': Pitch normalised to a single octave, represented as an integer between 0-11.
-- 'bar_num': Bar number
-- 'relative_chromatic_pitch': Chromatic pitch relative to the root note or tonal centre of the input sequence.
-- 'relative_diatonic_pitch': Diatonic pitch relative to the root note or tonal centre of the input sequence.
-- 'chromatic_scale_degree': Chromatic pitch class relative to the root note or tonal centre of the input sequence.
-- 'diatonic_scale_degree': Diatonic pitch class relative to the root note or tonal centre of the input sequence.
-- 'chromatic_interval': Change in chromatic pitch between two successive notes in the input sequence
-- 'diatonic_interval': Change in diatonic pitch between two successive notes in the input sequence
-- 'parsons_code': simple melodic contour. Please see Tune.extract_parsons_codes() docstring for detailed explanation.
-- 'parsons_cumsum': cumulative Parsons code values.

These classes also allow:
-- Extraction of the 'root' or tonal centre from an input score.
-- Filtering of the feature sequences at accent-level (retaining data only for notes which occur on accented beats)
-- Generation of duration-weighted feature sequences (combining feature and duration values in a single
   sequence, which represents pitch class value per eighth note rather than per note event).
'"""

import music21
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

pd.options.mode.chained_assignment = None


class Tune:

    """
    Tune class object represents a single tune in feature sequence format. A Tune object can be instantiated
    via the 'in_path' argument, which must point to a single MIDI file. Tune objects can be created individually or
    can be automatically instantiated in bulk at corpus-level when a Corpus object is instantiated.

    Attributes:

    in_path -- path to a music score file in any music21-compatible format.
    title -- Tune title extracted from filename; populates automatically on instantiation.
    score -- music21 Stream representation of the music data contained in the input file; populates automatically on
             instantiation.
    feat_seq -- Feature sequence representation of the music information held in 'score' attr.
    accent -- Accent-level feature sequence data
    duration_weighted -- 'duration-weighted' sequences for selected features. Duration-weighting is
                          explained below.
    chromatic_root -- Chromatic note number representing root or tonal centre of input sequence.
    diatonic_root -- Diatonic note number representing root or tonal centre of input sequence.
    """

    def __init__(self, in_path):

        """
        Initializes Tune object.

        Args:
            in_path -- path to a symbolic music file in any format parseable by music21 library.
        """

        self.in_path = in_path
        # Extract tune title from filename:
        self.title = in_path.split('/')[-1][:-4]
        # automatically read music data to music21 Stream on instantiation:
        self.score = music21.converter.parse(f"{self.in_path}")
        self.feat_seq = None
        self.feat_seq_accents = None
        self.duration_weighted = None
        self.chromatic_root = None
        self.diatonic_root = None

    def __iter__(self):
        return self

    def _get_attrs(self):
        """Private function: parse Tune class __dict__ and return any attrs populated with pandas DataFrame objects"""
        _attrs = self.__dict__.items()
        return [val for attr, val in _attrs if isinstance(val, pd.DataFrame)]

    def extract_root(self):

        """
        Populate Tune.chromatic_root and Tune.diatonic_root attrs with integer chromatic and diatonic pitch values
        using key signature information read from input files via music21.
        """

        score = self.score
        # Parse score for music21 Key objects and extract their tonics:
        roots = [key.tonic for key in score.recurse().getElementsByClass(music21.key.Key)]
        # Retain first root note name or set root to NaN if input file does not contain any root data.
        self.chromatic_root = int(roots[0].ps) if roots != [] else 7
        self.diatonic_root = roots[0].diatonicNoteNum if roots != [] else 5

    def convert_music21_streams_to_feature_sequences(self):

        """
        Generator function. Extract primary feature sequence data from music21 Stream representation of a tune.
        Output a list containing midi_note_num, diatonic_note_num, chromatic_pitch_class, offset, duration,
        bar_num, beat_strength and velocity values for
        each note in the tune. Yield a tune-level numpy array containing feature data for entire tune.
        """

        # check that Tune object's score attr exists before proceeding
        target = self.score
        if not isinstance(target, music21.stream.Score):
            print("Tune.music21_streams_to_feature_sequence_converter() input must be a music21 Stream object")
            pass

        # read score content
        score_content = target.flat.recurse().notesAndRests
        # read all notes and extract their pitches represented as MIDI note numbers:
        for idx, note in enumerate(score_content):
            prev_element = score_content[idx - 1]
            if note.isNote:
                midi_note = float(note.pitch.ps)
                diatonic_note_num = float(note.pitch.diatonicNoteNum)
                pitch_class = float(note.pitch.pitchClass)
                beat_strength = round(float(note.beatStrength), ndigits=3) if note.beatStrength else 0
                bar_num = note.measureNumber if note.measureNumber else 0

            # if rests are encountered, take data from previous note
            if note.isRest and prev_element.isNote:
                midi_note = float(prev_element.pitch.ps)
                diatonic_note_num = float(prev_element.pitch.diatonicNoteNum)
                pitch_class = float(prev_element.pitch.pitchClass)
                beat_strength = 0
                bar_num = note.measureNumber

            # if chords are encountered, take their root:
            if note.isChord:
                midi_note = float(note.root().ps)
                diatonic_note_num = float(note.root().diatonicNoteNum)
                pitch_class = float(note.root().pitchClass)
                beat_strength = round(float(note.beatStrength), 3) if note.beatStrength else 0
                bar_num = note.measureNumber

            # for each note, extract primary feature data and store all feature values in a numpy array:
            yield np.asarray([
                midi_note,  # MIDI (chromatic) note number
                diatonic_note_num,  # Diatonic note number
                pitch_class,  # chromatic pitch class
                beat_strength,  # beat strength
                bar_num,    # bar number
                round(float(note.offset) * 2, 2),  # offset
                round(float(note.duration.quarterLength) * 2, 2),  # duration
                0 if note.isRest else note.volume.velocity,  # MIDI velocity
            ])

    def extract_primary_feature_sequences(self):

        """
        Extract primary feature sequences via convert_music21_streams_to_feature_sequences() generator function.
        Format output and store as Tune.feat_seq attr.
        """

        # Add data for each note to 'feat_seq_data' dict:
        feat_seq_data = self.convert_music21_streams_to_feature_sequences()
        # convert to DataFrame
        output = pd.DataFrame(feat_seq_data, columns=["midi_note_num",
                                                      "diatonic_note_num",
                                                      "chromatic_pitch_class",
                                                      "beat_strength",
                                                      "bar_num",
                                                      "offset",
                                                      "duration",
                                                      "velocity"
                                                      ])
        # force types (to save memory):
        output["midi_note_num"] = output["midi_note_num"].astype('int8')
        output["diatonic_note_num"] = output["diatonic_note_num"].astype('int8')
        output["chromatic_pitch_class"] = output["chromatic_pitch_class"].astype('int8')
        output["beat_strength"] = output["beat_strength"].astype('float16').fillna(0).round(decimals=3)
        output["bar_num"] = output["bar_num"].astype('int8')
        output["offset"] = output["offset"].astype('float16')
        output["duration"] = output["duration"].astype('float16')
        output["velocity"] = output["velocity"].fillna(0).astype('int16')
        self.feat_seq = output

    def filter_feat_seq(self, by='velocity', thresh=80):

        """
        Filters Tune.feat_seq feature sequence DataFrame, retaining data for rhythmically-accented
        notes only. Output is stored in a DataFrame as Tune.feat_seq_accents attr.

        Args:
            by -- Select filtering by MIDI velocity (by='velocity') or music21 beatStrength (by='beat_strength')
            thresh -- If using by='beat_strength', set thresh=1 to filter and extract heavily-accented notes
                      (i.e.: the most prominent note in each bar). Set thresh=0.5 to filter and extract accented notes
                      (i.e.: one note per beat).

                      If using by='beat_strength', set thresh=80 to filter and extract heavily-accented notes. Set
                      thresh=105 to filter and extract accented notes.

        In the Irish ABC Notation corpora under investigation, MIDI velocity is used to encode rhythm and metric
        structure following a beat stress model applied in the preliminary conversion from ABC notation to MIDI format
        (via abc_ingest.py.). Accordingly, by must be set to 'velocity' for accurate filtration of feature sequences
        derived from such corpora.

        For all other inputs please use by='beatStrength' method to ensure accurate output.

        If no metric structure information is present in the input data, accent-level filtration cannot be applied via
        either approach.
        """

        target_feature = by
        feat_seq = self.feat_seq
        self.feat_seq_accents = feat_seq[feat_seq[target_feature] > thresh].reset_index()

    def extract_relative_chromatic_pitches(self):

        """
       Extract key-invariant chromatic pitch sequence (pitch relative to the chromatic root, Tune.chromatic_root).
       Applies to both note- and accent-level feature sequences.
        """

        # select all feature sequence DataFrames from Tune attrs
        targets = self._get_attrs()
        # read chromatic root
        chromatic_root = self.chromatic_root
        # calculate scale degrees
        for t in targets:
            t['relative_chromatic_pitch'] = (t['midi_note_num'] - chromatic_root).astype('int8')

    def extract_chromatic_scale_degrees(self):
        """Calculate chromatic scale degree values from note- and accent-level feature sequences."""
        # select all feature sequence DataFrames from Tune attrs:
        targets = self._get_attrs()
        for t in targets:
            t['chromatic_scale_degree'] = (t['relative_chromatic_pitch'] % 12).astype('int8')

    def extract_chromatic_intervals(self):
        """Calculate chromatic interval sequences for note- and accent-level feature sequences."""
        targets = [i for i in self._get_attrs()]
        for t in targets:
            t['chromatic_interval'] = (t['midi_note_num'] - t['midi_note_num'].shift(1)).fillna(0).astype('int8')
            
    def extract_relative_diatonic_pitches(self):

        """
       Extract key-invariant diatonic pitch sequence (pitch relative to the diatonic root, Tune.diatonic_root).
       Applies to both note- and accent-level feature sequences.
        """

        # select all feature sequence DataFrames from Tune instance attrs:
        targets = self._get_attrs()
        diatonic_root = self.diatonic_root
        for t in targets:
            t['relative_diatonic_pitch'] = (t['diatonic_note_num'] - diatonic_root).astype('int8')

    def extract_diatonic_pitch_classes(self):
        """Extract diatonic pitch class sequences for note- and accent-level feature sequences."""
        targets = self._get_attrs()
        for t in targets:
            t["diatonic_pitch_class"] = (t["diatonic_note_num"] % 7).astype('int8')

    def extract_diatonic_intervals(self):
        """Extract diatonic interval sequences for note- and accent-level feature sequences."""
        targets = self._get_attrs()
        for t in targets:
            t['diatonic_interval'] = (t['diatonic_note_num'] - t['diatonic_note_num'].shift(1)).fillna(0).astype('int8')

    def extract_diatonic_scale_degrees(self):
        """Extract diatonic scale degree sequences for note- and accent-level feature sequences."""
        targets = self._get_attrs()
        for t in targets:
            t['diatonic_scale_degree'] = ((t['relative_diatonic_pitch'] % 7) + 1).astype('int8')

    def strip_anacrusis(self):

        """
        Apply heuristic to check if first bar is less than 1/2 the length of second bar.
        If so, first bar is taken as a pickup or anacrusis and removed from note- and accent-level feature sequence data
        """

        # Note: this is a legacy method: anacruses can now be filtered simply by exclusion of data for which 'bar_num'
        # col value == 0.

        targets = self._get_attrs()
        for t in targets:
            # calculate duration of first and second bars
            bar1_duration = t[t['bar_num'] == 1]['duration'].sum()
            bar2_duration = t[t['bar_num'] == 2]['duration'].sum()
            # drop first bar if it is less than half duration of second bar
            t.drop(t[(t['bar_num'] == 0) & (bar1_duration / bar2_duration <= 0.5)].index, inplace=True)

            # reindex bar numbers for tunes from which pickups have been removed
            t['bar_num'] = np.where(t['bar_num'].values[0] == 1, t['bar_num'] - 1, t['bar_num'])

    @staticmethod
    def extract_parsons_codes(*feat_seqs):

        """
        Extract Parsons code sequences for feature sequence DataFrames passed as inputs; add output to feature
        sequence data at note- and accent-levels.
        
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

        for data in feat_seqs:
            prev_note = data['midi_note_num'].shift(1)
            # conditionally assign Parsons codes, comparing each note event with prev note:
            data['parsons_code'] = np.where(data['midi_note_num'] == prev_note, 0, 
                                            np.where(data['midi_note_num'] > prev_note, 1, -1))
            data['parsons_code'] = data['parsons_code'].astype('int8')
            # set first Parsons code value to 0
            data.loc[0, 'parsons_code'] = 0
            data.dropna(inplace=True)

    @staticmethod
    def extract_parsons_cumsum(*feat_seqs):

        """
        Extract cumulative Parsons code sequence values; add output to note- and accent-level feature sequence data.
        """

        for data in feat_seqs:
            # Calculate Parsons code coumsum:
            data['parsons_cumsum'] = data['parsons_code'].cumsum().astype('int8')

    def apply_duration_weighting(self, features=None):

        """
        Calculate duration-weighted sequence for selected features and return output as Tune.duration_weighted attr.

        Duration weighting re-indexes sequence data, converting from feature value per note event to
        one value per eighth note.
        
        Args:
            features -- list of features to include in duration-weighting process.
        """

        target = self.feat_seq
        duration_weighted = {}
        # create new index of eighth notes from 'offsets' column:
        if 'offset' in target.columns:
            offsets = target['offset'].to_numpy()
        else:
            print(f"No offset values encoded for {self.title}: duration-weighting is not possible for this input.")
            return None

        eighth_notes = np.arange(offsets[0], offsets[-1])
        idx = np.searchsorted(offsets, eighth_notes)

        # rescaled feature sequence data to new index
        for feat in features:
            feat_seq = target[feat].to_numpy()
            weighted_seq = np.asarray([feat_seq[i] for i in idx])
            duration_weighted[feat] = weighted_seq

        # return results to 'Tune.duration_weighted' attribute & set type as int16:
        duration_weighted = pd.DataFrame.from_dict(duration_weighted, orient='columns')
        duration_weighted = duration_weighted.rename_axis('eighth_note').astype('int16', errors='ignore')
        duration_weighted.pop('offset')
        self.duration_weighted = duration_weighted


class Corpus:
    __slots__ = [
        'in_path',
        'tunes',
        'pkl_out_path',
        'out_path'
    ]

    """
    A Corpus object is instantiated with a single argument, 'in_path', which is the path to a directory
    containing a minimum of one symbolic music file in any format parseable by the music21 library.

    Attributes:
        in_path -- per above
        tunes -- providing an 'in_path' argument automatically instantiates a Tune class object for
        each tune in the cre_corpus; 'tunes' attr holds a list of these Tune objects.
        out_path -- directory path to create corpus feature sequence data csv files
    """

    def __init__(self, in_path):

        """
        Initialize Corpus object.

        Args:
            in_path -- path to a directory containing a minimum of one symbolic music file in any format parseable by 
            the music21 library
        """

        self.in_path = in_path
        self.tunes = None
        self.out_path = None

    def read_corpus_files_to_music21(self):
        """Create Tune object for every symbolic music file in in_path dir and stores as tunes attr (list)"""
        in_path = self.in_path
        if os.path.isdir(in_path):
            # extract paths to all music files in in_path dir
            filenames = [file for file in os.listdir(in_path) if file.endswith('.mid')]
            # create a Tune object for each file
            self.tunes = [Tune(f"{in_path}/{file}") for file in tqdm(
                filenames, desc="Reading corpus files to Music21 streams")]
        else:
            print("'in_path' must point to a directory of symbolic music files.")
            self.tunes = None
        return self.tunes

    def filter_empty_scores(self):
        """Filter out blank scores before feature sequence calculations."""
        for tune in self.tunes:
            if len(tune.score) == 0:
                print(f"No input data for {tune.title}")
                self.tunes.remove(tune)

    def extract_roots(self):
        """Extract chromatic and diatonic roots for all tunes in the corpus"""
        for tune in tqdm(self.tunes, desc='Extracting chromatic and diatonic roots'):
            tune.extract_root()

    def convert_scores_to_feat_seqs(self, level=None):
        """Extract feature sequence data for all tunes in the corpus"""
        for tune in tqdm(self.tunes, desc=f'Calculating {level}-level feature sequences from music21 scores'):
            tune.extract_primary_feature_sequences(level=level)

    def filter_feat_seq_accents(self, thresh=80, by='beat_strength'):

        """
        Filter Tune.feat_seq to create Tune.feat_seq_accents for all Tune objects in Corpus.tunes.
         User can select whether to filter via velocity [for ABC-originating inputs] or beat strength [for all other
         inputs]. Default is filtration via beat strength.
        
        Args:
            thresh -- filter threshold value
            by -- set to 'velocity' or 'beat_strength' to select filtration method.
        """

        for tune in tqdm(self.tunes, desc=f'Filtering accent-level feature sequence data by MIDI velocity'):
            tune.filter_feat_seq(by=by, thresh=thresh)

    def extract_diatonic_intervals(self):
        """Add diatonic interval sequences to feature sequence data for all tunes in corpus."""
        for tune in tqdm(self.tunes, desc='Calculating diatonic interval sequences'):
            tune.extract_diatonic_intervals()

    def extract_relative_diatonic_pitch_seqs(self):
        """Add relative diatonic pitch sequences to feature sequence data for all tunes in corpus."""
        for tune in tqdm(self.tunes, desc='Calculating relative diatonic pitch sequences'):
            tune.extract_relative_diatonic_pitches()

    def extract_diatonic_pitch_class_seqs(self):
        """Add diatonic pitch class sequences to feature sequence data for all tunes in corpus."""
        for tune in tqdm(self.tunes, desc='Calculating diatonic pitch class sequences'):
            tune.extract_diatonic_pitch_classes()

    def extract_diatonic_scale_degree_seqs(self):
        """Add diatonic scale degree sequences to feature sequence data for all tunes in corpus."""
        for tune in tqdm(self.tunes, desc='Calculating diatonic scale degree sequences'):
            tune.extract_diatonic_scale_degrees()

    def extract_chromatic_intervals(self):
        """Add chromatic interval sequences to feature sequence data for all tunes in corpus."""
        for tune in tqdm(self.tunes, desc='Calculating chromatic interval sequences'):
            tune.extract_chromatic_intervals()

    def extract_relative_chromatic_pitch_seqs(self):
        """Add chromatic pitch sequences for all tunes in corpus."""
        for tune in tqdm(self.tunes, desc='Calculating relative chromatic pitch sequences'):
            tune.extract_relative_chromatic_pitches()

    def extract_chromatic_scale_degree_seqs(self):
        """Add chromatic scale degree sequences to feature sequence data for all tunes in corpus."""
        for tune in tqdm(self.tunes, desc='Calculating chromatic scale degree sequences'):
            tune.extract_chromatic_scale_degrees()

    def extract_parsons_codes(self):
        """Add Parsons code sequences to feature sequence data for all tunes in corpus."""
        for tune in tqdm(self.tunes, desc='Calculating cumulative Parsons code sequences'):
            if tune.feat_seq_accents is not None:
                tune.extract_parsons_codes(tune.feat_seq, tune.feat_seq_accents)
                tune.extract_parsons_cumsum(tune.feat_seq, tune.feat_seq_accents)
            else:
                tune.extract_parsons_codes(tune.feat_seq)
                tune.extract_parsons_cumsum(tune.feat_seq)

    def extract_bar_numbers(self):
        """Add bar numbers to feature sequence data for all tunes in corpus."""
        for tune in tqdm(self.tunes, desc='Adding bar numbers to feature sequence data'):
            tune.extract_bar_nums()

    def extract_duration_weighted_feat_seqs(self, features):

        """
        Run Tune.apply_duration_weighting() to create duration-weighted sequences of selected features for all tunes.

        Args:
            features -- list of musical feature names for which duration-weighted sequences are to be calculated.
        """

        for tune in tqdm(self.tunes, desc='Calculating duration-weighted feature sequences'):
            tune.apply_duration_weighting(features)

    def save_feat_seq_data_to_csv(self):

        """
        Create a subdirectory structure at path specified in Corpus.out_path for each tune in corpus. Write
        DataFrames at Tune.feat_seq, Tune.accent, and Tune.duration_weighted to csv files in the appropriate folders.
        """

        # create subdirectories under location set by Corpus.out_path attr (if they don't already exist):
        # note-level output data
        note_level_results_dir = f"{self.out_path}/note"
        if not os.path.isdir(note_level_results_dir):
            os.makedirs(note_level_results_dir)
        # accent-level output data
        accent_level_results_dir = f"{self.out_path}/accent"
        if not os.path.isdir(accent_level_results_dir):
            os.makedirs(accent_level_results_dir)
        # duration-weighted output data
        duration_weighted_results_dir = f"{self.out_path}/duration_weighted"
        if not os.path.isdir(duration_weighted_results_dir):
            os.makedirs(duration_weighted_results_dir)

        # write data to file:
        for tune in tqdm(self.tunes, desc='Saving feature sequence data to csv'):
            if tune.feat_seq is not None and len(tune.feat_seq) > 1:
                tune.feat_seq.rename_axis('index').astype('int16', errors='ignore')
                tune.feat_seq.to_csv(f"{note_level_results_dir}/{tune.title}.csv")
            if tune.feat_seq_accents is not None and len(tune.feat_seq_accents) > 1:
                tune.feat_seq_accents.to_csv(f"{accent_level_results_dir}/{tune.title}.csv")
            if tune.duration_weighted is not None and len(tune.duration_weighted) > 1:
                tune.duration_weighted.to_csv(f"{duration_weighted_results_dir}/{tune.title}.csv")

    @staticmethod
    def _ingest_factory(in_path):

        """
        Private generator function used in setup_corpus_iteratively(). Iteratively read each tune in corpus and
        extract feature sequence data.
        """

        filetypes = ('.mid', '.krn')

        if os.path.isdir(in_path):
            filenames = [file for file in os.listdir(in_path) if file.endswith(filetypes)]
            for file in tqdm(filenames, desc='Extracting primary feature sequence data'):
                tune = Tune(f"{in_path}/{file}")
                if len(tune.score) != 0:
                    tune.extract_root()
                    tune.extract_primary_feature_sequences()
                    del tune.score
                    yield tune
                else:
                    print(f"No input data for {tune.title}")

    def setup_corpus_iteratively(self):

        """
        Primary corpus ingest method. This is slower but more memory efficient than the alternative
        Corpus.read_corpus_files_to_music21(), and is recommended for large corpora (10k+ files).
        """

        in_path = self.in_path
        tunes = [tune for tune in self._ingest_factory(in_path)]
        self.tunes = tunes
