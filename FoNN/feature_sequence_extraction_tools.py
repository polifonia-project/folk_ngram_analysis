# TODO: Check Corpus docstrings

"""
Feature sequence data represents each note in a symbolic music document via numerical feature values, such
as: (midi_note_num: 68, onset: 10, duration: 2)
.
Tune class objects represent a single tune in feature sequence format.
Corpus class objects represent collections of Tune objects.
Together, these classes allow extraction of primary and secondary feature sequence data from MIDI files.

Primary feature sequences calculable via Tune and Corpus classes, with feature names as used throughout the FoNN
toolkit:

-- 'midi_note_num': Chromatic pitch represented as MIDI number
-- 'onset': note onset (1/8 notes)
-- 'duration': note (1/8 notes)
-- 'velocity': MIDI velocity

Secondary feature sequences calculable via Tune and Corpus classes:

-- 'diatonic_note_num': Diatonic pitch
-- 'chromatic_pitch_class': Pitch normalised to a single octave, represented as an integer between 0-11.
-- 'bar_count': Bar number
-- 'relative_chromatic_pitch': Chromatic pitch relative to the root note or tonal centre of the input sequence.
-- 'relative_diatonic_pitch': Diatonic pitch relative to the root note or tonal centre of the input sequence.
-- 'chromatic_scale_degree': Chromatic pitch class relative to the root note or tonal centre of the input sequence.
-- 'diatonic_scale_degree': Diatonic pitch class relative to the root note or tonal centre of the input sequence.
-- 'chromatic_interval': Change in chromatic pitch between two successive notes in the input sequence
-- 'diatonic_interval': Change in diatonic pitch between two successive notes in the input sequence
-- 'parsons_code': simple melodic contour. Please see Tune.extract_parsons_codes() docstring for detailed explanation.
-- 'parsons_cumsum': cumulative Parsons code values.

These classes also allow:
-- Extraction of the 'root' or tonal centre from a MIDI input.
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
    Tune class objects represent a single tune in feature sequence format. A Tune object can be instantiated
    via the 'inpath' argument, which must point to a single MIDI file. Tune objects can be created individually or
    can be automatically instantiated in bulk at corpus-level when a Corpus object is instantiated.

    Attributes:

    inpath -- path to a monophonic MIDI file.
    title -- Tune title extracted from MIDI filename; populates automatically on instantiation.
    score -- music21 Stream representation of the music data contained in the MIDI file; populates automatically on
    instantiation.
    score_accents -- A filtered accent-level version of 'score' attr.
    feat_seq -- Feature sequence representation of the data held in 'score' attr.
    feat_seq_accents -- Accent-level feature sequence data
    duration_weighted -- 'duration-weighted' sequences for selected features. Duration-weighting is
    explained below.
    chromatic_root -- Chromatic note number representing root or tonal centre of input sequence.
    diatonic_root -- Diatonic note number representing root or tonal centre of input sequence.
    """

    def __init__(self, inpath: str):

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
        self.chromatic_root = None
        self.diatonic_root = None

    def __iter__(self):
        return self

    def _get_attrs(self):

        """Private function to parse Tune class __dict__ and return any attrs populated with pandas DataFrame objects"""
        _attrs = self.__dict__.items()
        return [val for attr, val in _attrs if isinstance(val, pd.DataFrame)]

    def extract_root(self):
        """Populates chromatic_root and diatonic_root atts using key signature information read from MIDI via music21"""
        score = self.score
        # Parse score for music21 Key objects and extract their tonics:
        roots = [key.tonic for key in score.recurse().getElementsByClass(music21.key.Key)]
        # Retain first root note name or set root to NaN if MIDI file does not contain any root data.
        self.chromatic_root = int(roots[0].ps) if roots != [] else 7
        self.diatonic_root = roots[0].diatonicNoteNum if roots != [] else 5

    def convert_music21_streams_to_feature_sequences(self):

        """
        Generator function. Extracts primary feature sequence data from music21 Stream representation of a tune.
        Outputs a pandas DataFrame containing midi_note_num, diatonic_note_num, onset, duration, velocity values for
        each note in the tune. Outputs is stored in numpy array.
        """

        # check that Tune object's score attr exists before proceeding
        target = self.score
        if not isinstance(target, music21.stream.Score):
            print("Tune.music21_streams_to_feature_sequence_converter() input must be a music21 Stream object")
            pass

        # read score content
        score_content = target.recurse().notes
        # score_content = target.recurse().notesAndRests
        # read all notes and extract their pitches represented as MIDI note numbers:
        for idx, note in enumerate(score_content):
            # prev_element = score_content[idx - 1]
            if note.isNote:
                midi_note = float(note.pitch.ps)
                diatonic_note_num = float(note.pitch.diatonicNoteNum)
                pitch_class = float(note.pitch.pitchClass)

            # # if rests are encountered, copy pitch data from previous note
            # if note.isRest and prev_element.isNote:
            #     midi_note = float(prev_element.pitch.ps)
            #     diatonic_note_num = float(prev_element.pitch.diatonicNoteNum)
            #     pitch_class = float(prev_element.pitch.pitchClass)

            # if chords are encountered, take their root: (this is rare in our corpora)
            elif note.isChord:
                midi_note = float(note.root().ps)
                diatonic_note_num = float(note.root().diatonicNoteNum)
                pitch_class = float(note.root().pitchClass)

            # if note.isRest and prev_element.isChord:
            #     midi_note = float(prev_element.root().ps)
            #     diatonic_note_num = float(prev_element.root().diatonicNoteNum)
            #     pitch_class = float(prev_element.root().pitchClass)

            # for each note, extract primary feature data and store all feature values in a numpy array:
            yield np.asarray([
                midi_note,  # MIDI (chromatic) note number
                diatonic_note_num,  # Diatonic note number
                pitch_class,  # chromatic pitch class
                round(float(note.offset) * 2, 2),  # onset
                round(float(note.duration.quarterLength) * 2, 2),  # duration
                note.volume.velocity if not note.isRest else 0,  # MIDI velocity
            ])

    def extract_primary_feature_sequences(self):

        """Extracts primary feature sequences via convert_music21_streams_to_feature_sequences() generator function.
        Formats output and stores in pandas DataFrame at feat_seq attr."""

        # Add data for each note to 'feat_seq_data' dict:
        feat_seq_data = self.convert_music21_streams_to_feature_sequences()
        # convert to DataFrame
        output = pd.DataFrame(feat_seq_data, columns=["midi_note_num",
                                                      "diatonic_note_num",
                                                      "chromatic_pitch_class",
                                                      "onset",
                                                      "duration",
                                                      "velocity"
                                                      ])
        # force types (to save memory):
        output["midi_note_num"] = output["midi_note_num"].astype('int8')
        output["diatonic_note_num"] = output["diatonic_note_num"].astype('int8')
        output["chromatic_pitch_class"] = output["chromatic_pitch_class"].astype('int8')
        output["onset"] = output["onset"].astype('float16')
        output["duration"] = output["duration"].astype('float16')
        output["velocity"] = output["velocity"].astype('int16')
        self.feat_seq = output

    def filter_score_by_beat_strength(self, thresh=0.5):

        """Filters Tune.score attr via an adjustable beat strength threshold, output is stored as music21 Stream at
        score_accents attr.

        args:
            thresh -- threshold music21 Note.beatStrength attr value by which the score is filtered. Only Note objects
            with beatStrength >= thresh will be retained in the filtered output.

        To filter and extract heavily-accented notes (i.e.: the most prominent note in each bar) from a score,
        set thresh=1
        To filter and extract accented notes (i.e.: one note per beat) from a score, set thresh=0.5

        NOTE: Scores can also be filtered after conversion to feature sequence format via
        filter_feat_seq_by_velocity() method. The alternative method is faster and less memory-intensive, but will
        only filter accurately on MIDI data originating from ABC Notation files.
        For MIDI inputs please use this method to ensure accurate output.
        """

        score = self.score
        filtered_score = music21.stream.Score()
        for idx, note in enumerate(score.recurse().notes):
            if note.isNote and float(note.beatStrength) >= thresh:
                filtered_score.append(note)
        self.score_accents = filtered_score

    def filter_feat_seq_by_velocity(self, thresh=80):

        """
        Filters feat_seq feature sequence DataFrame, retaining data for rhythmically-accented
        notes only. Filtered output is stored in a DataFrame at feat_seq_accents attr. 

        In the ABC Notation corpora under investigation, MIDI velocity is used to encode rhythm and structure following 
        a beat stress model applied in the preliminary conversion from ABC notation to MIDI format via abc_ingest.py.
        In this beat stress model, all notes with MIDI velocity values above 80 are rhythmically-accented.

        Args:
            thresh -- filter threshold MIDI velocity value
        """
        feat_seq = self.feat_seq
        # Filter feature sequence data via MIDI velocity threshold:
        self.feat_seq_accents = feat_seq[feat_seq['velocity'] > thresh].reset_index()

    def extract_relative_chromatic_pitches(self):

        """
        Uses chromatic root value assigned by Corpus.assign_roots() method to extract key-invariant chromatic pitch
        sequence of pitches relative to the root. Applies to bot note- and accent-level feature sequence DataFrames.
        """

        # select all feature sequence DataFrames from Tune instance attrs:
        targets = self._get_attrs()
        chromatic_root = self.chromatic_root
        for t in targets:
            t['relative_chromatic_pitch'] = (t['midi_note_num'] - chromatic_root).astype('int8')

    def extract_chromatic_scale_degrees(self):

        """
        Derives chromatic scale degree values from relative pitch sequence for note- and accent-level feature
        sequence DataFrames.
        """

        # select all feature sequence DataFrames from Tune instance attrs:
        targets = self._get_attrs()
        for t in targets:
            t['chromatic_scale_degree'] = (t['relative_chromatic_pitch'] % 12).astype('int8')

    def extract_chromatic_intervals(self):
        """Extracts chromatic interval sequences for note- and accent-level feature sequence DataFrames."""
        targets = [i for i in self._get_attrs()]
        for t in targets:
            t['chromatic_interval'] = (t['midi_note_num'] - t['midi_note_num'].shift(1)).fillna(0).astype('int8')
            
    def extract_relative_diatonic_pitches(self):

        """
        Uses diatonic root value assigned by Corpus.assign_roots() method to extract key-invariant diatonic pitch 
        sequences for note- and accent-level feature sequence DataFrames..
        """

        # select all feature sequence DataFrames from Tune instance attrs:
        targets = self._get_attrs()
        diatonic_root = self.diatonic_root
        for t in targets:
            t['relative_diatonic_pitch'] = (t['diatonic_note_num'] - diatonic_root).astype('int8')

    def extract_diatonic_pitch_classes(self):
        """Extracts diatonic pitch class sequences for note- and accent-level feature sequence DataFrames."""
        targets = self._get_attrs()
        for t in targets:
            t["diatonic_pitch_class"] = (t["diatonic_note_num"] % 7).astype('int8')

    def extract_diatonic_intervals(self):
        """Extracts diatonic interval sequences for note- and accent-level feature sequence DataFrames."""
        targets = self._get_attrs()
        for t in targets:
            t['diatonic_interval'] = (t['diatonic_note_num'] - t['diatonic_note_num'].shift(1)).fillna(0).astype('int8')

    def extract_diatonic_scale_degrees(self):
        """Extracts diatonic scale degree sequences for note- and accent-level feature sequence DataFrames."""
        targets = self._get_attrs()
        for t in targets:
            t['diatonic_scale_degree'] = ((t['relative_diatonic_pitch'] % 7) + 1).astype('int8')

    def extract_bar_count(self):
        """Adds bar numbers to note- and accent-level feature sequence DataFrames."""
        # Note: Due to its reliance on ABC Notation beat stress modelling, this method only works on IDI files outputted
        # by abc_ingest.py 
        targets = self._get_attrs()
        for t in targets:
            # create bar counter column via MIDI velocity values:
            bar_lines = np.where(np.logical_and(t['velocity'] == 105, t.index > 0), True, False)
            t['bar_count'] = bar_lines.cumsum().astype('int16')

    def strip_anacrusis(self):

        """
        Applies heuristic to check if first bar is less than 1/2 the length of second bar.
        If so, first bar is taken as a pickup or anacrusis and removed from note- and accent-level feature sequence data
        """

        targets = self._get_attrs()
        for t in targets:
            # calculate duration of first and second bars
            bar1_duration = t[t['bar_count'] == 0]['duration'].sum()
            bar2_duration = t[t['bar_count'] == 1]['duration'].sum()
            t.drop(t[(t['bar_count'] == 0) & (bar1_duration / bar2_duration <= 0.5)].index, inplace=True)

            # reindex bar numbers for tunes from which pickups have been removed
            t['bar_count'] = np.where(t['bar_count'].values[0] == 1, t['bar_count'] - 1, t['bar_count'])

    @staticmethod
    def extract_parsons_codes(*feat_seqs):

        """
        Extracts Parsons code sequences for feature sequence DataFrames passed as inputs; adds output as new 
        Dataframe columns.
        
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
        Extracts cumulative Parsons code sequence values;
        adds output to note- and accent-level feature sequence DataFrames.
        """

        for data in feat_seqs:
            # Calculate Parsons code coumsum:
            data['parsons_cumsum'] = data['parsons_code'].cumsum().astype('int8')

    def apply_duration_weighting(self, features: list = None):

        """
        Extracts duration-weighted sequence for selected features and returns output to pandas DataFrame at
        duration_weighted attr.

        Duration weighting re-indexes sequence data, converting from feature value per note event to
        one value per eighth note.
        
        Args:
            features -- list of features to include in duration-weighting process.
        """

        target = self.feat_seq
        duration_weighted = {}
        # create new index of eighth notes from 'onsets' column:
        onsets = target['onset'].to_numpy()
        eighth_notes = np.arange(onsets[0], onsets[-1])
        idx = np.searchsorted(onsets, eighth_notes)

        # rescaled feature sequence data to new index
        for feat in features:
            feat_seq = target[feat].to_numpy()
            weighted_seq = np.asarray([feat_seq[i] for i in idx])
            duration_weighted[feat] = weighted_seq

        # return results to 'Tune.duration_weighted' attribute & set type as int16:
        duration_weighted = pd.DataFrame.from_dict(duration_weighted, orient='columns')
        duration_weighted = duration_weighted.rename_axis('eighth_note').astype('int16', errors='ignore')
        duration_weighted.pop('onset')
        self.duration_weighted = duration_weighted


class Corpus:
    __slots__ = [
        'inpath',
        'tunes',
        'pkl_outpath',
        'csv_outpath'
    ]

    """
    A Corpus object is instantiated with a single argument, 'inpath', which is the path to a directory
    containing monophonic MIDI files.

    Attributes:
        inpath -- per above
        tunes -- providing an 'inpath' argument automatically instantiates a Tune class object for
        each tune in the cre_corpus; 'tunes' attr holds a list of these Tune objects.
        csv_outpath -- directory path to create corpus feature sequence data csv files
    """

    def __init__(self, inpath):

        """
        Initializes Corpus object.

        Args:
            inpath -- path to a directory of monophonic MIDI files.
        """

        self.inpath = inpath
        self.tunes = None
        self.pkl_outpath = None
        self.csv_outpath = None

    def read_all_midi_files_to_music21(self):
        """Creates Tune object for every MIDI file in inpath dir and stores them in list at tunes attr"""
        inpath = self.inpath
        if os.path.isdir(inpath):
            # extract paths to all MIDI files in inpath dir
            filenames = [file for file in os.listdir(inpath) if file.endswith('.mid')]
            # create a Tune object for each MIDI file
            self.tunes = [Tune(f"{inpath}/{file}") for file in tqdm(
                filenames, desc="Reading MIDI files to Music21 streams")]
        else:
            print("'inpath' must point to a directory of MIDI files.")
            self.tunes = None
        return self.tunes

    def filter_empty_scores(self):

        """The Session cre_corpus contains some 'experimental' pieces of music which are simply blank scores.
        This method filters out such scores before feature sequence calculations.
        """

        for tune in self.tunes:
            if len(tune.score) == 0:
                print(f"No input data for {tune.title}")
                self.tunes.remove(tune)

    def extract_roots(self):
        """Runs Tune.extract_root() to extract chromatic and diatonic roots for all tunes in the corpus"""
        for tune in tqdm(self.tunes, desc='Extracting chromatic and diatonic roots'):
            tune.extract_root()

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
            tune.filter_score_by_beat_strength(thresh)

    def convert_scores_to_feat_seqs(self, level=None):

        """
        Runs Tune.extract_primary_feature_sequences() to extract feature sequence data for all Tune objects in
        Corpus.tunes
        """

        for tune in tqdm(self.tunes, desc=f'Calculating {level}-level feature sequences from music21 scores'):
            tune.extract_primary_feature_sequences(level=level)

    def filter_feat_seq_accents(self, thresh=80, by=None):

        """
        For all Tune objects in Corpus.tunes, this method filters Tune.feat_seq to create Tune.feat_seq_accents
        attr. User can select whether to filter via velocity [for ABC inputs] or 
        beat strength [for MIDI inputs]. Default is via velocity.
        
        Args:
            thresh -- filter threshold value
            by -- set to 'velocity' or 'beat_strength' to select filtration method."""

        if by == 'velocity':
            for tune in tqdm(self.tunes, desc=f'Filtering accent-level feature sequence data by MIDI velocity'):
                tune.filter_feat_seq_by_velocity(thresh=thresh)
        elif by == 'beat_strength':
            for tune in tqdm(self.tunes, desc=f'Filtering accent-level feature sequence data by music21 beatStrength'):
                tune.filter_score_by_beat_strength(thresh=thresh)

    def extract_diatonic_intervals(self):
        
        """Runs Tune.extract_diatonic_intervals() to add diatonic interval sequences to feature sequence data for every 
        Tune object in Corpus.tunes."""
        
        for tune in tqdm(self.tunes, desc='Calculating diatonic interval sequences'):
            tune.extract_diatonic_intervals()

    def extract_relative_diatonic_pitch_seqs(self):

        """Runs Tune.extract_relative_diatonic_pitches() to add relative diatonic pitch sequences to feature sequence data
        for every Tune object in Corpus.tunes."""

        for tune in tqdm(self.tunes, desc='Calculating relative diatonic pitch sequences'):
            tune.extract_relative_diatonic_pitches()

    def extract_diatonic_pitch_class_seqs(self):

        """Runs Tune.extract_diatonic_pitch_classes() to add diatonic pitch class sequences to feature sequence data for
        all Tune objects in Corpus.tunes."""
        
        for tune in tqdm(self.tunes, desc='Calculating diatonic pitch class sequences'):
            tune.extract_diatonic_pitch_classes()

    def extract_diatonic_scale_degree_seqs(self):

        """Runs Tune.extract_diatonic_scale_degrees() to add diatonic scale degree sequences to feature sequence data for 
                all Tune objects in Corpus.tunes."""
        
        for tune in tqdm(self.tunes, desc='Calculating diatonic scale degree sequences'):
            tune.extract_diatonic_scale_degrees()

    def extract_chromatic_intervals(self):
        """Runs Tune.extract_chromatic_intervals() to add chromatic interval sequences to feature sequence data for 
                all Tune objects in Corpus.tunes."""
        for tune in tqdm(self.tunes, desc='Calculating chromatic interval sequences'):
            tune.extract_chromatic_intervals()

    def extract_relative_chromatic_pitch_seqs(self):
        """Runs Tune.extract_relative_chromatic_pitches() for all Tune objects in Corpus.tunes."""
        for tune in tqdm(self.tunes, desc='Calculating relative chromatic pitch sequences'):
            tune.extract_relative_chromatic_pitches()

    def extract_chromatic_scale_degree_seqs(self):
         
        """Runs Tune.extract_chromatic_scale_degrees() to add chromatic scale degree sequences to feature sequence data for
                all Tune objects in Corpus.tunes."""
         
        for tune in tqdm(self.tunes, desc='Calculating chromatic scale degree sequences'):
            tune.extract_chromatic_scale_degrees()

    def extract_parsons_codes(self):
        """Applies Tune.extract_parsons_codes() and Tune.extract_parsons_cumsum() to all Tune objects in Corpus.tunes"""
        for tune in tqdm(self.tunes, desc='Calculating cumulative Parsons code sequences'):
            if tune.feat_seq_accents is not None:
                tune.extract_parsons_codes(tune.feat_seq, tune.feat_seq_accents)
                tune.extract_parsons_cumsum(tune.feat_seq, tune.feat_seq_accents)
            else:
                tune.extract_parsons_codes(tune.feat_seq)
                tune.extract_parsons_cumsum(tune.feat_seq)

    def extract_bar_numbers(self):
        """Runs Tune.extract_bar_count() to add bar numbers to feature sequence data for all Tune objects in Corpus.tunes"""
        for tune in tqdm(self.tunes, desc='Adding bar numbers to feature sequence data'):
            tune.extract_bar_count()

    def strip_anacruses(self):

        """Runs Tune.strip_anacruses() to remove anacruses from feature sequence data
         for all Tune objects in Corpus.tunes"""

        return [tune.strip_anacrusis() for tune in tqdm(self.tunes, desc='Removing anacruses (pick-up measures)')]

    def extract_duration_weighted_feat_seqs(self, features):

        """
        Runs Tune.apply_duration_weighting() for all Tune objects in Corpus.tunes.

        Args:
            features -- list of musical features for which
            duration-weighted sequences are to be calculated.
        """

        for tune in tqdm(self.tunes, desc='Calculating duration-weighted feature sequences'):
            tune.apply_duration_weighting(features)

    def save_feat_seq_data_to_csv(self):

        """For all Tune objects in Corpus.tunes, this method creates a subdirectory structure at the path specified in
        Corpus.csv_outpath. It then writes the DataFrames at Tune.feat_seq, Tune.feat_seq_accents, and
        Tune.duration_weighted to csv files in the appropriate folders."""

        # create subdirectories under location set by Corpus.csv_outpath attr (if they don't already exist):
        # note-level output data
        note_level_results_dir = f"{self.csv_outpath}/feat_seq_note"
        if not os.path.isdir(note_level_results_dir):
            os.makedirs(note_level_results_dir)
        # accent-level output data
        accent_level_results_dir = f"{self.csv_outpath}/feat_seq_accents"
        if not os.path.isdir(accent_level_results_dir):
            os.makedirs(accent_level_results_dir)
        # duration-weighted output data
        duration_weighted_results_dir = f"{self.csv_outpath}/feat_seq_duration_weighted"
        if not os.path.isdir(duration_weighted_results_dir):
            os.makedirs(duration_weighted_results_dir)

        # write data to file:
        for tune in tqdm(self.tunes, desc='Saving feature sequence data to csv'):
            if tune.feat_seq is not None and len(tune.feat_seq) > 2:
                tune.feat_seq.rename_axis('index').astype('int16', errors='ignore')
                tune.feat_seq.to_csv(f"{note_level_results_dir}/{tune.title}.csv")
            if tune.feat_seq_accents is not None and len(tune.feat_seq_accents) > 2:
                tune.feat_seq_accents.to_csv(f"{accent_level_results_dir}/{tune.title}.csv")
            if tune.duration_weighted is not None and len(tune.duration_weighted) > 2:
                tune.duration_weighted.to_csv(f"{duration_weighted_results_dir}/{tune.title}.csv")

    @staticmethod
    def midi_ingest_factory(inpath):

        """Generator function used in setup_corpus_iteratively() to read MIDI data and extract feature sequence data
        iteratively from each file in corpus"""

        if os.path.isdir(inpath):
            filenames = [file for file in os.listdir(inpath) if file.endswith('.mid')]
            for file in tqdm(filenames, desc='Extracting primary feature sequence data from MIDI'):
                tune = Tune(f"{inpath}/{file}")
                if len(tune.score) != 0:
                    tune.extract_root()
                    tune.extract_primary_feature_sequences()
                    del tune.score
                    yield tune
                else:
                    print(f"No input data for {tune.title}")

    def setup_corpus_iteratively(self):

        """Alternative corpus ingest method to read_all_midi_files_to_music21(). This is slower but more memory
        efficient and is recommended for large corpora (10k+ MIDI files)."""

        inpath = self.inpath
        tunes = [tune for tune in self.midi_ingest_factory(inpath)]
        self.tunes = tunes
