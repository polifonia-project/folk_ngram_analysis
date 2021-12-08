"""
Constants.py contains:

MUSIC21_LOOKUP_TABLE -- a dictionary, formatted per: Music21-style note names (keys): chromatic pitch classes (values)

NOTES_NAMES_MIDI_NUMBERS -- a dictionary, formatted per: Music21-style note names (keys): lists of corresponding MIDI note
numbers (values)

setup_lookup_table() -- a function to convert NOTES_NAMES_NUMBERS constants dict into a Pandas dataframe, allowing
flexible multi-directional lookups in corpus_processing_tools.py root assignment.
"""


import pandas as pd

MUSIC21_LOOKUP_TABLE = {

    'C': 0, 'C#': 1, 'D-': 1, 'D': 2, 'D#': 3, 'E-': 3, 'E': 4, 'F': 5, 'F#': 6,
    'G-': 6, 'G': 7, 'G#': 8, 'A-': 8, 'A': 9, 'A#': 10, 'B-': 10, 'B': 11
}

NOTES_NAMES_MIDI_NUMBERS = {

    'C': list(range(0, 109, 12)),
    'C# or D-': list(range(1, 109, 12)),
    'D': list(range(2, 109, 12)),
    'D# or E-': list(range(3, 109, 12)),
    'E': list(range(4, 109, 12)),
    'F': list(range(5, 109, 12)),
    'F# or G-': list(range(6, 109, 12)),
    'G': list(range(7, 109, 12)),
    'G# or A-': list(range(8, 109, 12)),
    'A': list(range(9, 109, 12)),
    'A# or B-': list(range(10, 109, 12)),
    'B': list(range(11, 109, 12))
}


def setup_music21_lookup_table(data=None):

    if data is None:
        data = MUSIC21_LOOKUP_TABLE
    note_names = [key for key in data.keys()]
    pitch_classes = [val for val in data.values()]
    lookup_data = {
        'note name': note_names,
        'pitch class': pitch_classes
    }

    res = pd.DataFrame.from_dict(lookup_data)
    print("\nSetting up Music21 root detection lookup table:")
    print(res.head(), '\n\n')
    return res


def setup_lookup_table(data=None):

    if data is None:
        data = NOTES_NAMES_MIDI_NUMBERS
    note_names = [key for key in data.keys()]
    fourth_oct_midi_nums = [(val[5]) for val in data.values()]
    root_nums = [val % 12 for val in fourth_oct_midi_nums]
    lookup_data = {
        'note names': note_names,
        'midi num': fourth_oct_midi_nums,
        'root num': root_nums
    }

    res = pd.DataFrame.from_dict(lookup_data)
    print("\nSetting up lookup table for root assignment:")
    print(res.head(), '\n\n')
    return res


def main():

    setup_lookup_table()
    setup_music21_lookup_table()


if __name__ == "__main__":
    main()
else:
    lookup_table = setup_lookup_table()
    music21_lookup_table = setup_music21_lookup_table()