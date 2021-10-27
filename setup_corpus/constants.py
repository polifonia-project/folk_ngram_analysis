"""
Constants.py contains:

NOTES_NAMES_NUMBERS -- a dictionary, formatted per: note names (keys): lists of corresponding MIDI note numbers (values)

setup_lookup_table() -- a function to convert NOTES_NAMES_NUMBERS constants dict into a Pandas dataframe, allowing
flexible multi-directional lookups in corpus_processing_tools.py root assignment.
"""

import pandas as pd

NOTES_NAMES_NUMBERS = {

    # NOTE: This dict begins with D rather than C, as D is the tonal centre of most instruments and repertoire in the
    # Irish tradition. This can be modified as necessary.

    'D': list(range(2, 109, 12)),
    'D# or E-': list(range(3, 109, 12)),
    'E': list(range(4, 109, 12)),
    'F': list(range(5, 109, 12)),
    'F# or G-': list(range(6, 109, 12)),
    'G': list(range(7, 109, 12)),
    'G# or A-': list(range(8, 109, 12)),
    'A': list(range(9, 109, 12)),
    'A# or B-': list(range(10, 109, 12)),
    'B': list(range(11, 109, 12)),
    'C': list(range(12, 109, 12)),
    'C# or D-': list(range(13, 109, 12))
}

# TODO: Above works for root assignment but will need to add a second, reformatted, table for root detection
#  with separate entries for all sharps and flats.
# TODO: Rename 'fourth_oct_midi_nums' variable.


def setup_lookup_table(data=NOTES_NAMES_NUMBERS):

    note_names = [key for key in data.keys()]
    midi_nums = [(val[5]) for val in data.values()]
    root_nums = [val % 12 for val in midi_nums]
    lookup_data = {
        'note names': note_names,
        'midi num': midi_nums,
        'root num': root_nums
    }

    res = pd.DataFrame.from_dict(lookup_data)
    print("\nSetting up constants.py lookup table for root assignment:")
    print(res.head(), '\n\n')
    return res


def main():

    print(f'\nRunning constants.py\nLookup table:')
    setup_lookup_table()


if __name__ == "__main__":
    main()
else:
    lookup_table = setup_lookup_table()


# TODO: Make numbers_note_names more robust
