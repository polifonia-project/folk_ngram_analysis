"""
'abc_ingest.py' calls abc2MIDI command-line tool and uses it to create an individual MIDI file for each tune in a
monophonic ABC Notation corpus. This preprocessing step allows FONN feature sequence extraction and
pattern similarity tools to be used on ABC format input corpora.

An ABC Notation corpus is a single file containing scores for one or more tunes in ABC (.abc) format.
For more information on ABC Notation, please see:

https://ifdo.ca/~seymour/runabc/top.html [info on abc2MIDI packages and versions]
https://abcmidi.sourceforge.io [abc2MIDI docs]

NOTE: This script requires prior instillation of the abc2MIDI external dependency. An up-to-date version of
the abc2MIDI package can be directly downloaded and installed via https://ifdo.ca/~seymour/runabc/abcMIDI-2022.06.14.zip
"""

import os
import shutil

import subprocess


def create_midi_corpus_from_abc(abc_filename='CRE_clean.abc'):

    """
    Runs abc2MIDI to extract an individual MIDI file for every tune in an ABC corpus file stored in './corpus/abc' dir.
    Output MIDI files are saved to '.corpus/MIDI' dir.

    Args:
        abc_filename -- Name of ABC corpus file to be processed. Default is 'CRE_clean.abc' which points to the included
        'Ceol Rince na hEireann' sample corpus. If working with external data, copy ABC corpus to './corpus/abc' dir and
        assign 'abc_filename' to file name.
    """

    # Run abc2MIDI:
    print("Running abc2MIDI command-line tool to covert ABC Notation corpus to MIDI format...")
    create_midi = subprocess.run(["abc2midi", abc_filename, "-t", "-n", "100"], cwd="./corpus/abc", capture_output=True)
    # Reformat and print output
    raw_stdout = str(create_midi.stdout)
    formatted_stdout = list(raw_stdout.split("\\n"))
    for line in formatted_stdout:
        print(line)
    # Print any errors and exit status:
    if create_midi.stderr:
        print(create_midi.stderr)
    print('exit status:', create_midi.returncode)

    # create "./corpus/midi" dir:
    midi_path = "./corpus/midi"
    abc_path = "./corpus/abc"
    # Move MIDI files to "./corpus/midi" dir:
    if not os.path.isdir(midi_path):
        os.makedirs(midi_path)
    else:
        file_names = os.listdir(abc_path)
        for name in file_names:
            if name.endswith('.mid'):
                shutil.move(os.path.join(abc_path, name), os.path.join(midi_path, name))


def main():

    create_midi_corpus_from_abc()


if __name__ == "__main__":
    main()
