"""
'abc_ingest.py' calls abc2midi command-line tool and uses it to create an individual MIDI file for each tune in a
monophonic ABC Notation corpus.

An ABC Notation cre_corpus is a single file containing scores for one or more tunes in ABC (.abc) format.
For more information on ABC Notation, please see:
https://ifdo.ca/~seymour/runabc/top.html [info on abc2midi packages and versions]
https://abcmidi.sourceforge.io [abc2midi docs]

NOTE: This script requires prior instillation of the abc2midi external dependency. An up-to-date version of
the abc2midi package can be directly downloaded and installed via https://ifdo.ca/~seymour/runabc/abcMIDI-2022.06.14.zip
"""

import os
import shutil

import subprocess


def create_midi_files_from_abc_corpus(abc_dir=None, abc_filename=None, midi_out_dir=None):

    """
    Runs abc2midi to extract an individual MIDI file for every tune in an ABC cre_corpus file.

    Args:
        abc_filename -- Name of ABC Notation corpus file to be processed.
        abc_dir -- Path to directory containing ABC corpus file
        midi_out_dir -- Directory to save MIDI files

    """

    # Run abc2midi:
    print("Running abc2midi command-line tool...")
    create_midi = subprocess.run(
        ["abc2midi", f"{abc_dir}/{abc_filename}", "-t", "-n", "252", "-NGRA", "-Q"],
        cwd=f"{abc_dir}",
        capture_output=True
    )
    # store abc2midi conversion log
    raw_stdout = str(create_midi.stdout)

    # Optionally format and print log
    # formatted_stdout = list(raw_stdout.split("\\n"))
    # for line in formatted_stdout:
    #     print(line)

    # Print any errors and exit status:
    if create_midi.stderr:
        print(create_midi.stderr)
    print('Exit status:', create_midi.returncode)

    # Move MIDI files to midi_out_dir, create dir if it doesn't already exist:
    if not os.path.isdir(midi_out_dir):
        os.makedirs(midi_out_dir)

    file_names = os.listdir(abc_dir)
    for name in file_names:
        if name.endswith('.mid'):
            shutil.move(os.path.join(abc_dir, name), os.path.join(midi_out_dir, name))


def main():

    create_midi_files_from_abc_corpus()


if __name__ == "__main__":
    main()
