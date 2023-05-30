**The Session corpus notes**

Subdirectory '''../thesession_corpus/abc''' contains a data dump of The Session corpus (www.thesession.org)
in ABC Notation format, which was downloaded from https://github.com/adactio/TheSession-data on 13 December 2021.
All other data in '''../thesession_corpus''' dir is derived from this original input.

ABC --> MIDI preprocessing via FoNN.abc_ingest.py
Feature sequence data extracted at note-level, (duration-weighted) note-level and accent-level.
Patterns extracted for 3 <= n <= 12 at both levels.

KG data processed per:
Patterns and pattern locations extracted for 4 <= n <= 6 at  accent-level.
'Pattern corpus' filtered to include only patterns occurring at least twice in the corpus.

NOTE: Due to corpus size, it was not possible to push the outputs above via Git. They are available on request of the
authors. Currently, only the ABC Notation corpus file is provided in the remote version of the corpus.