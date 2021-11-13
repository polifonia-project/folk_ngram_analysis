---
id: folk_ngram_analysis
name: FONN - FOlk N-gram aNalysis
brief-description: Work-in-progress on pattern extraction and melodic similarity on a corpus of monophonic Irish folk tunes.
type: Repository
release-date: TBD
release-number: v0.1-dev
work-package: 
- WP3
licence: Public domain, https://unlicense.org
links:
- https://github.com/polifonia-project/folk_ngram_analysis
- https://github.com/danDiamo/music_ngram_analysis
credits:
- https://github.com/danDiamo
- https://github.com/ashahidkhattak
---

# FONN - FOlk N-gram aNalysis 

FONN repo targets the goals of the Polifonia WP3 i.e., identification of patterns that are useful in detecting relationships between pieces of music, with particular focus on Europeean musical heritage. At present, it includes scripts that make use of n-grams and Damerau-Levenshtein edit distance on monophonic Irish folk tunes.

This document covers component 1, comprising all code tools in the folk_ngrams_analysis repo.
For component 2 documentation, covering the associated MIDI input corpus, please see: [./corpus/readme.md] (https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/readme.md)

Following is the summary of each task, along with relevant file:

* 1- Extraction of numeric feature sequences representing pitch, onset, duration, and velocity for all pieces of music in a corpus of monophonic MIDI files.
Relevant file: [./setup_corpus/setup_corpus.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/setup_corpus/setup_corpus.py)
NOTE: It is necessary to set the input/output paths of the relevant files/folders to run the script.

* 2- Deriving secondary feature sequences from the primary features outputted by item 1 above, including interval, key-invariant pitch, pitch class, and melodic contour represented in Parsons code.
Relevant file: [./setup_corpus/setup_corpus.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/setup_corpus/setup_corpus.py)
NOTE: It is necessary to set the input/output paths of the relevant files/folders to run the script.

* 3- Extraction of n-gram patterns for selected musical features(s). So far, work-in-progress has extracted on n-grams for 3 <= n <= 12 on five feature sequences, including melodic contour, interval, pitch, pitch-class interval, and pitch class. 
Relevant file: [./setup_ngrams_tfidf.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/setup_ngrams_tfidf.py)
NOTE: It is necessary to set the input/output paths of the relevant files/folders to run the script.

* 5- Initially focusing on accent-level pitch class 6-grams, each melody's unique n-gram patterns are counted and ranked by tf–idf. 
Relevant file: [./ngram_tfidf_tools.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/ngram_tfidf_tools.py)

* 6- This ranked collection of patterns can be searched for pattern similarities via the Damerau-Levenshtein edit distance algorithm.
Relevant file: [./ngram_pattern_search.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/ngram_pattern_search.py)
NOTE: It is necessary to set variables in ngram_pattern_search.main() to run the script: please see docstring for further information.

* 7- Work-in-progress on identification of similar pieces of music by counting the similar patterns that occur in their accent-level pitch class seqeunces.
Relevant file: [./ngram_pattern_search.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/ngram_pattern_search.py) 
NOTE: It is necessary to set variables in ngram_pattern_search.main() to run the script: please see docstring for further information.
