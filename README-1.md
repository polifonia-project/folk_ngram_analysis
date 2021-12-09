---
id: folk_ngram_analysis
name: FONN - FOlk N-gram aNalysis
brief-description: Work-in-progress on pattern extraction and melodic similarity on a corpus of monophonic Irish folk tunes.
type: Repository
release-date: 8/12/2021
release-number: v0.2-dev
work-package: 
- WP3
licence:  CC By 4.0, https://creativecommons.org/licenses/by/4.0/
links:
- https://github.com/polifonia-project/folk_ngram_analysis
- https://zenodo.org/record/5768216#.YbEAbS2Q3T8
credits:
- https://github.com/danDiamo
- https://github.com/ashahidkhattak
- https://github.com/jmmcd
---

[![DOI](https://zenodo.org/badge/427469033.svg)](https://zenodo.org/badge/latestdoi/427469033)

# FONN - FOlk N-gram aNalysis 

*FONN* repo targets the goals of the Polifonia WP3 i.e., identification of patterns that are useful in detecting relationships between pieces of music, with particular focus on European musical heritage. At present, it includes scripts that make use of n-grams and Damerau-Levenshtein edit distance on monophonic Irish folk tunes.

## In this strand of research we have created three Polifonia components:

1. Folk -gram aNalysis (FONN)
   1. Tools for extraction of feature sequence data from MIDI files -- these resources are available in [./setup_corpus](https://github.com/polifonia-project/folk_ngram_analysis/tree/master/setup_corpus) subfolder. 
   2. Tools to extract, compile, and rank patterns in various musical features such as (chromatic) pitch, pitch class, and interval from musical feature sequence data.
   3. Tools to explore similarity between tunes within the corpus via frequent and similar patterns.
2. Ceol Rince na hÉireann (CRE) MIDI corpus
   1. covering the associated *Ceol Rince na hÉireann* MIDI corpus, please see: [./corpus/readme.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/readme.md).
3. Root Note detection
   1. Covering work-in-progress on automatic detection of musical root for each tune in the corpus, please see: [/.root_key_detection/README.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_key_detection/README.md)


## 1. FONN - Prerequisites 

In ```./setup_corpus/setup_corpus.py``` file, first we need to assign proper path of the corpus to  ``` basepath ``` variable. Then we should have a corpus of folk tunes in MIDI format, and the same should assign to ```inpath``` variable. By default ``` basepath ``` is ```../corpus/. ``` If the corpus is elsewhere, change ```basepath ``` accordingly. The code will be writing outputs to subdirectories of ``` basepath ```. The ```corpus``` should include a ```roots.csv``` file containing a root note (integer between 0 and 11) for each MIDI file.

* Install the following libraries:

``` pip install feather music21 pyarrow fastDamerauLevenshtein ```

* or just:

``` pip install -r requirements.txt ```

## Execution and summary tasks performed by each Scripts

1- The ```./setup_corpus/setup_corpus.py``` script

Running this file will take about 15 minutes. It will produce many csv files under ```<basepath>/feat_seq_data/note```, ```<basepath>/feat_seq_data/accent```, ```<basepath>/feat_seq_data/duration_weighted```. To save time for common situations we will check whether these files exist first, and skip running the code if they do.

* Perform Tasks
  1. Extraction of numeric feature sequences representing pitch, onset, duration, and velocity for all pieces of music in a corpus of monophonic MIDI files.
  2. Deriving secondary feature sequences from the primary features outputted by item above, including interval, key-invariant pitch, pitch class, and melodic contour represented in Parsons code.


2- The ```./setup_ngrams_tfidf.py```, ```./ngram_tfidf_tools.py``` scripts

Again, this script will take about 25 minutes to run, so we check whether the output files already exist before running. For proper execution of this script, you need to set the path of ngram folder in ```./setup_corpus/setup_corpus.py``` file. Below is the relevant code:

```
pitch_class_accents_ngrams_freq_path = basepath + "/ngrams/cre_pitch_class_accents_ngrams_freq.csv"
ngram_inpath = basepath + "/feat_seq_data/accent"
ngram_outpath = basepath + "/ngrams"
ngram_sim_inpath = basepath + "/ngrams/cre_pitch_class_accents_ngrams_tfidf.ftr" # please check
```
* Perform Tasks
  1. Extraction of n-gram patterns for selected musical features(s). So far, work-in-progress has extracted on n-grams for 3 <= n <= 12 on five feature sequences, including melodic contour, interval, pitch, pitch-class interval, and pitch class. 
  2. Initially focusing on accent-level pitch class 6-grams, each melody's unique n-gram patterns are counted and ranked by tf–idf. 

3- The ```./ngram_pattern_search.py``` script
Finally, This script demonstrate some work-in-progress for calculating similarity between tunes, based on similarity between their n-grams. This uses the Damerau-Levenshtein algorithm. The following is the relevant code that need to be set in ```./setup_corpus/setup_corpus.py``` file.

```
ngram_sim_inpath = basepath + "/ngrams/cre_pitch_class_accents_ngrams_tfidf.ftr"
```
* Perform Tasks:
  1. This script ranked collection of patterns can be searched for pattern similarities via the Damerau-Levenshtein edit distance algorithm.
  2. Work-in-progress on identification of similar pieces of music by counting the similar patterns that occur in their accent-level pitch class seqeunces.


## 2. Ceol Rince na hÉireann (CRE) MIDI corpus 

This is Irish dance tunes corpus. It covers the associated *Ceol Rince na hÉireann* MIDI corpus, please see: [./corpus/readme.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/readme.md).
* Highlights:
  * Corpus title: Ceol Rince na hÉireann 
  * Source: Black, B 2020, The Bill Black Irish tune archive homepage, viewed 5 January 2021. 
  * Contents: 1,224 traditional Irish dance tunes, each of which is represented as a monophonic MIDI file.
  
## 3. Root Note detection 
  Covering work-in-progress on automatic detection of musical root for each tune in the corpus, please see: [/.root_key_detection/README.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/README.md).
  This component contains one jupyter notebook script that make use of  ```cre_root_detection.csv``` file. This is expert annotated file and then the script make use of machine learning methods to classify the root note. The root note detection notebook can be accessed using this link [/.root_note_detection/root_note_detection.ipynb](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/root_note_detection.ipynb).
  
