---
component-id: folk_ngram_analysis
name: FONN - FOlk N-gram aNalysis
description: Work-in-progress on pattern extraction and melodic similarity tools, with an associated test corpus of monophonic Irish folk tunes.
type: Repository
release-date: 19/05/2022
release-number: v0.5-dev
work-package: 
- WP3
licence:  CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
links:
- https://github.com/polifonia-project/folk_ngram_analysis
- https://zenodo.org/record/5768216#.YbEAbS2Q3T8
credits:
- https://github.com/danDiamo
- https://github.com/ashahidkhattak
- https://github.com/jmmcd
---



# FONN - FOlk _N_-gram aNalysis 

Targetting the goals of Polifonia WP3, FONN contains tools to extract patterns and detect similarity within a monophonic MIDI corpus. The software can be used on any monophonic MIDI corpus, though some of its functionality is tailored to Western European folk music in particular.

The repo contains a fully functional work-in-progress version of the software, along with a cleaned and annotated MIDI test corpus.

In v0.5dev, the FONN toolkit has been comprehensively refactored for speed and memory performance. 
FONN is now capable of ingesting and searching corpora of over 40,000 MIDI files, versus c. 1,000 MIDI files in v0.4dev.
Speed to ingest the 1,200-tune [*CRÉ* test corpus](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/readme.md), extract patterns and run similarity search has decreased from approx. 50 min n v0.4dev to under 5 min in the current release. See [./Demo.ipynb](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/Demo.ipynb) to explore tune similarity in the *CRÉ* corpus.

NOTE: Deliverable 3.3 of the Polifonia project describes the context and research in more detail. It will be published on [Cordis](https://cordis.europa.eu/project/id/101004746/it).

## FONN -- Polifonia components:

1. **Pattern and Similarity Toolkit**
   * 1.1. Tools for extraction of feature sequence data from MIDI files.
   * 1.2. Tools to extract, compile, and rank patterns in various musical features such as (chromatic) pitch, pitch class, and interval from musical feature sequence data. 
   * 1.3. Tools to explore similarity between tunes within the corpus via frequent and similar patterns. These resources are available in [root](https://github.com/polifonia-project/folk_ngram_analysis/tree/master/) folder. 
2. **Ceol Rince na hÉireann (CRÉ) MIDI corpus**
   * 2.1. For the associated *Ceol Rince na hÉireann* MIDI corpus of 1,224 monophonic Irish traditional dance tunes, please see: [./corpus/readme.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/readme.md).
3. **Root Note Detection**
   * 3.1. Work-in-progress on automatic detection of musical root for each tune in the corpus, please see: [/.root_key_detection/README.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/README.md)


## FONN - Requirements

To ensure FONN runs correctly, please install the following libraries:

``` pip install fastDamerauLevenshtein music21 numpy pandas tqdm ```

...or just:

``` pip install -r requirements.txt ```

## 1. Pattern and Similarity Toolkit - running the tools

1.1. **Reading a MIDI corpus: setup_corpus.py**

Running ```./setup_corpus/setup_corpus.py``` converts a monophonic MIDI corpus to feature sequence representation.
By default this conversion script points to the provided test MIDI corpus at ```./corpus/MIDI``` and outputs feature sequence data to ```./corpus/feat_seq_corpus```, but these paths can be edited in ```./setup_corpus/setup_corpus.py``` main function .

NOTE: ``./corpus``` should include a ```roots.csv``` file containing the root note of each MIDI file, represented as a chromatic [pitch class](https://en.wikipedia.org/wiki/Pitch_class) (an integer between 0 and 11). This is necessary to calculate secondary key-invariant feature sequences. A a ```roots.csv``` file is provided for the test corpus, and such a file must be provided for any other corpus on which the tools are to be used.

1.2. **Extracting patterns: pattern_extraction.py**

Running ```pattern_extraction.py``` extracts all patterns which occur at least once in the corpus for a target musical feature (i.e. 'pitch', 'interval', 'pitch class', etc) at a selectable range of pattern lengths and level of graunlarity. A table is compiled, frequency and TF-IDF values are calculated, and the output is written to file under ```./corpus/pattern_corpus/```. 
By default, running this file will extract all 3-7 item accent-level key-invariant pitch class patterns from the test corpus. 
The default target fesature, pattern length, and level of granularity can all be changed in ```pattern_extraction.py``` main function.

NOTE: 'accent-level' above referes to a higher level of granularity, in which the feature sequenc data is filtered to retain only data for note-events which occur on accented beats.

1.3. **Finding similar tunes: similarity_search.py**

Running ```similarity_search.py``` firstly extracts significant pattern(s) as ranked by TF-IDF from a user-selectable candidate tune.
Using these patterns as search terms, similar patterns are identified across the corpus via the Damerau-Levenshtein local alignment algorithm. 

Tunes in the corpus which contain a high number of similar patterns are returned in a simple results table, displaying a count of similar patterns per tune. This table is written to csv. The effectiveness of a local pattern count as a metric of tune similarity is currently undergoing quantitative testing, evaluation, and methodological tuning. This release of the FONN Pattern and Similarity Toolkit provides a working version of the core methodology, whihc will be refined and expanded over the course of the Polifonia project.

By default, ```Lord McDonald's (reel)``` is set as the search candidate tune, but this can be changed in ```similarity_search.py``` main function.

NOTE: For a stepwise worked example, please see the included [demo](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/Demo.ipynb) notebook.

## 2. Ceol Rince na hÉireann (CRÉ) MIDI corpus 

A new MIDI version of the existing *Ceol Rince na hÉireann* corpus of 1,224 monophonic Irish dance tunes. Please see: [./corpus/readme.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/readme.md).
* Highlights:
  * Corpus title: _Ceol Rince na hÉireann_
  * Source: Black, B 2020, [The Bill Black Irish tune archive homepage](http://www.capeirish.com/webabc), viewed 5 January 2021.
  * Contents: 1,224 traditional Irish dance tunes, each of which is represented as a monophonic MIDI file. Also included is roots.csv, a file giving the root note for every file in the corpus as a chromatic pitch class in integer notation.
  
## 3. Root Note detection 
Work-in-progress on automatic detection of musical root for each tune in the corpus. Please see: [/.root_key_detection/README.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/README.md).
  This component contains one jupyter notebook script that makes use of  ```cre_root_detection.csv```, which is an expert-annotated file containing pitch class values assigned to each piece of music in the corpus by a variety of root-detection metrics. From this input, the script makes use of machine learning methods to classify the root note. The root note detection notebook can be accessed using this link: [/.root_note_detection/root_note_detection.ipynb](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/root_note_detection.ipynb).
  
##  Attribution

[![DOI](https://zenodo.org/badge/427469033.svg)](https://zenodo.org/badge/latestdoi/427469033)

If you use the code in this repository, please cite this software as follow: 
```
@software{diamond_fonn_2022,
	address = {Galway, Ireland},
	title = {{FONN} - {FOlk} {N}-gram {aNalysis}},
	shorttitle = {{FONN}},
	url = {https://github.com/polifonia-project/folk_ngram_analysis},
	publisher = {National University of Ireland, Galway},
	author = {Diamond, Danny and Shahid, Abdul and McDermott, James},
	year = {2022},
}
```

## License
This work is licensed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
