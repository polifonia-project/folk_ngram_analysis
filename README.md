---
component-id: folk_ngram_analysis
name: FONN - FOlk N-gram aNalysis
description: Work-in-progress on pattern extraction and melodic similarity tools, with an associated test corpus of monophonic Irish folk tunes.
type: Repository
release-date: 15/06/2022
release-number: v0.7-dev
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

Targetting the goals of Polifonia WP3, FONN contains tools to extract patterns and detect similarity within a monophonic music corpus. Although some of FONN's functionality is tailored to Western European folk music in particular, the software can be used on any monophonic corpora in MIDI or ABC Notation formats.

The repo contains a fully functional work-in-progress version of the software, along with a cleaned and annotated test corpus, which is available in both ABC Notation and MIDI formats at [./corpus](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/).

In v0.5dev, the FONN toolkit has been comprehensively refactored for speed and memory performance. 
FONN is now capable of ingesting and searching corpora of over 40,000 MIDI files, versus c. 1,000 MIDI files in v0.4dev.
Speed to ingest the 1,200-tune test corpus, extract patterns and run similarity search has decreased from approx. 50 min n v0.4dev to under 5 min in the current release.<br> 
In addition to MIDI, FONN now can also ingest ABC Notation corpora.

See [./fonn_demo.ipynb](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/fonn_demo.ipynb) to explore pattern extraction and tune similarity in the supplied *Ceol Rince a hÉireann (CRÉ)* corpus using the FONN toolkit.

See [./corpus/corpus_demo.ipynb](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/corpus_demo.ipynb) to explore the *CRÉ* corpus data.

NOTE: Deliverable 3.3 of the Polifonia project describes the context and research in more detail. It will be published on [Cordis](https://cordis.europa.eu/project/id/101004746/it).


## FONN -- Polifonia components:

1. **FONN - FOlk _N_-gram aNalysis**
   * 1.1. Tools for extraction of feature sequence data and root detection metrics from MIDI files.
   * 1.2. Tools to extract, compile, and rank patterns in various musical features such as (chromatic) pitch, pitch class, and interval from musical feature sequence data. 
   * 1.3. Tools to explore similarity between tunes within the corpus via frequent and similar patterns. 
   
NOTE: 1.1-1.3 are available in [root](https://github.com/polifonia-project/folk_ngram_analysis/tree/master/) folder. 
   
2. **Ceol Rince na hÉireann (CRÉ) corpus**
   * 2.1. For the associated *Ceol Rince na hÉireann* corpus of 1,195 monophonic Irish traditional dance tunes in ABC and MIDI formats, please see: [./corpus/readme.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/readme.md).
3. **Root Note Detection**
   * 3.1. Work-in-progress on automatic detection of musical root for each tune in the corpus, please see: [/.root_key_detection/README.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/README.md)


## FONN - Requirements

To ensure FONN runs correctly, please install the following libraries:

``` pip install fastDamerauLevenshtein music21 numpy pandas tqdm ```

...or just:

``` pip install -r requirements.txt ```


## FONN - preliminary setup for ABC corpora

NOTE: The *CRÉ* corpus is provided in both ABC Notation and MIDI formats in [./corpus](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/) subdirectory, along with a demo Jupyter notebook and README.

- To ingest a corpus in ABC Notation format, first install the abc2MIDI external dependency, which can be downloaded directly [here](https://ifdo.ca/~seymour/runabc/abcMIDI-2022.06.14.zip). For information on abc2MIDI, please see the project [documentation](https://abcmidi.sourceforge.io).

- Convert from ABC Notation to MIDI by running the ```./abc_ingest.py``` script. This preliminary step uses abc2MIDI to encode a specific 'beat model' into the MIDI output, which is used later in the workflow to filter data for rhythmically-accented notes. Such higher-level data is of particular interest in the study of Irish and related European & North American folk musics.

- Workflow from here onwards is the same for both ABC and MIDI corpora. It is detailed below and illustrated stepwise in the [./fonn_demo.ipynb](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/fonn_demo.ipynb) Jupyter notebook. 


## 1. FONN - FOlk _N_-gram aNalysis: running the tools

1.1. **Reading a MIDI corpus: setup_corpus.py**

- Running ```./setup_corpus.py``` converts a monophonic MIDI corpus to feature sequence representation.
By default this conversion script reads the *CRÉ* MIDI corpus at ```./corpus/MIDI``` and outputs feature sequence data to ```./corpus/feat_seq_corpus```. Input and output paths can be edited if desired via ```./setup_corpus/setup_corpus.py``` main function.

- ```./corpus``` must include a ```roots.csv``` file containing the root note of each MIDI file, represented as a chromatic [pitch class](https://en.wikipedia.org/wiki/Pitch_class) (an integer between 0 and 11). This is necessary to calculate secondary key-invariant feature sequences later in the workflow. An expert-annotated ```roots.csv``` file is provided for the test corpus.

- By default, ```./setup_corpus.py``` also writes root note detection metrics to ```./root_note_detection_metrics.csv```. Although root note detection is not necessary for the supplied *CRÉ* corpus due to the provided expert-annotated roots.csv table, this provides sample the input for component 3, Root Note Detection. Please see [./root_note_detection/README.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/README.md) for further information. 


1.2. **Extracting patterns: pattern_extraction.py**

- Running ```pattern_extraction.py``` extracts all patterns which occur at least once in the corpus for a target musical feature (i.e. 'pitch', 'interval', 'pitch class', etc) at a user-selectable range of pattern lengths and level of graunlarity. A table is compiled, frequency and TF-IDF values are calculated, and the output is written to file in ```./corpus/pattern_corpus/``` subdirectory.

- By default, running this file will extract all 3-7 item accent-level key-invariant pitch class patterns from the test corpus. 
The default target fesature, pattern length, and level of granularity can all be changed in ```pattern_extraction.py``` main function.

- NOTE: 'accent-level' above refers to higher level music data, in which the feature sequenc data is filtered to retain only data for notes which occur on accented beats.


1.3. **Finding similar tunes: similarity_search.py**

- Running ```similarity_search.py``` firstly extracts significant pattern(s) as ranked by TF-IDF from a user-selectable candidate tune.
Using these patterns as search terms, similar patterns are identified across the corpus via the Damerau-Levenshtein local alignment algorithm. 

- Tunes in the corpus which contain a high number of similar patterns are returned in a simple results table, displaying a count of similar patterns per tune. This table is written to csv. The effectiveness of a local pattern count as a metric of tune similarity is currently undergoing quantitative testing, evaluation, and methodological tuning. This release of the FONN Pattern and Similarity Toolkit provides a working version of the core methodology, whihc will be refined and expanded over the course of the Polifonia project.

By default, ```LordMcDonaldsreel``` is set as the search candidate tune, but this can be changed in ```similarity_search.py``` main function.



## 2. Ceol Rince na hÉireann (CRÉ) MIDI corpus 

- A new version of the previously-existing *Ceol Rince na hÉireann* corpus, containing 1,195 monophonic Irish traditional dance tunes. the corpus in provided in ABC Notation and in MIDI. Please see: [./corpus/readme.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/README.md) for more information.

* Highlights:
  * Corpus title: _Ceol Rince na hÉireann_
  * Source: Black, B 2020, [The Bill Black Irish tune archive homepage](http://www.capeirish.com/webabc), viewed 5 January 2021.
  * Contents: 1,195 traditional Irish dance tunes, each of which is represented as a monophonic MIDI file. Also included is ```roots.csv```, a file giving the expert-annotated root note for every file in the corpus as a chromatic integer pitch class.
  
## 3. Root Note Detection 


Work-in-progress on automatic detection of musical root for each tune in the corpus. Please see: [/.root_key_detection/README.md](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/README.md).
  This component contains a jupyter notebook script that makes use of  ```cre_root_detection.csv```, which is a file containing pitch class values assigned to each piece of music in the corpus by the above-mentioned root-detection metrics outputted by ```setup_corpus.py```. From this input data, the script makes use of machine learning methods to classify the root note. The root note detection notebook can be accessed at [/.root_note_detection/root_note_detection.ipynb](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/root_note_detection.ipynb).
  
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
