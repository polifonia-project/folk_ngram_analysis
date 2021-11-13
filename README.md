# FONN - FOlk N-gram aNalysis 

FONN repo targets the goals of the Polifonia WP3 i.e., identification of patterns that are useful in detecting relationships between pieces of music, with particular focus on Europe’s musical heritage. At present, it includes scripts that make use of sequence mining approaches on monophonic Irish folk tunes.

Following is the summary of each task alognwith relevant file

* 1- The corpus must first be translated from ABC notation to MIDI using EasyABC 1.3.7 as a preliminary step. After we obtain the MIDI corpus, we may extract the numeric feature sequences that reflect pitch, onset, duration, and velocity. 
Relevant File: [./setup_corpus/setup_corpus.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/setup_corpus/setup_corpus.py)
NOTE: You need to set up the paths of the relevant files/folders for taking the input and getting the output.

* 2- The included scripts help you to derive secondary feature sequences from these primary features, including interval, key-invariant pitch, pitch class, pitch-class interval, inter-onset interval, bar number, and simple melodic contour represented in Parsons code.
Relevant File: [./setup_corpus/setup_corpus.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/setup_corpus/setup_corpus.py)
NOTE: You need to set up the paths of the relevant files/folders for taking the input and getting the output.

	
* 3- Pattern extraction is another function of this component, which is accomplished using the PrefixSpan sequence mining technique. So far, the emphasis has been on n-grams (N-grams for 3 <= n <= 12) on five feature sequences, including melodic contour, interval, pitch, pitch-class interval, and pitch class. 
Relevant File: [./setup_ngrams_tfidf.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/setup_ngrams_tfidf.py)
NOTE: You need to setup the paths of the relevant files/folders for taking the input and getting the output

* 5- Initially focusing on accent-level pitch class 6-grams, each melody's distinctive n-gram patterns are counted and ranked by tf–idf. 
Relevant File: [./ngram_tfidf_tools.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/ngram_tfidf_tools.py)


* 6- This ranking collection of patterns can be used to search for pattern similarities. The scripts perform geometric distance measurements (Cosine, Euclidean), compression and edit distance algorithms.
Relevant File: [./ngram_pattern_search.py](https://github.com/polifonia-project/folk_ngram_analysis/blob/master/ngram_pattern_search.py)
