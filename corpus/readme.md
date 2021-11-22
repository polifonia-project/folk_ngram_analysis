---
id: cre_corpus
name: Ceol Rince na hÉireann MIDI corpus
brief-description: A corpus of 1,224 monophonic instrumental Irish traditional dance tunes.
type: Corpus
release-date: TBD
release-number: v0.1-dev
work-package: 
- WP3
licence: Public domain, https://unlicense.org
links:
- https://github.com/polifonia-project/folk_ngram_analysis/corpus
- https://github.com/danDiamo/music_ngram_analysis/corpus
- http://www.capeirish.com/webabc
credits:
- https://github.com/danDiamo
- https://github.com/ashahidkhattak
- http://www.capeirish.com/
---

## Corpus title: _Ceol Rince na hÉireann_

**Source:** [Black, B 2020, _The Bill Black Irish tune archive homepage_, viewed 5 January 2021.][1]

**Contents:** 1,224 traditional Irish dance tunes, each of which is represented as a monophonic MIDI file.<br><br>

**About dataset:** 

Between 1963 and 1999, Irish State publishing companies Oifig an tSolatáthair and An Gúm issued five printed volumes of tunes from the collections of Breadán Breathnach (1912-1985) under the series title _Ceol Rince na hÉireann_ (Dance Music of Ireland, hereafter _CRÉ_). The five volumes of _CRÉ_ contain 1,208 traditional tunes, a subset of Breathnach's more extensive personal collection of 5,000+ melodies. The collection has been transcribed into ABC notation by American traditional music researcher Bill Black, and made freely available online via his [personal website][1]. Addition of alternative tune versions and variation in numbering of unique melodies has resulted in a total of 1,224 tunes in the Bill Black ABC corpus. This resource has been used in previous research work, for example it makes up part of a larger aggregated corpus used in the [_Tunepal_][2] Music Information Retrieval app. We have created a new cleaned and annotated MIDI version of the corpus, from which feature sequence data can be extracted and analysed via Polifonia's [FONN][3] music pattern analysis toolkit.<br><br>


**About data collection methodology**

Bill Black's ABC version of the _CRÉ_ collection has been manually edited and annotated, and converted to MIDI. This work included:
* Removal of alternative tune versions, so that the ABC collection more accurately reflects the original print collection.
* Removal of non-valid ABC notation characters.
* Editing of repeat markers to ensure accurate MIDI output.
* Conversion to MIDI via EasyABC software.
* Manual assignment of root note (as chromatic pitch class) for every piece of music in the corpus. This data is stored in the file [roots.csv][4], which is used to derive key-invariant  secondary feature sequence data from the MIDI files.<br><br>


**Description of the data:**

```
corpus/
  -MIDI/
    -1,224 monophonic MIDI files (.mid)
  -roots.csv
  -README.md
  -LICENSE.md

```

Each melody in the corpus is represented as a monophonic MIDI file, named per the melody title. There are 1,224 files in total, stored in the [./MIDI][4] directory. 

The [corpus][6] root directory contains a [roots.csv][5] file, this readme, and a LICENSE.md file.
Roots.csv holds two columns with one row per each MIDI file in the corpus:
'title': MIDI file title
'root': expert-assigned root note of each melody, represented as a [chromatic pitch class][7] (i.e.: An integer value from C=0 through B=11). 

<img width="400" alt="image" src="https://user-images.githubusercontent.com/78231894/142916162-9ace1c42-ceae-412f-95df-98ce34acd359.png">
<br><br>

To extract feature sequence data from the MIDI corpus, please download the corpus data and run [setup_corpus.main()][9] from folk_ngram_analysis component. Please see [folk_ngram_analysis readme][8] for further information.<br><br>


**[Online repository link:][6]**
<br><br>
**Authors:**

* Danny Diamond
<br><br>
**License:**
This project is licensed under the MIT License - see [LICENSE.md][10] file for details
<br><br>

[1]: http://www.capeirish.com/webabc
[2]: https://tunepal.org/index.html
[3]: https://github.com/polifonia-project/folk_ngram_analysis
[4]: https://github.com/polifonia-project/folk_ngram_analysis/tree/master/corpus/MIDI
[5]: https://github.com/danDiamo/music_pattern_analysis/blob/master/corpus/roots.csv
[6]: https://github.com/polifonia-project/folk_ngram_analysis/tree/master/corpus
[7]: https://en.wikipedia.org/wiki/Pitch_class
[8]: https://github.com/polifonia-project/folk_ngram_analysis/blob/master/README.md
[9]: https://github.com/danDiamo/music_pattern_analysis/blob/master/setup_corpus/setup_corpus.py
[10]: https://github.com/polifonia-project/folk_ngram_analysis/blob/master/corpus/readme.md
