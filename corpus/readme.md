## Corpus title: _Ceol Rince na hÉireann_

**Source:** [Black, B 2020, _The Bill Black Irish tune archive homepage_, viewed 5 January 2021.][1]

**Contents:** 1,224 traditional Irish dance tunes, each of which is represented as a monophonic MIDI file.<br><br>

**Breathnach's _Ceol Rince na hÉireann_:** 

Between 1963 and 1999, Irish State publishing companies Oifig an tSolatáthair and An Gúm issued five printed volumes of tunes from the collections of Breadán Breathnach under the series title _Ceol Rince na hÉireann_ (Dance Music of Ireland). Breathnach (1912-1985) was a State-funded independent folk music collector, researcher, and publisher: his work was crucial in evolving the study of Irish traditional music into an academic discipline, and central to the establishment  of key institutions in the field, the _Irish Traditional Music Archive / Taisce Cheol Dúchais Éireann_ and _Na Piobairí Uilleann_.

The five volumes of _Ceol Rince na hÉireann_ (hereafter _CRÉ_) contain 1,208 traditional tunes, a subset of Breathnach's more extensive personal collection of 5,000+ melodies, which contains material from a variety of sources including private and commercial recordings; manuscript collections; and tune versions transcribed directly from prominent traditional musicians of Breathnach's era.

Breathnach's collecting concentrated on materials sourced from musical communities in Dublin city, along with counties Clare and Galway, and the _Sliabh Luachra_ region of Southwestern Ireland. Despite the relative lack of coverage of material from the north of the island, the CRÉ series is generally regarded as the key canonical print collection of the mid-late 20th century in the Irish instrumental tradition, documenting the revival of the tradition from its lowest ebb in the 1950s to mainstream prominence by the time of Breathnach's death in 1985.<br><br>

**Conversion to ABC notation by Bill Black:** The CRÉ collection has been transcribed into ABC notation by America traditional music researcher Bill Black, and made freely available online via the link above. This resource has been used in previous research work, for example it makes up part of a larger aggregated corpus used in the [_Tunepal_][2] Music Information Retrieval app.

During conversion to ABC notation, alternative versions of tunes in the collection were not treated in the same manner as in the print collection, and additional alternative versions were appended in some cases. Due to these changes the online collection comprises 1,224 rather than 1,208 individual tunes.<br><br>


**Conversion to annotated MIDI corpus for analysis in Polifonia:**

The collection has been manually edited and annotated, and converted to MIDI for use as an input corpus for Polifonia pattern analysis tools. This work included:
* Removal of alternative tune versions, so that the ABC collection more accurately reflects the original print collection.
* Removal of non-valid ABC notation characters.
* Editing of repeat markers to ensure accurate MIDI output.
*Conversion to MIDI via EasyABC software.
* Manual assignment of root note (as chromatic pitch class) for every piece of music in the corpus. This step generated the file [roots.csv][3], which is essential to deriving key-invariant feature sequence data from the MIDI files.<br><br>

**Extracting feature sequence data from corpus:**

To extract feature sequence data from the MIDI corpus, please download the corpus and run [setup_corpus.main()][4]. Please see module docstrings for further information.

[1]: http://www.capeirish.com/webabc
[2]: https://tunepal.org/index.html
[3]: https://github.com/danDiamo/music_pattern_analysis/blob/master/corpus/roots.csv
[4]: https://github.com/danDiamo/music_pattern_analysis/blob/master/setup_corpus/setup_corpus.py



