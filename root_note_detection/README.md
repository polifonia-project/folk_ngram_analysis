---
component-id: root_note_detection

name: Root Note Detection

brief-description: Work-in-progress on root note detection on a corpus of monophonic Irish folk tunes.

type: Repository

release-date: 19/05/2022

release-number: v0.5-dev

work-package: 
- WP3
licence: CC By 4.0, https://creativecommons.org/licenses/by/4.0/
links:
- https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_note_detection/root_note_detection.ipynb
- https://zenodo.org/record/5768216#.YbEAbS2Q3T8

credits:
- https://github.com/danDiamo
- https://github.com/ashahidkhattak
---

# Root Note Detection

This folder contains the Jupyter Notebook script for the root note detection task. This folder contains two files. The one is Jupyter Notebook script and the other is the ```cre_root_detection.csv``` file. The jupyter file read the CSV file and then perform the following steps:

* 1- Performed Exploratory Data Analysis, such as null value, classes count, box plot, correlations, etc. 
* 2- Global settings are defined to control feature selection
* 3- Multiple dataset are created for model development and its evaluation
* 4- Minority classes are balanced with help of SMOTE
* 5- Classification report of state-of-the-art models for root note detection are generated for comparison
* 6- Factorial design experimental setup is developed to evaluate different classification algorithms such as  Decision Tree, RandomForest, NaiveBayes 
* 7- The best models are selected, and finally they are compared with SOA models.
