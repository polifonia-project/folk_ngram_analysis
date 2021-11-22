---
id: root_key_detection

name: Root Key Detection

brief-description: Work-in-progress on root key detection on a corpus of monophonic Irish folk tunes.

type: Repository

release-date: TBD

release-number: v0.1-dev

work-package: 
- WP3
licence: Public domain, https://unlicense.org
links:
- https://github.com/polifonia-project/folk_ngram_analysis/blob/master/root_key_detection/root_key_detection.py

credits:
- https://github.com/danDiamo
- https://github.com/ashahidkhattak
---

# Root Key Detection

This folder contains the Jupyter Notebook script for the root key detection task. This folder contains two files. The one is Jupyter Notebook script and the other is the .csv file. The file read the CSV files and then perform the following steps:

* 1- Perform Exploratory Data Analysis, such as null value, classes count, box plot, correlations, etc. 

* 2- Create Models such as Decision Tree using Gini and entropy and then computes its classification accuracy

* 3- Create Random Forest models using different hyperparameters for achieving better results.
